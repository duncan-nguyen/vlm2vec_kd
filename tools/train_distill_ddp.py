# Run directly from the repo root, e.g. `torchrun tools/train_distill_ddp.py`: put the repo root on
# sys.path so `import src.…` resolves without installing the project.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser

from src import profiling
from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.criterions import build_criterion
from src.distiller import DistillationCollator, DistillationDataset, Distiller
from src.utils import print_rank

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Todo


def get_optimizer_params(model, training_args):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters


def get_optimizer(model, training_args):
    while isinstance(model, DDP):
        model = model.module
    optimizer_grouped_parameters = get_optimizer_params(model, training_args)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    return optimizer


def prepare_dataset(data_args, model_args):
    dataset = DistillationDataset(data_args, model_args)
    return dataset


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def to_device(obj, device, non_blocking=True):
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        # non_blocking only has an effect for pinned source memory (which the
        # DataLoader now provides); it is a no-op otherwise, and the copy is
        # ordered against the following kernels on the same stream either way.
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [to_device(v, device, non_blocking) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    else:
        if hasattr(obj, "to") and callable(obj.to):
            return obj.to(device)
        return obj


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


class Trainer:
    def __init__(
        self,
        distiller,
        train_data,
        optimizer,
        lr_scheduler,
        criterion,
        model_args,
        training_args,
    ):
        print_rank("Initializing Trainer...")
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.distiller = distiller.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.model_args = model_args
        self.training_args = training_args

        # broadcast_buffers: the teacher is frozen and the student has no
        # running stats, so the per-forward buffer broadcast is pure overhead.
        # gradient_as_bucket_view: lets DDP reuse the reduction buckets as the
        # .grad storage instead of keeping a second copy.
        self.distiller = DDP(
            self.distiller,
            device_ids=[self.gpu_id],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    def _debug_batch_devices(self, obj, prefix=""):
        if obj is None:
            print(f"{prefix}Value: None")
            return

        try:
            if isinstance(obj, torch.Tensor):
                print(f"{prefix}Tensor device: {obj.device}, shape: {obj.shape}")
            elif isinstance(obj, dict):
                if len(obj) == 0:
                    print(f"{prefix}Empty dict")
                for k, v in obj.items():
                    self._debug_batch_devices(v, prefix=f"{prefix}{k}.")
            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    print(f"{prefix}Empty {type(obj).__name__}")
                for i, v in enumerate(obj):
                    self._debug_batch_devices(v, prefix=f"{prefix}[{i}].")
            else:
                print(f"{prefix}Type: {type(obj).__name__}, Value: {obj}")
        except Exception as e:
            print(f"{prefix}ERROR: {e}")

    LOSS_KEYS = (
        "loss",
        "span_loss",
        "contrastive_loss",
        "kd_loss_rkd",
        "cross_modal_loss",
        "kd_loss_dtw",
    )

    def _as_scalar(self, value):
        if isinstance(value, torch.Tensor):
            return (
                value.detach().to(device=self.device, dtype=torch.float32).reshape(())
            )
        return torch.tensor(float(value), device=self.device, dtype=torch.float32)

    def run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        grad_accum = self.training_args.gradient_accumulation_steps
        logging_steps = max(1, int(self.training_args.logging_steps))

        # Loss bookkeeping is accumulated on the GPU. Calling .item() on every
        # component of every micro-batch forces a host sync per call and stalls
        # the pipeline; we pull the running means back once per logging step.
        running = torch.zeros(
            len(self.LOSS_KEYS), device=self.device, dtype=torch.float32
        )
        running_n = 0
        opt_step = 0

        progress_bar = tqdm(
            total=len(self.train_data.dataset)
            // self.training_args.per_device_train_batch_size
            // self.training_args.gradient_accumulation_steps
            // dist.get_world_size(),
            desc=f"Epoch {epoch}",
            disable=not dist.get_rank() == 0,
        )
        for batch_idx, batch in enumerate(
            profiling.timed_iter(self.train_data, "data_wait")
        ):
            with profiling.section("to_device"):
                batch = to_device(batch, self.device)

            # Skip the DDP all-reduce on every micro-batch but the last one of
            # an accumulation window; otherwise gradients are reduced
            # grad_accum times per optimizer step for no reason.
            is_sync_step = (batch_idx + 1) % grad_accum == 0
            sync_ctx = nullcontext() if is_sync_step else self.distiller.no_sync()

            with sync_ctx:
                with profiling.section("forward"):
                    loss_dict = self.distiller(self.criterion, batch)
                    loss = loss_dict["loss"] / grad_accum

                with profiling.section("bookkeeping"):
                    running += torch.stack(
                        [self._as_scalar(loss_dict.get(k, 0.0)) for k in self.LOSS_KEYS]
                    )
                    running_n += 1

                with profiling.section("backward"):
                    loss.backward()

            if is_sync_step:
                with profiling.section("optimizer"):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                opt_step += 1
                profiling.profiler.step()

                if is_main_process():
                    progress_bar.update(1)
                    if opt_step % logging_steps == 0:
                        # Single device->host transfer for all components.
                        means = (running / max(running_n, 1)).tolist()
                        stats = dict(zip(self.LOSS_KEYS, means))
                        progress_bar.set_postfix(
                            {
                                "loss": f"{stats['loss']:.4f}",
                                "kd_loss": f"{stats['span_loss']:.4f}",
                                "contrastive_loss": f"{stats['contrastive_loss']:.4f}",
                                "kd_rkd_loss": f"{stats['kd_loss_rkd']:.4f}",
                                "cross_modal_loss": f"{stats['cross_modal_loss']:.4f}",
                                "kd_dtw_loss": f"{stats['kd_loss_dtw']:.4f}",
                                "lr": f"{self.lr_scheduler.get_last_lr()[0]:.6f}",
                            }
                        )

                if profiling.profiler.should_report and is_main_process():
                    print(profiling.profiler.report(f"epoch {epoch}"), flush=True)
                if profiling.profiler.should_stop:
                    print_rank(
                        f"VLM2VEC_PROFILE_STEPS reached "
                        f"({profiling.profiler.n_steps} steps), stopping epoch early."
                    )
                    break

        progress_bar.close()
        if profiling.profiler.enabled and is_main_process():
            print(profiling.profiler.report(f"epoch {epoch} final"), flush=True)

    def train(self):
        for epoch in range(self.training_args.num_train_epochs):
            self.run_epoch(epoch)
            if is_main_process() and self.training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(
                    self.training_args.output_dir, f"checkpoint-epoch-{epoch}"
                )
                projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
                os.makedirs(ckpt_dir, exist_ok=True)

                student = self.distiller.module.student
                student.encoder.save_pretrained(ckpt_dir)
                if self.model_args.model_backbone in [
                    "llava_onevision",
                    "llava_two_vision",
                ]:
                    torch.save(
                        student.encoder.model.multi_modal_projector.state_dict(),
                        projector_dir,
                    )
                else:
                    torch.save(
                        student.encoder.model.model.mm_projector.state_dict(),
                        projector_dir,
                    )

                student_config = (
                    AutoConfig.from_pretrained(self.model_args.model_name)
                    if self.model_args.model_name
                    else None
                )
                tokenizer = (
                    AutoTokenizer.from_pretrained(self.model_args.model_name)
                    if self.model_args.model_name
                    else None
                )
                if student_config:
                    student_config.save_pretrained(ckpt_dir)
                if tokenizer:
                    tokenizer.save_pretrained(ckpt_dir)
                try:
                    processor = (
                        AutoProcessor.from_pretrained(self.model_args.model_name)
                        if self.model_args.model_name
                        else None
                    )
                    if processor:
                        processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning: Could not save processor: {e}")
                print_rank(f"Saved checkpoint to {ckpt_dir}")

                ## for evaluation
                print(f"Start evaluating student at epoch {epoch}")
            dist.barrier()

        if is_main_process():
            final_ckpt_dir = os.path.join(
                self.training_args.output_dir, "checkpoint-final"
            )
            projector_dir = os.path.join(final_ckpt_dir, "mm_projector.pth")
            os.makedirs(final_ckpt_dir, exist_ok=True)
            student = self.distiller.module.student
            student.encoder.save_pretrained(final_ckpt_dir)
            if self.model_args.model_backbone in [
                "llava_onevision",
                "llava_two_vision",
            ]:
                torch.save(
                    student.encoder.model.multi_modal_projector.state_dict(),
                    projector_dir,
                )
            else:
                torch.save(
                    student.encoder.model.model.mm_projector.state_dict(), projector_dir
                )
            student_config = (
                AutoConfig.from_pretrained(self.model_args.model_name)
                if self.model_args.model_name
                else None
            )
            tokenizer = (
                AutoTokenizer.from_pretrained(self.model_args.model_name)
                if self.model_args.model_name
                else None
            )
            if student_config:
                student_config.save_pretrained(final_ckpt_dir)
            if tokenizer:
                tokenizer.save_pretrained(final_ckpt_dir)
            try:
                processor = (
                    AutoProcessor.from_pretrained(self.model_args.model_name)
                    if self.model_args.model_name
                    else None
                )
                if processor:
                    processor.save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Warning: Could not save processor: {e}")
            print_rank(f"Saved final model to {final_ckpt_dir}")
        dist.barrier()


def main():
    for arg in sys.argv:
        if arg.startswith("--local_rank"):
            local_rank = int(arg.split("=")[-1])
            sys.argv.remove(arg)
            sys.argv.append("--local_rank")
            sys.argv.append(f"{local_rank}")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    distiller = Distiller(model_args, training_args, data_args=data_args)
    train_dataset = prepare_dataset(data_args, model_args)
    dist_sampler = DistributedSampler(train_dataset, shuffle=True)
    for n, p in distiller.student.named_parameters():
        if p.requires_grad:  # thường chỉ là LoRA
            p.data = p.data.to(torch.bfloat16)

    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    # The collator runs four processor passes per batch (student/teacher x
    # qry/pos), each looping over samples to decode, resize and tokenize. With
    # num_workers=0 all of that ran on the main process and the GPU sat idle
    # waiting for it. Overlap it with compute instead.
    num_workers = training_args.dataloader_num_workers
    dataloader_kwargs = dict(
        batch_size=training_args.per_device_train_batch_size,
        sampler=dist_sampler,
        collate_fn=collator,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = (
            training_args.dataloader_prefetch_factor or 4
        )
    print_rank(
        f"DataLoader: num_workers={num_workers}, "
        f"pin_memory={dataloader_kwargs['pin_memory']}, "
        f"prefetch_factor={dataloader_kwargs.get('prefetch_factor')}"
    )
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    num_trainable_vision = 0
    for n, p in distiller.student.named_parameters():
        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True

        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True

        if "lm_head" in n:
            p.requires_grad = False
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
            num_trainable_vision += p.numel()
    print_rank(f"Number of trainable vision parameters: {num_trainable_vision}")

    optimizer = AdamW(
        distiller.student.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    print(f"Len of train dataset: {len(train_dataloader.dataset)}")
    total_steps = (
        len(train_dataloader.dataset)
        // (training_args.per_device_train_batch_size * dist.get_world_size())
        // training_args.gradient_accumulation_steps
    ) * training_args.num_train_epochs
    if model_args.projector_config_path is not None:
        optimizer = distiller.add_optimizer_param_group(optimizer)

    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in optimizer.param_groups[0]["params"] if p.requires_grad),
    )

    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    else:
        from transformers import get_constant_schedule_with_warmup

        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
        )
    criterion = build_criterion(training_args)
    trainer = Trainer(
        distiller,
        train_dataloader,
        optimizer,
        lr_scheduler,
        criterion,
        model_args,
        training_args,
    )
    trainer.train()


if __name__ == "__main__":
    ddp_setup()
    main()
    destroy_process_group()
