"""Argument parsing, model/optimizer setup and launch, shared by the entrypoints.

`tools/train_distill_ddp.py` and `tools/train_distill_no_deepspeed.py` are now
three lines each on top of :func:`run_training`.
"""

import os

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.criterions import build_criterion, get_spec
from src.distiller import DistillationCollator, DistillationDataset, Distiller
from src.training.dataloader import build_train_dataloader
from src.training.loop import DistillTrainer, world_size
from src.training.sam import build_sharpness_aware
from src.utils import print_master

# The HF tokenizers' own thread pool deadlocks after a fork, and the DataLoader
# workers are forks. Set before anything imports a fast tokenizer.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()


def setup_distributed():
    """Join the process group when launched under torchrun; no-op otherwise."""
    if "LOCAL_RANK" not in os.environ:
        return False
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
    return True


def prepare_trainable_parameters(student):
    """Decide what trains, and cast exactly that to bf16.

    The multimodal projector is trained alongside the LoRA adapters; the LM head
    is not, because an embedding model never reads logits and its gradient is
    the single largest tensor in the backward pass.
    """
    trainable = 0
    for name, param in student.named_parameters():
        if "mm_projector" in name or "multi_modal_projector" in name:
            param.requires_grad = True
        if "lm_head" in name:
            param.requires_grad = False
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
            trainable += param.numel()
    print_master(f"Trainable student parameters: {trainable}")
    return trainable


def attach_criterion(distiller, criterion):
    """Give a criterion that owns weights the same treatment as the projectors.

    Most methods have none and nothing happens. One that does -- TALAS builds a
    projection per anchored layer, sized from a flag rather than from a
    projector config -- needs its parameters registered on the `Distiller`
    before `DistillTrainer` moves it to the device and wraps it in DDP, or they
    stay on the CPU in fp32 and are never synchronised.
    """
    criterion.build_parameters(distiller)
    if any(p.requires_grad for p in criterion.parameters()):
        # Assigning an nn.Module attribute registers it as a submodule.
        distiller.criterion = criterion
        trainable = sum(p.numel() for p in criterion.parameters() if p.requires_grad)
        print_master(f"Criterion owns {trainable} trainable parameters")
    return criterion


def build_optimizer(distiller, model_args, training_args):
    """AdamW over the trainable parameters only.

    Handing AdamW every parameter of a frozen 0.5B backbone -- which both
    entrypoints did -- makes it walk the full parameter list on every step and
    makes `param_groups[0]` a misleading thing to report.
    """
    optimizer = AdamW(
        [p for p in distiller.student.parameters() if p.requires_grad],
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    if model_args.projector_config_path is not None:
        optimizer = distiller.add_optimizer_param_group(optimizer)
    # A criterion's own projections are trained like the KD projectors are, at
    # --projector_lr rather than at the student's learning rate. Read off
    # model_args, which is where the flag actually lives; `add_optimizer_param_group`
    # looks for it on training_args and so always falls back to --learning_rate.
    criterion = getattr(distiller, "criterion", None)
    if criterion is not None:
        params = [p for p in criterion.parameters() if p.requires_grad]
        if params:
            lr = getattr(model_args, "projector_lr", None) or training_args.learning_rate
            optimizer.add_param_group({"params": params, "lr": lr})
            print_master(f"Criterion parameters added to optimizer at lr {lr}")
    return optimizer


def _warmup_steps(training_args, total_steps):
    """Warmup length, from whichever of the two flags this transformers has.

    `warmup_ratio` was removed in transformers 5.x; the repo pins 4.56.1, but a
    silently-zero warmup on a box with a newer version is the kind of difference
    that shows up as an unexplained accuracy gap rather than an error.
    """
    steps = getattr(training_args, "warmup_steps", 0) or 0
    if steps:
        return int(steps)
    ratio = getattr(training_args, "warmup_ratio", None)
    if ratio is None:
        print_master(
            "Warning: this transformers version exposes neither --warmup_ratio "
            "nor --warmup_steps; training with no warmup."
        )
        return 0
    return int(ratio * total_steps)


def build_scheduler(optimizer, training_args, total_steps):
    """The schedule named by `--lr_scheduler_type`.

    Through HF's own `get_scheduler`, so every name the flag accepts works. The
    hand-rolled if/elif this replaces knew only `linear` and `cosine` and
    silently gave everything else a constant schedule -- including
    `cosine_with_restarts`, `polynomial` and `inverse_sqrt`, which are valid
    values of the flag.
    """
    from transformers import get_scheduler

    total_steps = max(1, total_steps)
    return get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=_warmup_steps(training_args, total_steps),
        num_training_steps=total_steps,
    )


def run_training(autocast_dtype=None):
    """Parse arguments, build everything and train.

    Args:
        autocast_dtype: dtype for `torch.autocast` around the forward, or None.
            The backbones load in bf16 but the image processors emit fp32
            pixels; an entrypoint that ran under Accelerate's `bf16` mixed
            precision needs autocast to keep behaving the same way, and one
            that never used it must not silently gain it.
    """
    model_args, data_args, training_args = parse_args()
    setup_distributed()

    spec = get_spec(training_args.kd_loss_type)
    print_master(f"Method: {spec.name} -- {spec.summary}")

    # Built before the dataset so a bad --teacher_embedding_cache or an
    # unusable --kd_loss_type fails in seconds rather than after the dataset
    # has been downloaded and scanned.
    distiller = Distiller(model_args, training_args, data_args=data_args)
    prepare_trainable_parameters(distiller.student)

    # Before the optimizer: a criterion may own weights, and they have to be on
    # the Distiller by the time it is moved to the device and wrapped in DDP.
    criterion = attach_criterion(distiller, build_criterion(training_args))

    train_dataset = DistillationDataset(data_args, model_args)
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    dataloader = build_train_dataloader(train_dataset, collator, training_args)

    optimizer = build_optimizer(distiller, model_args, training_args)
    # Length of the sharded loader, not of the dataset: the sampler has already
    # divided by the world size, and dividing again (as one entrypoint did) or
    # not at all (as the other did) puts the LR schedule on the wrong horizon.
    steps_per_epoch = max(
        1, len(dataloader) // max(1, training_args.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * int(training_args.num_train_epochs)
    print_master(
        f"world_size={world_size()}  batches/epoch={len(dataloader)}  "
        f"optimizer steps/epoch={steps_per_epoch}  total={total_steps}"
    )
    lr_scheduler = build_scheduler(optimizer, training_args, total_steps)

    sharpness_aware = build_sharpness_aware(optimizer, distiller, training_args)
    if sharpness_aware is not None:
        print_master(
            f"Sharpness-aware optimization: {training_args.sharpness_aware} "
            f"(rho={training_args.sam_rho}); every optimizer step runs its "
            f"accumulation window twice."
        )

    DistillTrainer(
        distiller=distiller,
        criterion=criterion,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_args=model_args,
        training_args=training_args,
        autocast_dtype=autocast_dtype,
        sharpness_aware=sharpness_aware,
    ).train()

    print_master("Training completed successfully!")
    if dist.is_initialized():
        dist.destroy_process_group()
