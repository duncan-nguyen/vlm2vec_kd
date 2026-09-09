"""The distillation training loop, shared by every method.

Method-independent by construction: it hands `(criterion, batch)` to the
`Distiller` and reads back a dict of scalars. A new criterion's own loss terms
appear in the progress bar and the epoch summary without this file knowing their
names -- the keys are discovered from the first batch.
"""

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src import profiling
from src.training.checkpoint import save_checkpoint
from src.utils import print_master, print_rank


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def to_device(obj, device, non_blocking=True):
    """Move every tensor in a nested batch structure to `device`.

    `non_blocking` only does anything for pinned source memory (which the
    DataLoader provides); it is a no-op otherwise, and the copy is ordered
    against the following kernels on the same stream either way.
    """
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device, non_blocking) for v in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


class _LossMeter:
    """Running means of whatever scalars the criterion returned.

    Kept entirely on the GPU. Pulling `.item()` off every component of every
    micro-batch forces a host sync per call and drains the pipeline the
    prefetching was there to fill; this transfers once per logging step, for
    every component at once.
    """

    def __init__(self, device):
        self.device = device
        self.keys = None
        self._sum = None
        self._n = 0

    def update(self, loss_dict):
        if self.keys is None:
            # Fixed on the first batch so the stacking order is stable.
            self.keys = sorted(loss_dict)
            self._sum = torch.zeros(
                len(self.keys), device=self.device, dtype=torch.float32
            )
        self._sum += torch.stack(
            [self._scalar(loss_dict.get(k, 0.0)) for k in self.keys]
        )
        self._n += 1

    def _scalar(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.device, dtype=torch.float32).reshape(())
        return torch.tensor(float(value), device=self.device, dtype=torch.float32)

    def means(self):
        """One device->host transfer for every component."""
        if not self._n or self.keys is None:
            return {}
        return dict(zip(self.keys, (self._sum / self._n).tolist()))

    def reset(self):
        if self._sum is not None:
            self._sum.zero_()
        self._n = 0

    def __bool__(self):
        return self._n > 0


class DistillTrainer:
    """Runs one distillation experiment.

    Args:
        distiller: the `Distiller` (student + teacher/cache + projectors).
        criterion: the loss, from `src.criterions.build_criterion`.
        dataloader: from `src.training.build_train_dataloader`.
        optimizer / lr_scheduler: already built over the trainable parameters.
        model_args / training_args: parsed arguments.
        device: where to run; defaults to this rank's CUDA device.
        autocast_dtype: run the forward under `torch.autocast` with this dtype,
            or None for no autocast. The backbones are loaded in bf16 while the
            image processors hand back fp32 pixels, so which of the two an
            entrypoint uses is a real behavioural choice rather than a detail --
            it is passed in rather than guessed here.
        sharpness_aware: a `src.training.sam.SharpnessAwareOptimizer` wrapping
            `optimizer`, or None for the ordinary single-pass step. When set,
            every optimizer step replays its whole accumulation window at the
            perturbed weights, so it costs two forward/backward passes.
    """

    def __init__(
        self,
        distiller,
        criterion,
        dataloader,
        optimizer,
        lr_scheduler,
        model_args,
        training_args,
        device=None,
        autocast_dtype=None,
        sharpness_aware=None,
    ):
        self.training_args = training_args
        self.model_args = model_args
        self.criterion = criterion
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.autocast_dtype = autocast_dtype
        self.sharpness_aware = sharpness_aware

        if device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = (
                torch.device(f"cuda:{local_rank}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.distiller = distiller.to(self.device)
        self.module = self.distiller
        if dist.is_initialized() and world_size() > 1:
            # broadcast_buffers: the teacher is frozen and the student has no
            # running stats, so the per-forward buffer broadcast is pure
            # overhead. gradient_as_bucket_view: DDP reuses its reduction
            # buckets as .grad storage instead of keeping a second copy.
            ddp_kwargs = dict(
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            # A criterion that leaves the KD projectors unused makes DDP abort
            # with "expected to have finished reduction". `--ddp_find_unused_parameters
            # true` is the escape hatch; it costs a graph traversal per step, so
            # it stays off unless asked for.
            find_unused = getattr(training_args, "ddp_find_unused_parameters", None)
            if find_unused is not None:
                ddp_kwargs["find_unused_parameters"] = find_unused
            bucket_cap = getattr(training_args, "ddp_bucket_cap_mb", None)
            if bucket_cap is not None:
                ddp_kwargs["bucket_cap_mb"] = bucket_cap
            self.distiller = DDP(self.distiller, **ddp_kwargs)
            self.module = self.distiller.module

        self.grad_accum = max(1, int(training_args.gradient_accumulation_steps))
        self.logging_steps = max(1, int(training_args.logging_steps))
        self.max_grad_norm = getattr(training_args, "max_grad_norm", None)

    # ------------------------------------------------------------- helpers

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _no_sync(self, is_sync_step):
        """Skip the DDP all-reduce on every micro-batch but the last of a window.

        Without this, gradients are reduced `grad_accum` times per optimizer
        step and `grad_accum - 1` of those reductions are thrown away.
        """
        if is_sync_step or not isinstance(self.distiller, DDP):
            return nullcontext()
        return self.distiller.no_sync()

    def _trainable_parameters(self):
        return [p for p in self.module.parameters() if p.requires_grad]

    def _micro_step(self, batch, is_sync_step):
        """One forward/backward over a micro-batch, accumulating into `.grad`."""
        with self._no_sync(is_sync_step):
            with profiling.section("forward"), self._autocast():
                loss_dict = self.distiller(self.criterion, batch)
                loss = loss_dict["loss"] / self.grad_accum
            with profiling.section("backward"):
                loss.backward()
        return loss_dict

    def _optimizer_step(self, window_batches=None):
        """Apply the accumulated gradients.

        With `--sharpness_aware`, the gradient currently in `.grad` is the one at
        `w` and is used only to choose the perturbation: the window is replayed
        at `w + eps` and it is *that* gradient the base optimizer sees. The
        replay needs the same micro-batches, which is why the loop hands them
        back here rather than the wrapper holding them.
        """
        if self.sharpness_aware is not None and window_batches:
            with profiling.section("sam_ascent"):
                perturbed = self.sharpness_aware.ascent_step()
            if perturbed:
                # The ascent gradient has done its job; the descent pass must
                # not accumulate on top of it.
                self.optimizer.zero_grad(set_to_none=True)
                last = len(window_batches) - 1
                try:
                    for i, batch in enumerate(window_batches):
                        self._micro_step(batch, is_sync_step=(i == last))
                finally:
                    # An OOM in the replay must not leave the model sitting at
                    # w + eps, which is silent rather than fatal.
                    self.sharpness_aware.restore()

        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self._trainable_parameters(), self.max_grad_norm
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _steps_per_epoch(self):
        return max(1, len(self.dataloader) // self.grad_accum)

    # --------------------------------------------------------------- epoch

    def run_epoch(self, epoch):
        sampler = getattr(self.dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            # Without this every epoch draws the same permutation.
            sampler.set_epoch(epoch)

        self.distiller.train()
        window = _LossMeter(self.device)  # since the last log line
        epoch_meter = _LossMeter(self.device)  # since the start of the epoch
        opt_step = 0
        micro_in_window = 0
        # Only populated for a sharpness-aware run, which has to replay the
        # window at the perturbed weights. At --gradient_accumulation_steps 1
        # this holds the one batch that is alive anyway.
        window_batches = []

        progress_bar = tqdm(
            total=self._steps_per_epoch(),
            desc=f"Epoch {epoch + 1}",
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(
            profiling.timed_iter(self.dataloader, "data_wait")
        ):
            with profiling.section("to_device"):
                batch = to_device(batch, self.device)

            is_sync_step = (batch_idx + 1) % self.grad_accum == 0
            if self.sharpness_aware is not None:
                window_batches.append(batch)

            loss_dict = self._micro_step(batch, is_sync_step)

            # The logged numbers are the clean ones, at `w`: a sharpness-aware
            # run's second pass is evaluated at the perturbed weights and its
            # loss is not the loss the run is minimising.
            with profiling.section("bookkeeping"):
                window.update(loss_dict)
                epoch_meter.update(loss_dict)
                micro_in_window += 1

            if is_sync_step:
                with profiling.section("optimizer"):
                    self._optimizer_step(window_batches)
                window_batches = []
                micro_in_window = 0
                opt_step += 1
                profiling.profiler.step()

                if is_main_process():
                    progress_bar.update(1)
                    if opt_step % self.logging_steps == 0:
                        progress_bar.set_postfix(self._postfix(window))
                        window.reset()

                if profiling.profiler.should_report and is_main_process():
                    print(profiling.profiler.report(f"epoch {epoch + 1}"), flush=True)
                if profiling.profiler.should_stop:
                    print_rank(
                        f"VLM2VEC_PROFILE_STEPS reached "
                        f"({profiling.profiler.n_steps} steps), stopping epoch early."
                    )
                    break

        # Gradients from a trailing partial accumulation window would otherwise
        # sit in .grad and be folded into the first step of the next epoch.
        if micro_in_window:
            self._optimizer_step(window_batches)

        progress_bar.close()
        if profiling.profiler.enabled and is_main_process():
            print(profiling.profiler.report(f"epoch {epoch + 1} final"), flush=True)
        return epoch_meter.means()

    def _postfix(self, meter):
        stats = meter.means()
        postfix = {k: f"{v:.4f}" for k, v in stats.items()}
        postfix["lr"] = f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
        return postfix

    # ---------------------------------------------------------------- train

    def train(self):
        num_epochs = int(self.training_args.num_train_epochs)
        for epoch in range(num_epochs):
            print_master(f"Start epoch {epoch + 1}/{num_epochs}")
            stats = self.run_epoch(epoch)
            if is_main_process() and stats:
                summary = " | ".join(f"{k}: {v:.4f}" for k, v in sorted(stats.items()))
                print_master(f"Epoch {epoch + 1} done. {summary}")

            if self.training_args.save_strategy == "epoch":
                save_checkpoint(
                    self.module,
                    self.model_args,
                    os.path.join(
                        self.training_args.output_dir, f"checkpoint-epoch{epoch + 1}"
                    ),
                )
            if dist.is_initialized():
                dist.barrier()

        save_checkpoint(
            self.module,
            self.model_args,
            os.path.join(self.training_args.output_dir, "checkpoint-final"),
        )
        if dist.is_initialized():
            dist.barrier()
