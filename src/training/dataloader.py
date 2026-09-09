"""One place that decides how training data is fed to the GPU.

The collator runs up to four processor passes per batch (student/teacher x
query/positive), each looping over samples to tokenise and to turn PIL images
into pixel tensors, on top of the JPEG decode in the dataset. On the default
`num_workers=0` all of that happens on the process that also has to drive the
GPU, so the device idles for the whole of it -- `VLM2VEC_PROFILE=1` reports it
as `data_wait`.

Everything below exists to keep that work off the critical path:

``num_workers``
    Subprocesses decode and collate ahead of the loop. This is the single
    largest win; anything above zero helps, and CPU-count-ish is usual.
``persistent_workers``
    Without it every epoch tears the workers down and forks new ones, which for
    a dataset holding a memory-mapped Arrow table and two HF processors is
    seconds of startup per epoch.
``prefetch_factor``
    How many batches each worker runs ahead. The default of 2 is thin when
    per-batch collation is this heavy.
``pin_memory``
    Staging batches in pinned memory is what makes the host->device copy
    asynchronous; without it `non_blocking=True` on the copy is a no-op.
"""

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from src.utils import print_master

#: Used when `--dataloader_num_workers` is left at HF's default of 0. This loop
#: is input-bound often enough that running the whole input pipeline on the
#: process driving the GPU is the wrong default.
#:
#: HF's default is itself 0, so "not set" and "set to 0" are indistinguishable.
#: To genuinely run in-process -- under a debugger, or on a machine with no
#: spare cores -- pass a negative value: `--dataloader_num_workers -1`.
DEFAULT_NUM_WORKERS = 4


def build_train_sampler(dataset, seed=0):
    """Shuffling sampler, sharded across ranks when running distributed."""
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True)
    return RandomSampler(dataset)


def build_train_dataloader(dataset, collator, training_args, sampler=None):
    """The training DataLoader, configured to overlap input work with compute.

    Args:
        dataset: the `DistillationDataset`.
        collator: the `DistillationCollator`.
        training_args: read for batch size, worker count, pin_memory,
            prefetch factor and seed.
        sampler: override the sampler; by default one is built with
            :func:`build_train_sampler`, which shards across ranks.

    Returns:
        A `DataLoader` with `drop_last=True` -- a short final batch would make
        the in-batch contrastive loss see a different number of negatives than
        every other step, and under DDP a rank running out of batches before
        the others hangs the all-reduce.
    """
    if sampler is None:
        sampler = build_train_sampler(dataset, seed=getattr(training_args, "seed", 0))

    requested = training_args.dataloader_num_workers
    num_workers = 0 if requested < 0 else (requested or DEFAULT_NUM_WORKERS)
    kwargs = dict(
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=getattr(training_args, "dataloader_pin_memory", True),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = (
            getattr(training_args, "dataloader_prefetch_factor", None) or 4
        )

    print_master(
        f"DataLoader: num_workers={num_workers}, "
        f"pin_memory={kwargs['pin_memory']}, "
        f"prefetch_factor={kwargs.get('prefetch_factor')}, "
        f"batch_size={kwargs['batch_size']}"
    )
    return DataLoader(dataset, **kwargs)
