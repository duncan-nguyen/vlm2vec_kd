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

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler

from src.utils import print_master

#: Used when `--dataloader_num_workers` is left at HF's default of 0. This loop
#: is input-bound often enough that running the whole input pipeline on the
#: process driving the GPU is the wrong default.
#:
#: HF's default is itself 0, so "not set" and "set to 0" are indistinguishable.
#: To genuinely run in-process -- under a debugger, or on a machine with no
#: spare cores -- pass a negative value: `--dataloader_num_workers -1`.
DEFAULT_NUM_WORKERS = 4


class TaskHomogeneousSampler(Sampler):
    """Shuffle globally while keeping each DDP batch inside one task.

    CM-Merge builds one retrieval graph per optimizer micro-batch.  Mixing, for
    example, VQA answers and classification labels introduces meaningless
    cross-task edges.  This sampler first makes full *global* batches within
    every contiguous task range, then shuffles those batches and gives each rank
    its disjoint local slice.  Consequently every rank sees the same task at a
    step and cross-rank gathering reconstructs exactly one task graph.
    """

    def __init__(
        self,
        dataset,
        local_batch_size,
        seed=0,
        rank=None,
        world_size=None,
    ):
        ranges = getattr(dataset, "task_index_ranges", None)
        if not ranges:
            raise ValueError(
                "task-homogeneous sampling requires dataset.task_index_ranges"
            )
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")

        distributed = dist.is_available() and dist.is_initialized()
        self.rank = (
            dist.get_rank() if rank is None and distributed else int(rank or 0)
        )
        self.world_size = (
            dist.get_world_size()
            if world_size is None and distributed
            else int(1 if world_size is None else world_size)
        )
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}"
            )

        self.task_ranges = [(int(start), int(end)) for start, end in ranges]
        self.local_batch_size = int(local_batch_size)
        self.global_batch_size = self.local_batch_size * self.world_size
        self.seed = int(seed)
        self.epoch = 0
        self._global_batch_count = sum(
            max(0, end - start) // self.global_batch_size
            for start, end in self.task_ranges
        )
        if self._global_batch_count == 0:
            raise ValueError(
                "no task contains one full global batch of "
                f"{self.global_batch_size} examples"
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _global_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        batches = []
        for start, end in self.task_ranges:
            size = end - start
            usable = (size // self.global_batch_size) * self.global_batch_size
            if usable == 0:
                continue
            indices = torch.randperm(size, generator=generator)[:usable] + start
            batches.extend(indices.reshape(-1, self.global_batch_size))

        order = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[i] for i in order]

    def __iter__(self):
        offset = self.rank * self.local_batch_size
        stop = offset + self.local_batch_size
        for global_batch in self._global_batches():
            yield from global_batch[offset:stop].tolist()

    def __len__(self):
        return self._global_batch_count * self.local_batch_size


def build_train_sampler(
    dataset,
    seed=0,
    batch_size=None,
    task_homogeneous=False,
):
    """Shuffling sampler, sharded across ranks when running distributed."""
    if task_homogeneous:
        if batch_size is None:
            raise ValueError("batch_size is required for task-homogeneous sampling")
        return TaskHomogeneousSampler(dataset, batch_size, seed=seed)
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
        task_homogeneous = (
            getattr(training_args, "kd_loss_type", None) == "cmtop"
            and getattr(training_args, "cmtop_task_homogeneous", True)
        )
        sampler = build_train_sampler(
            dataset,
            seed=getattr(training_args, "seed", 0),
            batch_size=training_args.per_device_train_batch_size,
            task_homogeneous=task_homogeneous,
        )

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
        f"batch_size={kwargs['batch_size']}, sampler={type(sampler).__name__}"
    )
    return DataLoader(dataset, **kwargs)
