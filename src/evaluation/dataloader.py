"""How evaluation batches are fed to the GPU.

The training loop learned this once already (see `src/training/dataloader.py`):
with `num_workers=0` the JPEG decode, the resize and the processor pass -- which
tokenises and turns PIL images into pixel tensors -- all run on the process that
also has to drive the GPU, so the device sits idle for the whole of it.

`tools/eval_mmeb.py` had exactly that: `num_workers=0`, no pinned memory, no
prefetch, on both the query and the candidate loader. Evaluation is pure
inference, so the forward is *shorter* than in training while the input work is
the same -- the ratio of stall to compute is worse here than it is there.
"""

from torch.utils.data import DataLoader

from src.utils import print_rank

#: Used when `--dataloader_num_workers` is left at HF's default of 0, which is
#: indistinguishable from "not set". Pass a negative value to genuinely run the
#: input pipeline in-process.
DEFAULT_NUM_WORKERS = 8


def resolve_num_workers(training_args):
    requested = getattr(training_args, "dataloader_num_workers", 0) or 0
    return 0 if requested < 0 else (requested or DEFAULT_NUM_WORKERS)


def build_eval_dataloader(dataset, collator, training_args, description=""):
    """A DataLoader that decodes and collates ahead of the encoder.

    `shuffle=False` and `drop_last=False`: the embeddings are written out
    positionally and zipped back against `dataset.paired_data`, so reordering or
    dropping a batch would silently misalign every score that follows.
    """
    num_workers = resolve_num_workers(training_args)
    kwargs = dict(
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=getattr(training_args, "dataloader_pin_memory", True),
    )
    if num_workers > 0:
        # No persistent_workers: each loader is consumed exactly once, so
        # keeping the workers alive afterwards only holds memory.
        kwargs["prefetch_factor"] = (
            getattr(training_args, "dataloader_prefetch_factor", None) or 4
        )
    print_rank(
        f"Eval DataLoader{f' ({description})' if description else ''}: "
        f"num_workers={num_workers}, pin_memory={kwargs['pin_memory']}, "
        f"prefetch_factor={kwargs.get('prefetch_factor')}, "
        f"batch_size={kwargs['batch_size']}"
    )
    return DataLoader(dataset, **kwargs)
