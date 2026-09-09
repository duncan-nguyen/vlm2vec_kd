import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
)
logger = logging.getLogger(__name__)
import os

import torch


def print_rank(message):
    """If distributed is initialized, print the rank."""
    if torch.distributed.is_initialized():
        logger.info(f"rank{torch.distributed.get_rank()}: " + message)
    else:
        logger.info(message)


def print_master(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)


def find_latest_checkpoint(output_dir):
    """Scan the output directory and return the latest checkpoint path.

    Directories are ranked by the trailing number in their name, e.g.
    `checkpoint-epoch3`. `checkpoint-final` carries no number and is treated as
    the newest of all, since it is only written once training has finished.
    Anything else without a number is ignored rather than raising -- the old
    `int(name.split("-")[-1])` here crashed on `checkpoint-final`, which is the
    directory every run produces.
    """
    import re

    if not os.path.exists(output_dir):
        return None

    def rank(name):
        if name == "checkpoint-final":
            return float("inf")
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else None

    ranked = [
        (rank(d), os.path.join(output_dir, d))
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    ranked = [(r, path) for r, path in ranked if r is not None]
    if not ranked:
        return None
    return max(ranked)[1]


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch
