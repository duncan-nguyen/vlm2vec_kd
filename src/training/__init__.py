"""Shared training machinery for every distillation method.

`tools/train_distill_ddp.py` and `tools/train_distill_no_deepspeed.py` used to
carry two independent copies of the loop, which had drifted apart on grad
accumulation, dataloader settings, per-step synchronisation and checkpointing.
Both now call into here, so a fix to the loop reaches every method at once.
"""

from src.training.dataloader import build_train_dataloader
from src.training.loop import DistillTrainer

__all__ = ["DistillTrainer", "build_train_dataloader"]
