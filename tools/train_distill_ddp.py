"""Distillation training under plain DDP.

    torchrun --nproc_per_node=N tools/train_distill_ddp.py --kd_loss_type ...

The loop lives in `src/training/`; this file only picks the precision policy.
No autocast here: the backbones are loaded in bf16 and this entrypoint has
always run them directly, so adding autocast would change the numbers of every
method launched from `scripts/train/{emkd,emo,hierd,rkd}/`.
"""

# Run directly from the repo root: put the repo root on sys.path so `import src.…`
# resolves without installing the project.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from src.training.entrypoint import run_training

if __name__ == "__main__":
    run_training(autocast_dtype=None)
