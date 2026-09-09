"""Distillation training without DeepSpeed, under bf16 autocast.

    torchrun --nproc_per_node=N tools/train_distill_no_deepspeed.py --kd_loss_type cmtop ...

Used by `scripts/train/cmtop/`. Same loop as `train_distill_ddp.py`; the
difference is the precision policy. This entrypoint used to run under
Accelerate's `mixed_precision="bf16"`, so it keeps bf16 autocast around the
forward -- the image processors hand back fp32 pixel values and the backbones
are bf16, and autocast is what reconciled the two.

Accelerate itself is gone. What it was doing here was wrong in three ways that
only showed up above one GPU: `accelerator.backward` divides by
`gradient_accumulation_steps` and the loop divided again (gradients scaled by
1/accum^2), the DataLoader was handed a `DistributedSampler` *and* then prepared
by Accelerate (each rank saw 1/world_size^2 of the data), and `distiller.student`
was read after `prepare()` had wrapped the distiller in DDP.
"""

# Run directly from the repo root: put the repo root on sys.path so `import src.…`
# resolves without installing the project.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch

from src.training.entrypoint import run_training

if __name__ == "__main__":
    run_training(autocast_dtype=torch.bfloat16)
