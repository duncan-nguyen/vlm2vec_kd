"""Saving a distillation checkpoint.

The two training entrypoints had two copies of this, and both hard-coded the
path to the multimodal projector by backbone name::

    if backbone in ["llava_onevision", "llava_two_vision"]:
        student.encoder.model.multi_modal_projector
    else:
        student.encoder.model.model.mm_projector

which raises `AttributeError` at the end of a finished epoch for any backbone
that has neither -- `qwen2_vl`, for one. Here the projector is looked up by
probing the known attribute paths and simply skipped when there is none.
"""

import os

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from src.utils import print_master

# Where a multimodal projector lives, by backbone family. Probed in order; the
# first that resolves is saved. A backbone with none is fine -- its projector
# weights are inside the encoder that `save_pretrained` already wrote.
_PROJECTOR_PATHS = (
    "encoder.model.multi_modal_projector",  # llava_onevision, llava_next
    "encoder.model.model.mm_projector",  # llava_qwen2 / FastVLM
)


def _resolve(root, dotted):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_main():
    import torch.distributed as dist

    return (not dist.is_initialized()) or dist.get_rank() == 0


def save_checkpoint(distiller, model_args, ckpt_dir):
    """Write the student (and anything trained alongside it) to `ckpt_dir`.

    Saves, in addition to what the old code saved:

    * the KD projectors (`distiller.projectors`). They are trained -- they get
      their own parameter group and their own `--projector_lr` -- and were
      dropped on every checkpoint, so a run could not be resumed or its KD loss
      reproduced from what was written to disk.
    """
    if not _is_main():
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    student = distiller.student

    student.encoder.save_pretrained(ckpt_dir)

    for path in _PROJECTOR_PATHS:
        projector = _resolve(student, path)
        if projector is not None and hasattr(projector, "state_dict"):
            torch.save(projector.state_dict(), os.path.join(ckpt_dir, "mm_projector.pth"))
            break

    projectors = getattr(distiller, "projectors", None)
    if projectors is not None and len(projectors) > 0:
        torch.save(
            projectors.state_dict(), os.path.join(ckpt_dir, "kd_projectors.pth")
        )

    # A criterion that owns weights (TALAS's per-layer W_l) is attached to the
    # distiller by `attach_criterion`, and they are trained, so they belong in
    # the checkpoint for the same reason the KD projectors do.
    criterion = getattr(distiller, "criterion", None)
    if criterion is not None and any(True for _ in criterion.parameters()):
        torch.save(
            criterion.state_dict(), os.path.join(ckpt_dir, "kd_criterion.pth")
        )

    _save_hub_artifacts(model_args.model_name, ckpt_dir)
    print_master(f"Saved checkpoint to {ckpt_dir}")


def _save_hub_artifacts(model_name, ckpt_dir):
    """Config, tokenizer and processor, each best-effort.

    A backbone without a registered `AutoProcessor` is normal, and a missing
    processor must not lose the weights that were just written.
    """
    if not model_name:
        return
    for label, loader in (
        ("config", AutoConfig),
        ("tokenizer", AutoTokenizer),
        ("processor", AutoProcessor),
    ):
        try:
            loader.from_pretrained(model_name).save_pretrained(ckpt_dir)
        except Exception as exc:  # noqa: BLE001 - best effort by design
            print_master(f"Warning: could not save {label} to {ckpt_dir}: {exc}")
