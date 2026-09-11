"""Uploading a finished checkpoint to the Hugging Face Hub.

Nothing in this repo pushed a trained checkpoint anywhere: the only working
upload was `tools/misc/push_to_hub.py`, a script with one hard-coded folder and
one hard-coded repo that had to be edited before every use, and the
`push_to_hub` helper inside `tools/train_distillation.py`, which is defined and
never called.

`TrainingArguments` already carries `--push_to_hub`, `--hub_model_id`,
`--hub_token` and `--hub_private_repo` -- they come from Hugging Face's own
dataclass. This loop is not `Trainer`, so they did nothing. Honour them here
instead of inventing a parallel set of flags.

Layout
------
Two repos, not one per run: the proposed methods in one, the baselines they are
compared against in the other, each run its own directory inside::

    nqdhocai/vlm2vec-kd-ours/       cmtop/FastVLM-0.5B/vqa/cmmerge_seed42/
    nqdhocai/vlm2vec-kd-baselines/  talas/FastVLM-0.5B/cls/talas_seed42/
                                    span_propose_attn/FastVLM-0.5B/cls/hierd_fastvlm_cls/

The directory is built from the run's own arguments rather than from
`--output_dir`, because the launchers disagree about what that looks like
(`training/RKD`, `training/meta_emkd_grounding`, `training/hierd_fastvlm_cls`) and
those names would be unreadable side by side in a shared repo.
"""

import json
import os

from src.evaluation.benchmarks import infer_task
from src.hf_auth import resolve_token as _resolve_hf_token
from src.utils import print_master

DEFAULT_OWNER = "nqdhocai"

# Repository per track, under `--hub_owner`.
TRACK_REPOS = {
    "ours": "vlm2vec-kd-ours",
    "baseline": "vlm2vec-kd-baselines",
}

# `--kd_loss_type` values that are this work's own contribution. Everything else
# is a published method being reproduced for comparison -- TALAS and HieRD have
# their papers in `docs/baseline methods/`. A new proposal goes here; a new
# competitor needs no change.
OURS_METHODS = {"cmtop"}

# The embedding dumps `tools/eval_mmeb.py` leaves next to its scores: one pickle
# per subset per side, gigabytes in total, and reproducible from the weights.
IGNORE_PATTERNS = ("**/*_qry", "**/*_tgt", "**/*_pred.txt", "**/__pycache__/**")

_METADATA_FILE = "vlm2vec_kd_run.json"


def resolve_token(training_args):
    """`--hub_token`, else `$HF_TOKEN`, else the repo's `.hf_token`, else None.

    None is fine: `huggingface_hub` falls back to the token cached by
    `huggingface-cli login`. The order lives in src/hf_auth.py, which is also
    what puts the token in the environment for the dataset and weight downloads.
    """
    return _resolve_hf_token(getattr(training_args, "hub_token", None))


def resolve_track(training_args):
    """Which of the two repos this run belongs in."""
    track = (getattr(training_args, "hub_track", None) or "auto").lower()
    if track != "auto":
        return track
    return "ours" if training_args.kd_loss_type in OURS_METHODS else "baseline"


def resolve_repo_id(training_args):
    """`--hub_model_id` if given, else `<owner>/<repo for this track>`."""
    if getattr(training_args, "hub_model_id", None):
        return training_args.hub_model_id
    track = resolve_track(training_args)
    if track not in TRACK_REPOS:
        raise ValueError(
            f"--hub_track must be one of auto, {', '.join(TRACK_REPOS)}; got '{track}'"
        )
    owner = getattr(training_args, "hub_owner", None) or DEFAULT_OWNER
    return f"{owner}/{TRACK_REPOS[track]}"


def resolve_path_in_repo(model_args, data_args, training_args):
    """`<method>/<student>/<task>/<run>`, the directory this run occupies.

    Every component comes from an argument the run cannot be launched without,
    so the same cell of the experiment grid always lands in the same place and
    two different cells never collide. `<run>` is the basename of `--output_dir`,
    which is what the launchers already vary per variant and seed.
    """
    if getattr(training_args, "hub_path_in_repo", None):
        return training_args.hub_path_in_repo.strip("/")

    parts = [training_args.kd_loss_type]
    student = os.path.basename((model_args.model_name or "student").rstrip("/"))
    parts.append(student)
    task = infer_task(data_args.subset_name)
    if task:
        parts.append(task)
    run = os.path.basename((training_args.output_dir or "run").rstrip("/"))
    parts.append(run)
    return "/".join(p for p in parts if p)


def run_metadata(model_args, data_args, training_args, path_in_repo):
    """What a directory in a shared repo needs in order to be identifiable.

    The checkpoint's own `config.json` describes the backbone, not the
    distillation: without this, two directories differing only in `--kd_weight`
    are indistinguishable once they are sitting next to thirty others.
    """
    return {
        "path_in_repo": path_in_repo,
        "track": resolve_track(training_args),
        "kd_loss_type": training_args.kd_loss_type,
        "student": model_args.model_name,
        "student_backbone": model_args.model_backbone,
        "teacher": model_args.teacher_model_name,
        "teacher_backbone": model_args.teacher_backbone,
        "task": infer_task(data_args.subset_name),
        "train_subsets": list(data_args.subset_name or []),
        "image_resolution": data_args.image_resolution,
        "pooling": model_args.pooling,
        "normalize": bool(model_args.normalize),
        "lora": bool(model_args.lora),
        "lora_r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "learning_rate": training_args.learning_rate,
        "projector_lr": model_args.projector_lr,
        "kd_weight": training_args.kd_weight,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "num_train_epochs": training_args.num_train_epochs,
        "seed": training_args.seed,
        "sharpness_aware": training_args.sharpness_aware,
        "output_dir": training_args.output_dir,
    }


def write_run_metadata(checkpoint_dir, metadata):
    """Drop the run description into the checkpoint so it is uploaded with it."""
    path = os.path.join(checkpoint_dir, _METADATA_FILE)
    try:
        with open(path, "w") as handle:
            json.dump(metadata, handle, indent=2, default=str)
    except OSError as exc:
        print_master(f"Warning: could not write {path}: {exc}")
    return path


def push_checkpoint(
    training_args, checkpoint_dir, model_args=None, data_args=None, commit_message=None
):
    """Upload `checkpoint_dir` into this run's directory. Returns True on success.

    Best-effort by design: a network failure at the very end of a multi-hour run
    must not take the exit code of the run with it -- the weights are on local
    disk and can be pushed later with `tools/misc/push_to_hub.py`.
    """
    if not getattr(training_args, "push_to_hub", False):
        return False

    upload_dir = getattr(training_args, "hub_upload_dir", None) or checkpoint_dir
    if not os.path.isdir(upload_dir):
        print_master(f"--push_to_hub: {upload_dir} does not exist; nothing uploaded.")
        return False

    try:
        repo_id = resolve_repo_id(training_args)
        path_in_repo = (
            resolve_path_in_repo(model_args, data_args, training_args)
            if model_args is not None and data_args is not None
            else (getattr(training_args, "hub_path_in_repo", "") or "").strip("/")
        )
    except ValueError as exc:
        print_master(f"--push_to_hub: {exc}")
        return False

    if model_args is not None and data_args is not None:
        write_run_metadata(
            upload_dir, run_metadata(model_args, data_args, training_args, path_in_repo)
        )

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=resolve_token(training_args))
        api.create_repo(
            repo_id,
            private=bool(getattr(training_args, "hub_private_repo", False)),
            exist_ok=True,
        )
        destination = f"{repo_id}/{path_in_repo}" if path_in_repo else repo_id
        print_master(
            f"Uploading {upload_dir} to https://huggingface.co/{destination} ..."
        )
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            commit_message=commit_message or f"Upload {path_in_repo or upload_dir}",
            ignore_patterns=list(IGNORE_PATTERNS),
        )
    except Exception as exc:  # noqa: BLE001 - the weights are safe on disk
        print_master(f"--push_to_hub failed: {exc}")
        return False

    print_master(f"Pushed to https://huggingface.co/{destination}")
    return True
