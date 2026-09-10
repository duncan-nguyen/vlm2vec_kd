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
"""

import os

from src.utils import print_master

# The embedding dumps `tools/eval_mmeb.py` leaves next to its scores: one pickle
# per subset per side, gigabytes in total, and reproducible from the weights.
_IGNORE = ("**/*_qry", "**/*_tgt", "**/*_pred.txt", "**/__pycache__/**")


def resolve_token(training_args):
    """`--hub_token`, else the usual environment variables, else None.

    None is fine: `huggingface_hub` falls back to the token cached by
    `huggingface-cli login`.
    """
    return (
        getattr(training_args, "hub_token", None)
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


def push_checkpoint(training_args, checkpoint_dir, commit_message=None):
    """Upload `checkpoint_dir` to `--hub_model_id`. Returns True on success.

    Best-effort by design: a network failure at the very end of a multi-hour run
    must not take the exit code of the run with it -- the weights are on local
    disk and can be pushed later with the same flags.
    """
    if not getattr(training_args, "push_to_hub", False):
        return False

    repo_id = getattr(training_args, "hub_model_id", None)
    if not repo_id:
        print_master(
            "--push_to_hub was set but --hub_model_id is empty; nothing uploaded."
        )
        return False

    upload_dir = getattr(training_args, "hub_upload_dir", None) or checkpoint_dir
    if not os.path.isdir(upload_dir):
        print_master(f"--push_to_hub: {upload_dir} does not exist; nothing uploaded.")
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=resolve_token(training_args))
        api.create_repo(
            repo_id,
            private=bool(getattr(training_args, "hub_private_repo", False)),
            exist_ok=True,
        )
        print_master(f"Uploading {upload_dir} to https://huggingface.co/{repo_id} ...")
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            commit_message=commit_message or f"Upload {os.path.basename(upload_dir)}",
            ignore_patterns=list(_IGNORE),
        )
    except Exception as exc:  # noqa: BLE001 - the weights are safe on disk
        print_master(f"--push_to_hub failed: {exc}")
        return False

    print_master(f"Pushed to https://huggingface.co/{repo_id}")
    return True
