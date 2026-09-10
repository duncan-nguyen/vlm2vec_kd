#!/usr/bin/env python3
"""Upload a checkpoint directory to the Hugging Face Hub.

    python tools/misc/push_to_hub.py <folder> <repo-id> [--private] [--path-in-repo DIR]

Prefer `--push_to_hub --hub_model_id <repo>` on the training command: the run
uploads `checkpoint-final` itself when it finishes (see src/training/hub.py).
This script is for pushing a checkpoint that was trained before that flag, or
re-pushing one after a failed upload.

The token comes from `--token`, `$HF_TOKEN`, or the login cache -- never from
this file. The version of this script in git history had a real token hard-coded
in it; it is in the published history and must be treated as compromised.
"""

# Run directly from the repo root: put the repo root on sys.path so `import src.…`
# resolves without installing the project.
import os as _os
import sys as _sys

_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import argparse
import os
import sys

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folder", help="local directory to upload, e.g. .../checkpoint-final")
    ap.add_argument("repo_id", help="target repo, e.g. DVLe/vlm_propose_hateful")
    ap.add_argument("--path-in-repo", default="", help="destination inside the repo")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--commit-message", default=None)
    ap.add_argument(
        "--token",
        default=None,
        help="Hub token; defaults to $HF_TOKEN, then the huggingface-cli login cache",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"error: {args.folder} is not a directory")

    token = args.token or os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message
        or f"Upload {os.path.basename(args.folder.rstrip('/'))}",
        # The eval embedding dumps are gigabytes and reproducible from the weights.
        ignore_patterns=["**/*_qry", "**/*_tgt", "**/*_pred.txt", "**/__pycache__/**"],
    )
    print(f"Pushed {args.folder} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
