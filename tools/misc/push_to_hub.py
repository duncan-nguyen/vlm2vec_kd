#!/usr/bin/env python3
"""Upload a checkpoint directory to the Hugging Face Hub.

    python tools/misc/push_to_hub.py <folder> --track baseline --path-in-repo talas/FastVLM-0.5B/cls/talas_seed42
    python tools/misc/push_to_hub.py <folder> --repo nqdhocai/vlm2vec-kd-ours --path-in-repo cmtop/...

Prefer `--push_to_hub` on the training command: the run uploads
`checkpoint-final` into the right repo and directory itself when it finishes,
deriving both from its own arguments (see src/training/hub.py). This script is
for pushing a checkpoint trained before that flag existed, or re-pushing one
after a failed upload.

`--track ours|baseline` selects the same two collection repos the trainer uses,
so a manual push lands next to the automatic ones rather than in a repo of its
own. Give `--path-in-repo` the directory the run should occupy; without it the
upload goes to the repo root, which in a shared repo is almost never right.

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

from src.hf_auth import resolve_token
from src.training.hub import DEFAULT_OWNER, IGNORE_PATTERNS, TRACK_REPOS


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "folder", help="local directory to upload, e.g. .../checkpoint-final"
    )
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--track",
        choices=sorted(TRACK_REPOS),
        help=f"push to <--owner>/{{{', '.join(f'{k}: {v}' for k, v in sorted(TRACK_REPOS.items()))}}}",
    )
    target.add_argument("--repo", dest="repo_id", help="an explicit repo id instead")
    ap.add_argument("--owner", default=DEFAULT_OWNER, help="owner of the --track repos")
    ap.add_argument(
        "--path-in-repo",
        default="",
        help="directory inside the repo, e.g. cmtop/FastVLM-0.5B/vqa/cmtop_h0_seed42. "
        "The trainer's own layout is <kd_loss_type>/<student>/<task>/<run>",
    )
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--commit-message", default=None)
    ap.add_argument(
        "--token",
        default=None,
        help="Hub token; defaults to $HF_TOKEN, then the repo's .hf_token, "
        "then the huggingface-cli login cache",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"error: {args.folder} is not a directory")

    repo_id = args.repo_id or f"{args.owner}/{TRACK_REPOS[args.track]}"
    path_in_repo = args.path_in_repo.strip("/")
    if not path_in_repo and args.track:
        # The root of a collection repo holds every other run; an upload there
        # would scatter this checkpoint's files among them.
        sys.exit(
            f"error: --path-in-repo is required for --track {args.track}, "
            f"which pushes into the shared repo {repo_id}"
        )

    api = HfApi(token=resolve_token(args.token))
    api.create_repo(repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=args.folder,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        commit_message=args.commit_message
        or f"Upload {path_in_repo or os.path.basename(args.folder.rstrip('/'))}",
        # The eval embedding dumps are gigabytes and reproducible from the weights.
        ignore_patterns=list(IGNORE_PATTERNS),
    )
    destination = f"{repo_id}/{path_in_repo}" if path_in_repo else repo_id
    print(f"Pushed {args.folder} -> https://huggingface.co/{destination}")


if __name__ == "__main__":
    main()
