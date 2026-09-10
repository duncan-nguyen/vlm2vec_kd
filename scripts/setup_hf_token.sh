#!/bin/bash
# Store the Hugging Face token the trainer picks up by default.
#
#   bash scripts/setup_hf_token.sh                 # prompts, does not echo
#   bash scripts/setup_hf_token.sh hf_xxxxxxxx     # non-interactive
#
# Writes <repo root>/.hf_token, owner-readable only. `.gitignore` excludes it,
# which is the whole point: a token pasted into src/arguments.py or a launcher
# ends up in a commit, and this repo has already had to treat one token as
# compromised for exactly that reason.
#
# From then on every training run, evaluation and `--push_to_hub` finds it with
# no flag and no environment variable (see src/hf_auth.py). $HF_TOKEN, if set,
# still wins -- so a one-off run can use a different token without touching this.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/.hf_token"

TOKEN="${1:-}"
if [ -z "$TOKEN" ]; then
    read -r -s -p "Hugging Face token (input hidden): " TOKEN
    echo
fi
if [ -z "$TOKEN" ]; then
    echo "error: no token given" >&2
    exit 1
fi
case "$TOKEN" in
    hf_*) ;;
    *) echo "warning: a Hugging Face token normally starts with 'hf_'" >&2 ;;
esac

# Create with the right mode before anything is written to it, so the token is
# never briefly world-readable.
touch "$DEST"
chmod 600 "$DEST"
printf '%s\n' "$TOKEN" > "$DEST"

echo "Wrote $DEST (mode $(stat -c '%a' "$DEST"))"
git -C "$ROOT" check-ignore -q "$DEST" \
    && echo "git ignores it, as it must." \
    || echo "WARNING: git does NOT ignore $DEST -- check .gitignore before committing." >&2
