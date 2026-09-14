#!/bin/bash
# Several benchmark groups in a row, then the combined table.
#
#   bash scripts/eval/all.sh <checkpoint-dir>                       # CLS + VQA, IND and OOD
#   EVAL_GROUPS="ret ret_ood" bash scripts/eval/all.sh <dir>        # a retrieval checkpoint
#   EVAL_GROUPS="gd gd_ood" bash scripts/eval/all.sh <dir>          # a grounding checkpoint
#   BACKBONE=llava_onevision IMAGE_RESOLUTION=336 bash scripts/eval/all.sh <dir>
#
# EVAL_GROUPS names the wrappers in this directory (cls, vqa, ret, gd and their
# _ood twins); each writes to <checkpoint>/mmeb_<wrapper>. The default keeps the
# CLS/VQA table this script has always produced. (Not GROUPS: bash reserves it.)
#
# Sequential, one GPU. Training with `--eval_after_train` runs the groups of the
# task it trained on sharded across every rank instead, which is what to prefer
# when the checkpoint has just been produced -- it also inherits the image
# resolution and backbone from the run rather than taking them from the
# environment here.
set -euo pipefail

CKPT="${1:-${CKPT:-}}"
if [ -z "$CKPT" ]; then
    echo "usage: bash scripts/eval/all.sh <checkpoint-dir>" >&2
    exit 1
fi
HERE="$(dirname "$0")"
EVAL_GROUPS="${EVAL_GROUPS:-cls vqa cls_ood vqa_ood}"
read -r -a groups <<<"$EVAL_GROUPS"

# Check every name before the first multi-minute evaluation starts.
for group in "${groups[@]}"; do
    if [ ! -f "$HERE/$group.sh" ]; then
        echo "unknown group '$group': $HERE/$group.sh does not exist" >&2
        exit 1
    fi
done

# Through the per-group wrappers, so each group lands in the output directory
# tools/summarize_mmeb.py is pointed at below.
dirs=()
for group in "${groups[@]}"; do
    bash "$HERE/$group.sh" "$CKPT" "${@:2}"
    dirs+=("$CKPT/mmeb_$group")
done

python tools/summarize_mmeb.py "${dirs[@]}" --json "$CKPT/mmeb_summary.json"
