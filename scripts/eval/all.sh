#!/bin/bash
# The whole Table 1: CLS/VQA on the in-distribution and out-of-distribution
# benchmarks, then the combined table.
#
#   bash scripts/eval/all.sh <checkpoint-dir>
#   BACKBONE=llava_onevision IMAGE_RESOLUTION=336 bash scripts/eval/all.sh <dir>
#
# Sequential, one GPU. Training with `--eval_after_train` runs the same 20
# benchmarks sharded across every rank instead, which is what to prefer when the
# checkpoint has just been produced -- it also inherits the image resolution and
# backbone from the run rather than taking them from the environment here.
set -euo pipefail

CKPT="${1:-${CKPT:-}}"
if [ -z "$CKPT" ]; then
    echo "usage: bash scripts/eval/all.sh <checkpoint-dir>" >&2
    exit 1
fi
HERE="$(dirname "$0")"

# Through the per-group wrappers, so each group lands in the output directory
# tools/summarize_mmeb.py is pointed at below.
for group in cls vqa cls_ood vqa_ood; do
    bash "$HERE/$group.sh" "$CKPT" "${@:2}"
done

python tools/summarize_mmeb.py \
    "$CKPT/mmeb_cls" "$CKPT/mmeb_vqa" "$CKPT/mmeb_cls_ood" "$CKPT/mmeb_vqa_ood" \
    --json "$CKPT/mmeb_summary.json"
