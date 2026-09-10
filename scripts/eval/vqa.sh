#!/bin/bash
# MMEB evaluation on the vqa_ind benchmarks of Table 1, Precision@1.
#
#   bash scripts/eval/vqa.sh <checkpoint-dir>
#   CKPT=<dir> BACKBONE=llava_onevision IMAGE_RESOLUTION=336 bash scripts/eval/vqa.sh
#
# A thin wrapper on scripts/eval/run_group.sh, which holds the shared body and
# reads the subset list from src/evaluation/benchmarks.py.
set -euo pipefail
CKPT="${1:-${CKPT:-}}"
if [ -z "$CKPT" ]; then
    echo "usage: bash scripts/eval/vqa.sh <checkpoint-dir>" >&2
    exit 1
fi
# Keep the output directory this script has always used.
export OUT="${OUT:-$CKPT/mmeb_vqa}"
exec bash "$(dirname "$0")/run_group.sh" vqa_ind "$CKPT" "${@:2}"
