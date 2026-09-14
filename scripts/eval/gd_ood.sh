#!/bin/bash
# MMEB evaluation on the gd_ood (visual grounding) benchmarks, Precision@1.
#
#   bash scripts/eval/gd_ood.sh <checkpoint-dir>
#   CKPT=<dir> BACKBONE=llava_onevision IMAGE_RESOLUTION=336 bash scripts/eval/gd_ood.sh
#
# A thin wrapper on scripts/eval/run_group.sh, which holds the shared body and
# reads the subset list from src/evaluation/benchmarks.py.
set -euo pipefail
CKPT="${1:-${CKPT:-}}"
if [ -z "$CKPT" ]; then
    echo "usage: bash scripts/eval/gd_ood.sh <checkpoint-dir>" >&2
    exit 1
fi
export OUT="${OUT:-$CKPT/mmeb_gd_ood}"
exec bash "$(dirname "$0")/run_group.sh" gd_ood "$CKPT" "${@:2}"
