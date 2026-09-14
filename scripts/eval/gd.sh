#!/bin/bash
# MMEB evaluation on the gd_ind (visual grounding) benchmarks, Precision@1.
#
#   bash scripts/eval/gd.sh <checkpoint-dir>
#   CKPT=<dir> BACKBONE=llava_onevision IMAGE_RESOLUTION=336 bash scripts/eval/gd.sh
#
# A thin wrapper on scripts/eval/run_group.sh, which holds the shared body and
# reads the subset list from src/evaluation/benchmarks.py.
set -euo pipefail
CKPT="${1:-${CKPT:-}}"
if [ -z "$CKPT" ]; then
    echo "usage: bash scripts/eval/gd.sh <checkpoint-dir>" >&2
    exit 1
fi
export OUT="${OUT:-$CKPT/mmeb_gd}"
exec bash "$(dirname "$0")/run_group.sh" gd_ind "$CKPT" "${@:2}"
