#!/bin/bash
# MMEB evaluation on the ret_ind benchmarks, Precision@1.
#
#   bash scripts/eval/ret.sh <checkpoint-dir>
#   CKPT=<dir> BACKBONE=llava_onevision IMAGE_RESOLUTION=336 bash scripts/eval/ret.sh
#
# A thin wrapper on scripts/eval/run_group.sh, which holds the shared body and
# reads the subset list from src/evaluation/benchmarks.py.
set -euo pipefail
CKPT="${1:-${CKPT:-}}"
if [ -z "$CKPT" ]; then
    echo "usage: bash scripts/eval/ret.sh <checkpoint-dir>" >&2
    exit 1
fi
export OUT="${OUT:-$CKPT/mmeb_ret}"
exec bash "$(dirname "$0")/run_group.sh" ret_ind "$CKPT" "${@:2}"
