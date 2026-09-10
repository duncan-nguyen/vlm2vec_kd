#!/bin/bash
# MMEB evaluation on one benchmark group of Table 1, Precision@1.
#
#   bash scripts/eval/run_group.sh <group> <checkpoint-dir>
#
# <group> is cls_ind, vqa_ind, cls_ood, vqa_ood, or any alias
# src/evaluation/benchmarks.py accepts (all, ind, ood, cls, vqa) -- the subset
# list is read from there, so the scripts and the post-training eval cannot
# drift apart on which benchmarks a column contains.
#
# The image resolution MUST match what the checkpoint was trained at; otherwise
# the encoder sees different preprocessing than it was distilled with. Paper
# values (Tables 6/7): FastVLM 448; LLaVA-OneVision 336, or 128 for EM-KD.
# `--eval_after_train` avoids this trap entirely by reading the resolution off
# the training arguments; see scripts/train/README.md.
set -euo pipefail

GROUP="${1:-}"
CKPT="${2:-${CKPT:-}}"
if [ -z "$GROUP" ] || [ -z "$CKPT" ]; then
    echo "usage: bash scripts/eval/run_group.sh <group> <checkpoint-dir>" >&2
    exit 1
fi

BACKBONE="${BACKBONE:-llava_qwen2}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-448}"
OUT="${OUT:-$CKPT/mmeb_$GROUP}"

# Nơi chứa ảnh MMEB-eval. Ghi đè: MMEB_EVAL_DIR=/duong/dan/khac bash ...
# Mặc định khớp với `python scripts/data/download_mmeb.py --eval`.
MMEB_EVAL_DIR="${MMEB_EVAL_DIR:-./eval_images}"

# One source of truth for the subset lists.
read -r -a SUBSETS <<<"$(python -c "
import sys; sys.path.insert(0, '.')
from src.evaluation.benchmarks import resolve_groups
print(' '.join(resolve_groups(['$GROUP'])))
")"

echo "group=$GROUP subsets=${SUBSETS[*]}"

python tools/eval_mmeb.py \
    --model_name "$CKPT" \
    --model_backbone "$BACKBONE" \
    --encode_output_path "$OUT" \
    --lora --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 16 \
    --image_dir "$MMEB_EVAL_DIR" \
    --image_resolution "$IMAGE_RESOLUTION" \
    --tgt_prefix_mod \
    "${@:3}"

python tools/summarize_mmeb.py "$OUT" --benchmarks "$GROUP"
