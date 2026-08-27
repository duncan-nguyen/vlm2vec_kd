#!/bin/bash
# MMEB evaluation on the VQA benchmarks of Table 1, Precision@1.
#
#   bash scripts/eval/vqa.sh <checkpoint-dir>
#
# The image resolution MUST match what the checkpoint was trained at; otherwise
# the encoder sees different preprocessing than it was distilled with. Paper
# values (Tables 6/7): FastVLM 448; LLaVA-OneVision 336, or 128 for EM-KD.
set -e

CKPT="${1:-$CKPT}"
if [ -z "$CKPT" ]; then
    echo "usage: bash scripts/eval/vqa.sh <checkpoint-dir>" >&2
    echo "   or: CKPT=<dir> BACKBONE=llava_onevision IMAGE_RESOLUTION=336 bash scripts/eval/vqa.sh" >&2
    exit 1
fi

BACKBONE="${BACKBONE:-llava_qwen2}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-448}"
OUT="${OUT:-$CKPT/mmeb_vqa}"

# Nơi chứa ảnh MMEB-eval. Ghi đè: MMEB_EVAL_DIR=/duong/dan/khac bash scripts/eval/vqa.sh
# Mặc định khớp với `python scripts/data/download_mmeb.py --eval`.
MMEB_EVAL_DIR="${MMEB_EVAL_DIR:-./eval_images}"

python tools/eval_mmeb.py \
    --model_name "$CKPT" \
    --model_backbone "$BACKBONE" \
    --encode_output_path "$OUT" \
    --lora --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W \
    --dataset_split test \
    --per_device_eval_batch_size 16 \
    --image_dir "$MMEB_EVAL_DIR" \
    --image_resolution "$IMAGE_RESOLUTION" \
    --tgt_prefix_mod
