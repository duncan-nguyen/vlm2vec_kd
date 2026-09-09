#!/bin/bash
# RKD -- LLaVA-OneVision-0.5B, VQA
# Relational KD (distance + angle) on the final embeddings.
#
# Method:  kd_loss_type contrastive_rkd
# Teacher: raghavlite/B3_Qwen2_2B (qwen2_vl)  ->  student LLaVA-OneVision-0.5B (llava_onevision)
# Config:  paper Table 6 for this student; verified by tools/check_paper_settings.py

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="tools/train_distill_no_deepspeed.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/rkd/llava_onevision_vqa.sh
# Mặc định khớp với thư mục mà scripts/data/download_mmeb.py giải nén ra.
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
OUTPUT_DIR="${OUTPUT_DIR:-training/rkd_llava_onevision_vqa}"
SEED="${SEED:-42}"

# This criterion reads nothing from the teacher but its final embedding, so it
# can train against a precomputed cache and skip the teacher forward, the
# teacher's image preprocessing and its weights in GPU memory entirely:
#
#   TASK=vqa STUDENT=llava_onevision TEACHER_CACHE=cache/b3_qwen2_2b_llava_onevision_vqa \
#     bash scripts/data/precompute_teacher_embeddings.sh
#   TEACHER_CACHE=cache/b3_qwen2_2b_llava_onevision_vqa bash scripts/train/rkd/llava_onevision_vqa.sh
#
# The cache is fingerprinted against the teacher, the subset list and the image
# settings, so this cell needs its own -- leave TEACHER_CACHE unset to run the
# teacher live.
TEACHER_CACHE="${TEACHER_CACHE:-}"
CACHE_FLAGS=()
if [ -n "$TEACHER_CACHE" ]; then
  CACHE_FLAGS=(--teacher_embedding_cache "$TEACHER_CACHE")
fi

torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --dataloader_num_workers 8 \
    --model_name "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_onevision" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" \
    --dataset_split "original" \
    --percent_data 1.0 \
    --image_dir "$MMEB_TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed "$SEED" \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 0.3 \
    --kd_loss_type "contrastive_rkd" \
    --image_resolution "336" \
    --projector_lr 5e-4 \
    "${CACHE_FLAGS[@]+"${CACHE_FLAGS[@]}"}" \
    "$@"
# Anything after the script name is forwarded to the trainer; because these are
# argparse options a repeat overrides what is set above, which makes a smoke test
# one flag away:  bash scripts/train/rkd/llava_onevision_vqa.sh --percent_data 0.01
