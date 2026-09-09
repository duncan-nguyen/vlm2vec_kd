#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="tools/train_distill_no_deepspeed.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/rkd/fastvlm_cls.sh
# Mặc định khớp với thư mục mà scripts/data/download_mmeb.py giải nén ra.
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"

# RKD reads nothing from the teacher but its final embedding, so it can train
# against a precomputed cache and skip the teacher forward, the teacher's image
# preprocessing and the teacher's weights in GPU memory entirely:
#
#   TEACHER_CACHE=cache/b3_qwen2_2b_cls bash scripts/data/precompute_teacher_embeddings.sh
#   TEACHER_CACHE=cache/b3_qwen2_2b_cls bash scripts/train/rkd/fastvlm_cls.sh
#
# The same cache serves cmtop, contrastive_rkd and universal_logit. Leave unset
# to run the teacher live.
TEACHER_CACHE="${TEACHER_CACHE:-}"
CACHE_FLAGS=()
if [ -n "$TEACHER_CACHE" ]; then
  CACHE_FLAGS=(--teacher_embedding_cache "$TEACHER_CACHE")
fi

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --percent_data 1.0 \
    --image_dir "$MMEB_TRAIN_DIR" \
    --output_dir "training/RKD" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 0.3 \
    --kd_loss_type "contrastive_rkd" \
    --image_resolution "448" \
    --projector_lr 5e-4 \
    "${CACHE_FLAGS[@]+"${CACHE_FLAGS[@]}"}" \
    "$@"
# Anything after the script name is forwarded to the trainer; because these are
# argparse options a repeat overrides what is set above, which makes a smoke
# test one flag away:  bash scripts/train/rkd/fastvlm_cls.sh --percent_data 0.01
