#!/bin/bash
# EM-KD -- FastVLM-0.5B, VQA
# Hungarian-matched vision tokens + vision/language affinity.
#
# Method:  kd_loss_type em_kd
# Teacher: raghavlite/B3_Qwen2_2B (qwen2_vl)  ->  student FastVLM-0.5B (llava_qwen2)
# Config:  paper Table 7 for this student; verified by tools/check_paper_settings.py

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="tools/train_distill_ddp.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/emkd/fastvlm_vqa.sh
# Mặc định khớp với thư mục mà scripts/data/download_mmeb.py giải nén ra.
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
OUTPUT_DIR="${OUTPUT_DIR:-training/emkd_fastvlm_vqa}"
SEED="${SEED:-42}"

torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --dataloader_num_workers 8 \
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
    --kd_loss_type "em_kd" \
    --image_resolution "448" \
    --projector_config_path "./configs/projector/projector_config_emo.json" \
    --projector_lr 5e-4 \
    "$@"
# Anything after the script name is forwarded to the trainer; because these are
# argparse options a repeat overrides what is set above, which makes a smoke test
# one flag away:  bash scripts/train/emkd/fastvlm_vqa.sh --percent_data 0.01
