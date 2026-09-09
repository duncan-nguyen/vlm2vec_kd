#!/bin/bash
# Proposal + projector CKA -- FastVLM-0.5B, CLS
# Span proposals compared through the layer projectors.
#
# Method:  kd_loss_type proposal_proj
# Teacher: raghavlite/B3_Qwen2_2B (qwen2_vl)  ->  student FastVLM-0.5B (llava_qwen2)
# Config:  paper Table 6 for this student; verified by tools/check_paper_settings.py
#
# ! Compatibility: this criterion indexes the student's hidden states and
# ! attention with input_ids positions taken from token-id matching
# ! (src/criterions/propose_with_proj.py). FastVLM's input_ids hold a single -200
# ! placeholder that the model expands into N vision embeddings, so the hidden
# ! states are longer than input_ids and every position after the image is off
# ! by N-1. The contrastive term is fine; the KD term is not.

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="tools/train_distill_ddp.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/pproj/fastvlm_cls.sh
# Mặc định khớp với thư mục mà scripts/data/download_mmeb.py giải nén ra.
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
OUTPUT_DIR="${OUTPUT_DIR:-training/pproj_fastvlm_cls}"
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
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --percent_data 1.0 \
    --image_dir "$MMEB_TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 16 \
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
    --kd_loss_type "proposal_proj" \
    --image_resolution "448" \
    --projector_config_path "./configs/projector/projector_config.json" \
    --projector_lr 5e-4 \
    "$@"
# Anything after the script name is forwarded to the trainer; because these are
# argparse options a repeat overrides what is set above, which makes a smoke test
# one flag away:  bash scripts/train/pproj/fastvlm_cls.sh --percent_data 0.01
