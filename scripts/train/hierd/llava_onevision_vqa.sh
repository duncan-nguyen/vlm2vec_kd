#!/bin/bash
# HieRD — LLaVA-OneVision-0.5B, VQA
# Paper: Table 1 (VQA block, student LLaVA-OneVision-0.5B).
# Config: Table 6 (bs 8, image resolution 336), Table 8 (lambda_struct 2.5),
#         Table 9 (word layer 0, phrase layers 18/21/24), Table 4 (min_samples 8).
# Method = --kd_loss_type span_propose_attn

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="tools/train_distill_ddp.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/hierd/llava_onevision_vqa.sh
# Mặc định khớp với thư mục mà scripts/data/download_mmeb.py giải nén ra.
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"


export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
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
    --image_dir "$MMEB_TRAIN_DIR" \
    --percent_data 1.0 \
    --output_dir "training/hierd_llava_onevision_vqa" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 2.5 \
    --w_cross_modal_loss 2.5 \
    --kd_loss_type "span_propose_attn" \
    --image_resolution "336" \
    --teacher_layer_mapping 0 22 25 28 \
    --student_layer_mapping 0 18 21 24 \
    --split_layer_mapping 0 1 4 4 4 \
    --min_samples_dbscan_teacher 8 \
    --projector_lr 5e-4