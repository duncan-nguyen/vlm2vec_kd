#!/bin/bash
# HieRD — FastVLM-0.5B, CLS
# Paper: Table 1 (CLS block, student FastVLM-0.5B); config Tables 6/8/9.
# Method = --kd_loss_type span_propose_attn

# Runtime overrides keep the paper defaults while allowing explicit DDP runs.
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-training/hierd_fastvlm_cls}"
KD_WEIGHT="${KD_WEIGHT:-2.5}"
W_CROSS_MODAL_LOSS="${W_CROSS_MODAL_LOSS:-2.5}"
HIERD_CONTRASTIVE_ONLY="${HIERD_CONTRASTIVE_ONLY:-False}"
TASK_HOMOGENEOUS_SAMPLING="${TASK_HOMOGENEOUS_SAMPLING:-False}"

GLOBAL_BATCH_SIZE=$((NUM_GPUS_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
EXPECTED_GLOBAL_BATCH_SIZE="${EXPECTED_GLOBAL_BATCH_SIZE:-$GLOBAL_BATCH_SIZE}"
if [[ "$GLOBAL_BATCH_SIZE" -ne "$EXPECTED_GLOBAL_BATCH_SIZE" ]]; then
    echo "Refusing to launch: global batch is $GLOBAL_BATCH_SIZE, expected $EXPECTED_GLOBAL_BATCH_SIZE" >&2
    exit 2
fi
echo "HieRD launch: world_size=$NUM_GPUS_PER_NODE per_device_batch=$PER_DEVICE_BATCH_SIZE grad_accum=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH_SIZE contrastive_only=$HIERD_CONTRASTIVE_ONLY task_homogeneous=$TASK_HOMOGENEOUS_SAMPLING"

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="tools/train_distill_ddp.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/hierd/fastvlm_cls.sh
# Mặc định khớp với thư mục mà scripts/data/download_mmeb.py giải nén ra.
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"


export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
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
    --image_dir "$MMEB_TRAIN_DIR" \
    --percent_data 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
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
    --kd_weight "$KD_WEIGHT" \
    --w_cross_modal_loss "$W_CROSS_MODAL_LOSS" \
    --hierd_contrastive_only "$HIERD_CONTRASTIVE_ONLY" \
    --task_homogeneous_sampling "$TASK_HOMOGENEOUS_SAMPLING" \
    --kd_loss_type "span_propose_attn" \
    --image_resolution "448" \
    --teacher_layer_mapping 0 22 25 28 \
    --student_layer_mapping 0 18 21 24 \
    --split_layer_mapping 0 1 4 4 4 \
    --min_samples_dbscan_teacher 8 \
    --projector_lr 5e-4 \
    "$@"
