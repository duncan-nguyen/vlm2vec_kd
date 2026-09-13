#!/bin/bash
# CMTop -- LLaVA-OneVision-0.5B, CLS
# CM-Merge: distil correspondence-aware multiscale connectivity of the
# query-candidate relation rather than an identity-agnostic persistence barcode.
#
# Method:  kd_loss_type cmtop
# Teacher: raghavlite/B3_Qwen2_2B (qwen2_vl)  ->  student LLaVA-OneVision-0.5B (llava_onevision)
# Config:  paper Table 6 for this student, so the ablation is comparable to the
#          other methods; see scripts/train/cmtop/README.md and
#          docs/cmtop_implementation.md.
#
# One script, two rows: the method and its no-teacher control.
# Pick one with VARIANT and vary SEED:
#
#   VARIANT=cmmerge SEED=42 bash scripts/train/cmtop/llava_onevision_cls.sh
#   for v in cmmerge student_only; do
#     for s in 42 43 44; do VARIANT=$v SEED=$s bash scripts/train/cmtop/llava_onevision_cls.sh; done
#   done

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="tools/train_distill_no_deepspeed.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/cmtop/llava_onevision_cls.sh
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# This criterion reads nothing from the teacher but its final embedding, so it
# can train against a precomputed cache and skip the teacher forward, the
# teacher's image preprocessing and its weights in GPU memory entirely:
#
#   TASK=cls STUDENT=llava_onevision TEACHER_CACHE=cache/b3_qwen2_2b_llava_onevision_cls \
#     bash scripts/data/precompute_teacher_embeddings.sh
#   TEACHER_CACHE=cache/b3_qwen2_2b_llava_onevision_cls bash scripts/train/cmtop/llava_onevision_cls.sh
#
# The cache is fingerprinted against the teacher, the subset list and the image
# settings, so this cell needs its own -- leave TEACHER_CACHE unset to run the
# teacher live.
TEACHER_CACHE="${TEACHER_CACHE:-}"
CACHE_FLAGS=()
if [ -n "$TEACHER_CACHE" ]; then
  CACHE_FLAGS=(--teacher_embedding_cache "$TEACHER_CACHE")
fi

VARIANT="${VARIANT:-cmmerge}"

# lambda_Merge. Tune once on development data, then freeze across tasks/seeds.
CMTOP_WEIGHT="${CMTOP_WEIGHT:-1.0}"

# Each variant is one row of the experiment plan. Only the KD-side flags differ.
# CM-Merge compares distances, never embeddings, so it needs no teacher/student
# projector -- declaring one without using it makes DDP abort on multi-GPU.
case "$VARIANT" in
  cmmerge)          # the method
    KD_FLAGS=(--cmtop_weight "$CMTOP_WEIGHT") ;;
  student_only)     # the no-teacher control, same sampler and batch construction
    KD_FLAGS=(--cmtop_weight 0.0) ;;
  *)
    echo "unknown VARIANT '$VARIANT'; expected: cmmerge student_only" >&2
    exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-training/CMTop/llava_onevision_cls/${VARIANT}_seed${SEED}}"
echo "variant=$VARIANT seed=$SEED topology_batch=$((BATCH_SIZE * NUM_GPUS_PER_NODE)) -> $OUTPUT_DIR"

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
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --percent_data 1.0 \
    --image_dir "$MMEB_TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
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
    --kd_loss_type "cmtop" \
    --image_resolution "336" \
    --projector_lr 5e-4 \
    "${CACHE_FLAGS[@]+"${CACHE_FLAGS[@]}"}" \
    "${KD_FLAGS[@]}" \
    "$@"
# Anything after the script name is forwarded to the trainer and, because these
# are argparse options, a repeat overrides what is set above. Handy for a smoke
# test:  bash scripts/train/cmtop/llava_onevision_cls.sh --percent_data 0.01 \
#            --push_to_hub False --eval_after_train False
# Those last two matter: a finished run uploads itself and evaluates by
# default, and a 1% run is not a result worth collecting.
