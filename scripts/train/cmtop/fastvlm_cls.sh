#!/bin/bash
#
# CM-Merge on the MMEB classification split.
# See docs/cross_modal_topological_distillation.md and docs/cmtop_implementation.md.
#
# One script, two rows: the method and its no-teacher control.
# Pick one with VARIANT and vary SEED for the multi-seed runs the plan asks for:
#
#   VARIANT=cmmerge SEED=42 bash scripts/train/cmtop/fastvlm_cls.sh
#   for v in cmmerge student_only; do
#     for s in 42 43 44; do VARIANT=$v SEED=$s bash scripts/train/cmtop/fastvlm_cls.sh; done
#   done
#
set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="tools/train_distill_no_deepspeed.py"

# Where MMEB-train images live. Override without editing this file:
#   MMEB_TRAIN_DIR=/some/path bash scripts/train/cmtop/fastvlm_cls.sh
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"

VARIANT="${VARIANT:-cmmerge}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Precomputed teacher embeddings. The teacher is frozen and the data is not
# augmented, so its embeddings are the same on every run -- build them once and
# every variant and seed below skips the teacher forward, the teacher's image
# preprocessing, and the teacher's weights in GPU memory:
#
#   TEACHER_CACHE=cache/b3_qwen2_2b_cls bash scripts/data/precompute_teacher_embeddings.sh
#   TEACHER_CACHE=cache/b3_qwen2_2b_cls VARIANT=cmmerge bash scripts/train/cmtop/fastvlm_cls.sh
#
# Leave TEACHER_CACHE unset to run the teacher live.
TEACHER_CACHE="${TEACHER_CACHE:-}"
CACHE_FLAGS=()
if [ -n "$TEACHER_CACHE" ]; then
  CACHE_FLAGS=(--teacher_embedding_cache "$TEACHER_CACHE")
fi
# lambda_Merge. One global coefficient for the identity-aligned merge-time discrepancy.
# Tune it on development data, then freeze it across tasks and seeds.
CMTOP_WEIGHT="${CMTOP_WEIGHT:-1.0}"

# Each variant is one row of the experiment plan. Only the KD-side flags differ;
# everything below the case block is held fixed across the comparison.
# CM-Merge compares distances, never embeddings, so it needs no teacher/student
# projector -- declaring one without using it makes DDP abort on multi-GPU
# ("expected to have finished reduction").
case "$VARIANT" in
  cmmerge)          # the method
    KD_FLAGS=(--cmtop_weight "$CMTOP_WEIGHT") ;;
  student_only)     # the no-teacher control, same sampler and batch construction
    KD_FLAGS=(--cmtop_weight 0.0) ;;
  *)
    echo "unknown VARIANT '$VARIANT'; expected: cmmerge student_only" >&2
    exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-training/CMTop/${VARIANT}_seed${SEED}}"
echo "variant=$VARIANT seed=$SEED topology_batch=$((BATCH_SIZE * NUM_GPUS_PER_NODE)) -> $OUTPUT_DIR"

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
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed "$SEED" \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_loss_type "cmtop" \
    --image_resolution "448" \
    --projector_lr 5e-4 \
    "${CACHE_FLAGS[@]+"${CACHE_FLAGS[@]}"}" \
    "${KD_FLAGS[@]}" \
    "$@"
# Anything after the script name is forwarded to the trainer and, because these
# are argparse options, a repeat overrides what is set above. Handy for a smoke
# test:  bash scripts/train/cmtop/fastvlm_cls.sh --percent_data 0.01 \
#            --push_to_hub False --eval_after_train False
# Those last two matter: a finished run uploads itself and evaluates by
# default, and a 1% run is not a result worth collecting.
