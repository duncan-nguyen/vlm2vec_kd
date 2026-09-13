#!/bin/bash
# Ours -- LLaVA-OneVision-0.5B, visual grounding
# Ours: distil correspondence-aware multiscale connectivity of the
# query-candidate relation rather than an identity-agnostic persistence barcode.
#
# Method:  kd_loss_type ours
# Teacher: raghavlite/B3_Qwen2_2B (qwen2_vl)  ->  student LLaVA-OneVision-0.5B (llava_onevision)
# Config:  paper Table 6 for this student; HieRD reports grounding under "the
#          same distillation setup as in the main experiments". See
#          scripts/train/ours/README.md, docs/ours/ours_implementation.md and,
#          for the subsets and their IND/OOD split, docs/datasets.md.
#
# Grounding trains on one subset, MSCOCO (100K rows, about 43K distinct object
# crops), so the task-homogeneous sampler has nothing to separate and candidate
# deduplication does most of the batch-level work. Queries are full images and
# candidates are crops, so both sides of the pair carry an image.
#
# One script, three rows: the method, its no-teacher control and its
# L_topo-only ablation.
# Pick one with VARIANT and vary SEED:
#
#   VARIANT=ours SEED=42 bash scripts/train/ours/llava_onevision_grounding.sh
#   for v in ours student_only; do
#     for s in 42 43 44; do VARIANT=$v SEED=$s bash scripts/train/ours/llava_onevision_grounding.sh; done
#   done

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="tools/train_distill_no_deepspeed.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/ours/llava_onevision_grounding.sh
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# This criterion reads nothing from the teacher but its final embedding, so it
# can train against a precomputed cache and skip the teacher forward, the
# teacher's image preprocessing and its weights in GPU memory entirely:
#
#   TASK=grounding STUDENT=llava_onevision TEACHER_CACHE=cache/b3_qwen2_2b_llava_onevision_grounding \
#     bash scripts/data/precompute_teacher_embeddings.sh
#   TEACHER_CACHE=cache/b3_qwen2_2b_llava_onevision_grounding bash scripts/train/ours/llava_onevision_grounding.sh
#
# The cache is fingerprinted against the teacher, the subset list and the image
# settings, so this cell needs its own -- leave TEACHER_CACHE unset to run the
# teacher live.
TEACHER_CACHE="${TEACHER_CACHE:-}"
CACHE_FLAGS=()
if [ -n "$TEACHER_CACHE" ]; then
  CACHE_FLAGS=(--teacher_embedding_cache "$TEACHER_CACHE")
fi

VARIANT="${VARIANT:-ours}"

# lambda_topo. Tune once on development data, then freeze across tasks/seeds.
OURS_WEIGHT="${OURS_WEIGHT:-1.0}"

# Each variant is one row of the experiment plan. Only the KD-side flags differ.
# Ours compares distances, never embeddings, so it needs no teacher/student
# projector -- declaring one without using it makes DDP abort on multi-GPU.
case "$VARIANT" in
  ours)          # the method
    KD_FLAGS=(--ours_weight "$OURS_WEIGHT") ;;
  student_only)     # the no-teacher control, same sampler and batch construction
    KD_FLAGS=(--ours_weight 0.0) ;;
  topo_only)        # L_topo alone: the retrieval loss is dropped, lambda unchanged
    KD_FLAGS=(--ours_weight "$OURS_WEIGHT" --ours_retrieval_loss False) ;;
  *)
    echo "unknown VARIANT '$VARIANT'; expected: ours student_only topo_only" >&2
    exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-training/ours/llava_onevision_grounding/${VARIANT}_seed${SEED}}"
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
    --subset_name "MSCOCO" \
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
    --kd_loss_type "ours" \
    --image_resolution "336" \
    --projector_lr 5e-4 \
    "${CACHE_FLAGS[@]+"${CACHE_FLAGS[@]}"}" \
    "${KD_FLAGS[@]}" \
    "$@"
# Anything after the script name is forwarded to the trainer and, because these
# are argparse options, a repeat overrides what is set above. Handy for a smoke
# test:  bash scripts/train/ours/llava_onevision_grounding.sh --percent_data 0.01 \
#            --push_to_hub False --eval_after_train False
# Those last two matter: a finished run uploads itself and evaluates by
# default, and a 1% run is not a result worth collecting. The evaluation picks
# the gd_ind and gd_ood groups on its own (--eval_benchmarks auto).
