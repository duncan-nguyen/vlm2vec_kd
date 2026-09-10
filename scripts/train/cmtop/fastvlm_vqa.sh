#!/bin/bash
# CMTop -- FastVLM-0.5B, VQA
# Cross-modal topological distillation: distil the persistent topology of the
# query-candidate retrieval relation rather than the geometry of the two clouds.
#
# Method:  kd_loss_type cmtop
# Teacher: raghavlite/B3_Qwen2_2B (qwen2_vl)  ->  student FastVLM-0.5B (llava_qwen2)
# Config:  paper Table 6 for this student, so the ablation is comparable to the
#          other methods; see scripts/train/cmtop/README.md and
#          docs/cmtop_implementation.md.
#
# One script, six variants. Pick one with VARIANT and vary SEED:
#
#   VARIANT=cmtop_h0 SEED=42 bash scripts/train/cmtop/fastvlm_vqa.sh
#   for v in student_only endpoint vsp pointcloud_h0 cmtop_h0 cmtop_h0_h1; do
#     for s in 42 43 44; do VARIANT=$v SEED=$s bash scripts/train/cmtop/fastvlm_vqa.sh; done
#   done

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="tools/train_distill_no_deepspeed.py"

# Nơi chứa ảnh MMEB-train. Ghi đè mà không cần sửa file:
#   MMEB_TRAIN_DIR=/duong/dan/khac bash scripts/train/cmtop/fastvlm_vqa.sh
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
SEED="${SEED:-42}"

# This criterion reads nothing from the teacher but its final embedding, so it
# can train against a precomputed cache and skip the teacher forward, the
# teacher's image preprocessing and its weights in GPU memory entirely:
#
#   TASK=vqa STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_vqa \
#     bash scripts/data/precompute_teacher_embeddings.sh
#   TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_vqa bash scripts/train/cmtop/fastvlm_vqa.sh
#
# The cache is fingerprinted against the teacher, the subset list and the image
# settings, so this cell needs its own -- leave TEACHER_CACHE unset to run the
# teacher live.
TEACHER_CACHE="${TEACHER_CACHE:-}"
CACHE_FLAGS=()
if [ -n "$TEACHER_CACHE" ]; then
  CACHE_FLAGS=(--teacher_embedding_cache "$TEACHER_CACHE")
fi

VARIANT="${VARIANT:-cmtop_h0}"

# lambda_CMTop. The topological term is a squared distance between per-bar cosine
# distances, so it is small next to the contrastive loss; retune this first if the
# KD term never moves the student.
CMTOP_WEIGHT="${CMTOP_WEIGHT:-1.0}"
KD_WEIGHT="${KD_WEIGHT:-0.3}"

# Each variant is one row of the experiment plan. Only the KD-side flags differ.
# The projector only exists to map teacher embeddings into the student space for
# the endpoint term; registering it without using it makes DDP abort on multi-GPU,
# so the variant that skips the endpoint term must not declare it.
PROJECTOR_FLAGS=(--projector_config_path "configs/projector/projector_config_emo.json")

case "$VARIANT" in
  student_only)     # contrastive only, no teacher signal
    KD_FLAGS=(--kd_weight 0.0 --cmtop_weight 0.0 --cmtop_endpoint_kd none)
    PROJECTOR_FLAGS=() ;;
  endpoint)         # standard endpoint KD on the final embeddings
    KD_FLAGS=(--kd_weight "$KD_WEIGHT" --cmtop_weight 0.0 --cmtop_endpoint_kd cosine) ;;
  vsp)              # pairwise / VSP-style geometry of the same relation
    KD_FLAGS=(--kd_weight "$KD_WEIGHT" --cmtop_weight 0.0 --cmtop_endpoint_kd cosine
              --cmtop_geometry_weight 1.0) ;;
  pointcloud_h0)    # ordinary H0 of each modality's own cloud
    KD_FLAGS=(--kd_weight "$KD_WEIGHT" --cmtop_weight "$CMTOP_WEIGHT"
              --cmtop_mode point_cloud) ;;
  cmtop_h0)         # the main proposal: H0 of the cross-modal relation
    KD_FLAGS=(--kd_weight "$KD_WEIGHT" --cmtop_weight "$CMTOP_WEIGHT"
              --cmtop_mode cross_modal) ;;
  cmtop_h0_h1)      # + the lightweight H1-birth term
    KD_FLAGS=(--kd_weight "$KD_WEIGHT" --cmtop_weight "$CMTOP_WEIGHT"
              --cmtop_mode cross_modal --cmtop_h1_weight 0.1) ;;
  *)
    echo "unknown VARIANT '$VARIANT'; expected one of: student_only endpoint vsp pointcloud_h0 cmtop_h0 cmtop_h0_h1" >&2
    exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-training/CMTop/fastvlm_vqa/${VARIANT}_seed${SEED}}"
echo "variant=$VARIANT seed=$SEED -> $OUTPUT_DIR"

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
    --kd_loss_type "cmtop" \
    --image_resolution "448" \
    --projector_lr 5e-4 \
    "${CACHE_FLAGS[@]+"${CACHE_FLAGS[@]}"}" \
    "${PROJECTOR_FLAGS[@]+"${PROJECTOR_FLAGS[@]}"}" \
    "${KD_FLAGS[@]}" \
    "$@"
# Anything after the script name is forwarded to the trainer and, because these
# are argparse options, a repeat overrides what is set above. Handy for a smoke
# test:  bash scripts/train/cmtop/fastvlm_vqa.sh --percent_data 0.01 \
#            --push_to_hub False --eval_after_train False
# Those last two matter: a finished run uploads itself and evaluates by
# default, and a 1% run is not a result worth collecting.
