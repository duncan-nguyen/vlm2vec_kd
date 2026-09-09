#!/bin/bash
# TALAS -- FastVLM-0.5B, classification
# Teacher-Anchored Layer Alignment with Adaptive Sharpness-aware minimization:
# anchor the student's top K layers to the teacher's final embedding (L_TAMD),
# propagate that geometry down the student itself (L_LASD), and optimise the
# whole thing with ASAM.
#
# Method:  kd_loss_type talas   (paper: docs/baseline methods/TALAS.pdf)
# Teacher: raghavlite/B3_Qwen2_2B (qwen2_vl)  ->  student FastVLM-0.5B (llava_qwen2)
# Config:  paper Table 6 for this student, so TALAS is comparable to the other
#          baselines in this repo rather than to the TALAS paper's own text
#          setup; see scripts/train/talas/README.md and
#          docs/talas_implementation.md for what that changes.
#
# ! Layout: TALAS reads only pooled embeddings -- one per student layer, through
#   the student's own pooling and attention mask -- so it makes no assumption
#   about where the vision tokens sit. Correct for both students.
#
# One script, five variants (the paper's Table 2 ablation). Pick one with
# VARIANT and vary SEED:
#
#   VARIANT=talas SEED=42 bash scripts/train/talas/fastvlm_cls.sh
#   for v in talas no_asam sam no_lasd no_tamd; do
#     for s in 42 43 44; do VARIANT=$v SEED=$s bash scripts/train/talas/fastvlm_cls.sh; done
#   done

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="tools/train_distill_no_deepspeed.py"

# Where MMEB-train images live. Override without editing this file:
#   MMEB_TRAIN_DIR=/some/path bash scripts/train/talas/fastvlm_cls.sh
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
SEED="${SEED:-42}"

# TALAS reads nothing from the teacher but its final embedding -- appendix C of
# the paper runs the teacher exactly once, offline, for precisely this reason --
# so it trains against a precomputed cache with no teacher model in the process:
#
#   TASK=cls STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls \
#     bash scripts/data/precompute_teacher_embeddings.sh
#   TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls bash scripts/train/talas/fastvlm_cls.sh
#
# Leave TEACHER_CACHE unset to run the teacher live.
TEACHER_CACHE="${TEACHER_CACHE:-}"
CACHE_FLAGS=()
if [ -n "$TEACHER_CACHE" ]; then
  CACHE_FLAGS=(--teacher_embedding_cache "$TEACHER_CACHE")
fi

# The paper's lambdas (appendix E). lambda_1 scales the contrastive term, which
# stands in for its unsupervised SimCSE loss; the paper's 0.001 is calibrated
# for that objective on unpaired text, so the default here is the 1.0 every
# other method in this repo gives the contrastive term. Set
# TALAS_CONTRASTIVE_WEIGHT=0.001 for the paper's balance.
TALAS_CONTRASTIVE_WEIGHT="${TALAS_CONTRASTIVE_WEIGHT:-1.0}"   # lambda_1
TAMD_WEIGHT="${TAMD_WEIGHT:-0.75}"                            # lambda_2
LASD_WEIGHT="${LASD_WEIGHT:-1.0}"                             # lambda_3
# K, the number of top layers anchored to the teacher. The paper's ablation
# (Table 6) peaks at 2 and degrades past 4.
TAMD_LAYERS="${TAMD_LAYERS:-2}"
# ASAM's radius, measured in units of |w|. The paper does not report it; 0.5 is
# the low end of ASAM's own range and the first thing to retune.
SAM_RHO="${SAM_RHO:-0.5}"

VARIANT="${VARIANT:-talas}"

# No --projector_config_path: TALAS builds its own W_l, one per anchored layer,
# from --talas_num_tamd_layers. Declaring an unused `t2s` on top of them makes
# DDP abort with "expected to have finished reduction" on multi-GPU.
case "$VARIANT" in
  talas)      # the full method
    KD_FLAGS=(--sharpness_aware asam --sam_rho "$SAM_RHO") ;;
  no_asam)    # L_SimCSE + L_TAMD + L_LASD, plain AdamW
    KD_FLAGS=(--sharpness_aware none) ;;
  sam)        # non-adaptive sharpness-aware minimization (paper's Figure 2)
    KD_FLAGS=(--sharpness_aware sam --sam_rho 0.05) ;;
  no_lasd)    # drop the self-distillation term
    KD_FLAGS=(--sharpness_aware asam --sam_rho "$SAM_RHO" --talas_lasd_weight 0.0) ;;
  no_tamd)    # drop the teacher anchor -- the paper's catastrophic row
    KD_FLAGS=(--sharpness_aware asam --sam_rho "$SAM_RHO" --talas_tamd_weight 0.0) ;;
  *)
    echo "unknown VARIANT '$VARIANT'; expected one of: talas no_asam sam no_lasd no_tamd" >&2
    exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-training/TALAS/fastvlm_cls/${VARIANT}_seed${SEED}}"
echo "variant=$VARIANT seed=$SEED -> $OUTPUT_DIR"

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
    --per_device_train_batch_size 16 \
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
    --kd_loss_type "talas" \
    --image_resolution "448" \
    --projector_lr 5e-4 \
    --talas_contrastive_weight "$TALAS_CONTRASTIVE_WEIGHT" \
    --talas_tamd_weight "$TAMD_WEIGHT" \
    --talas_lasd_weight "$LASD_WEIGHT" \
    --talas_num_tamd_layers "$TAMD_LAYERS" \
    "${CACHE_FLAGS[@]+"${CACHE_FLAGS[@]}"}" \
    "${KD_FLAGS[@]}" \
    "$@"
# Anything after the script name is forwarded to the trainer and, because these
# are argparse options, a repeat overrides what is set above. Handy for a smoke
# test:  bash scripts/train/talas/fastvlm_cls.sh --percent_data 0.01
