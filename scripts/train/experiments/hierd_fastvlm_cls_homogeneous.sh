#!/bin/bash
# Full HieRD on CLS with one task-homogeneous global batch per step.

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SEED="${SEED:-42}"

denominator=$((NUM_GPUS_PER_NODE * GRADIENT_ACCUMULATION_STEPS))
if (( GLOBAL_BATCH_SIZE % denominator != 0 )); then
    echo "Global batch $GLOBAL_BATCH_SIZE is not divisible by world_size*grad_accum=$denominator" >&2
    exit 2
fi
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / denominator))
OUTPUT_DIR="${OUTPUT_DIR:-training/experiments/hierd_fastvlm_cls/homogeneous_seed${SEED}_gb${GLOBAL_BATCH_SIZE}_${NUM_GPUS_PER_NODE}xh200}"
HUB_PATH="span_propose_attn/FastVLM-0.5B/cls/homogeneous_seed${SEED}_gb${GLOBAL_BATCH_SIZE}_${NUM_GPUS_PER_NODE}xh200"

echo "Prepared HieRD CLS: sampler=TaskHomogeneousSampler seed=$SEED world_size=$NUM_GPUS_PER_NODE per_device_batch=$PER_DEVICE_BATCH_SIZE grad_accum=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH_SIZE eval=cls_ind output=$OUTPUT_DIR"
if [[ "${DRY_RUN:-False}" == "True" ]]; then
    exit 0
fi

export NUM_GPUS_PER_NODE PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS SEED OUTPUT_DIR
export EXPECTED_GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
export TASK_HOMOGENEOUS_SAMPLING=True
export HIERD_CONTRASTIVE_ONLY=False
exec bash scripts/train/hierd/fastvlm_cls.sh \
    --eval_after_train True \
    --eval_benchmarks cls_ind \
    --eval_fail_hard True \
    --push_to_hub True \
    --hub_track baseline \
    --hub_path_in_repo "$HUB_PATH"
