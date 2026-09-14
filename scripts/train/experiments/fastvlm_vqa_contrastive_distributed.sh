#!/bin/bash
# Proposed Method contrastive-only on VQA with DistributedSampler.

set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS=1
SEED="${SEED:-42}"

if (( GLOBAL_BATCH_SIZE % NUM_GPUS_PER_NODE != 0 )); then
    echo "Global batch $GLOBAL_BATCH_SIZE is not divisible by world_size=$NUM_GPUS_PER_NODE" >&2
    exit 2
fi
BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS_PER_NODE))
OUTPUT_DIR="${OUTPUT_DIR:-training/experiments/fastvlm_vqa/contrastive_only_distributed_seed${SEED}_gb${GLOBAL_BATCH_SIZE}_${NUM_GPUS_PER_NODE}xh200}"
HUB_PATH="ours/FastVLM-0.5B/vqa/contrastive_only_distributed_seed${SEED}_gb${GLOBAL_BATCH_SIZE}_${NUM_GPUS_PER_NODE}xh200"

echo "Prepared Proposed contrastive-only VQA: sampler=DistributedSampler seed=$SEED world_size=$NUM_GPUS_PER_NODE per_device_batch=$BATCH_SIZE grad_accum=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH_SIZE eval=vqa_ind output=$OUTPUT_DIR"
if [[ "${DRY_RUN:-False}" == "True" ]]; then
    exit 0
fi

export NUM_GPUS_PER_NODE BATCH_SIZE SEED OUTPUT_DIR
export VARIANT=student_only
exec bash scripts/train/ours/fastvlm_vqa.sh \
    --ours_task_homogeneous False \
    --eval_after_train True \
    --eval_benchmarks vqa_ind \
    --eval_fail_hard True \
    --push_to_hub True \
    --hub_track ours \
    --hub_path_in_repo "$HUB_PATH"
