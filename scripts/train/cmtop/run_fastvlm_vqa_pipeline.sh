#!/bin/bash
# Run the FastVLM VQA CMTop pipeline on one 8-GPU node.
#
#   bash scripts/train/cmtop/run_fastvlm_vqa_pipeline.sh
#   EVAL_AFTER_TRAIN=1 bash scripts/train/cmtop/run_fastvlm_vqa_pipeline.sh
#
# The second form downloads the MMEB-eval images and evaluates
# checkpoint-final as the last step of the training run itself.
#
# The high nofile limit and conservative worker count prevent PyTorch's
# multiprocessing queues from exhausting file descriptors during the teacher
# cache pass. Override either value only after measuring the target host.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tensara/work/vlm_2_vec}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NOFILE_LIMIT="${NOFILE_LIMIT:-65535}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
HF_HOME="${HF_HOME:-/mnt/models/vlm_2_vec_storage/hf_cache}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/home/tensara/.hf_token_r3}"
TEACHER_CACHE="${TEACHER_CACHE:-cache/b3_qwen2_2b_fastvlm_vqa}"
FULL_OUTPUT="${FULL_OUTPUT:-training/CMTop/fastvlm_vqa/cmmerge_seed42}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-training/CMTop/fastvlm_vqa/cmmerge_seed42_smoke}"

# EVAL_AFTER_TRAIN=1 adds an MMEB evaluation to the end of the training run
# itself -- the subsets are sharded across the same eight ranks, and the image
# resolution and backbone come from the training arguments rather than from this
# file. It is off here, unlike the trainer's own default, only because it needs
# the 7.1 GB of MMEB-eval images; the DOWNLOAD phase below fetches them when it
# is on, and the TRAIN phase passes --eval_after_train False when it is not.
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-0}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-vqa_ind vqa_ood}"
MMEB_EVAL_DIR="${MMEB_EVAL_DIR:-/mnt/models/vlm_2_vec_storage/eval_images}"

cd "$PROJECT_DIR"
source .venv/bin/activate

hard_nofile="$(ulimit -Hn)"
target_nofile="$NOFILE_LIMIT"
if [[ "$hard_nofile" != "unlimited" ]] && (( target_nofile > hard_nofile )); then
  target_nofile="$hard_nofile"
fi
if ! ulimit -Sn "$target_nofile"; then
  echo "warning: could not raise nofile soft limit to $target_nofile" >&2
fi

export CUDA_VISIBLE_DEVICES HF_HOME
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ -z "${HF_TOKEN:-}" ]] && [[ -r "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  export HF_TOKEN
fi

mkdir -p logs
PIPELINE_LOG="logs/cmtop_vqa_pipeline.log"
CACHE_LOG="logs/cmtop_vqa_cache_fixed_8gpu.log"
SMOKE_LOG="logs/cmtop_vqa_smoke_fixed_8gpu.log"
TRAIN_LOG="logs/cmtop_vqa_train_fixed_8gpu.log"

archive_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
for log_file in "$PIPELINE_LOG" "$SMOKE_LOG" "$TRAIN_LOG"; do
  if [[ -s "$log_file" ]]; then
    mv "$log_file" "${log_file}.failed-${archive_stamp}"
  fi
done

stamp() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "$PIPELINE_LOG"
}

run_phase() {
  local phase="$1"
  local phase_log="$2"
  shift 2
  stamp "${phase}_START"
  if "$@" >"$phase_log" 2>&1; then
    stamp "${phase}_COMPLETE"
  else
    local status=$?
    stamp "${phase}_FAILED exit=${status}; tail follows"
    tail -n 100 "$phase_log" >>"$PIPELINE_LOG" || true
    return "$status"
  fi
}

stamp "PIPELINE_START devices=${CUDA_VISIBLE_DEVICES} ranks=${NUM_GPUS_PER_NODE} workers_per_rank=${NUM_WORKERS} nofile=$(ulimit -Sn)"

if .venv/bin/python -c \
  'import json, sys; sys.exit(not json.load(open(sys.argv[1])).get("complete", False))' \
  "$TEACHER_CACHE/meta.json" 2>/dev/null; then
  stamp "CACHE_REUSED path=${TEACHER_CACHE}"
else
  if [[ -s "$CACHE_LOG" ]]; then
    mv "$CACHE_LOG" "${CACHE_LOG}.failed-${archive_stamp}"
  fi
  run_phase CACHE "$CACHE_LOG" env \
    TASK=vqa \
    STUDENT=fastvlm \
    NUM_GPUS_PER_NODE="$NUM_GPUS_PER_NODE" \
    NUM_WORKERS="$NUM_WORKERS" \
    BATCH_SIZE=16 \
    TEACHER_CACHE="$TEACHER_CACHE" \
    bash scripts/data/precompute_teacher_embeddings.sh
fi

run_phase SMOKE "$SMOKE_LOG" env \
  NUM_GPUS_PER_NODE="$NUM_GPUS_PER_NODE" \
  NUM_WORKERS="$NUM_WORKERS" \
  TEACHER_CACHE="$TEACHER_CACHE" \
  VARIANT=cmmerge \
  SEED=42 \
  OUTPUT_DIR="$SMOKE_OUTPUT" \
  bash scripts/train/cmtop/fastvlm_vqa.sh \
    --max_steps 10 \
    --dataloader_num_workers "$NUM_WORKERS" \
    --save_strategy no \
    --report_to none \
    --overwrite_output_dir True \
    --push_to_hub False \
    --eval_after_train False

if [[ "$EVAL_AFTER_TRAIN" == "1" ]]; then
  read -r -a eval_groups <<<"$EVAL_BENCHMARKS"
  EVAL_FLAGS=(--eval_after_train True
              --eval_benchmarks "${eval_groups[@]}"
              --eval_image_dir "$MMEB_EVAL_DIR")
  run_phase EVAL_DOWNLOAD logs/cmtop_vqa_eval_download.log \
    env -u HF_XET_HIGH_PERFORMANCE python -u scripts/data/download_mmeb.py \
      --eval --eval-out "$MMEB_EVAL_DIR"
else
  # Explicit, not empty: the trainer evaluates by default, and this pipeline
  # only downloads the 7.1 GB of eval images in the branch above.
  EVAL_FLAGS=(--eval_after_train False)
fi

run_phase TRAIN "$TRAIN_LOG" env \
  NUM_GPUS_PER_NODE="$NUM_GPUS_PER_NODE" \
  NUM_WORKERS="$NUM_WORKERS" \
  TEACHER_CACHE="$TEACHER_CACHE" \
  VARIANT=cmmerge \
  SEED=42 \
  OUTPUT_DIR="$FULL_OUTPUT" \
  bash scripts/train/cmtop/fastvlm_vqa.sh \
    --dataloader_num_workers "$NUM_WORKERS" \
    --report_to none \
    "${EVAL_FLAGS[@]+"${EVAL_FLAGS[@]}"}"

if [[ "$EVAL_AFTER_TRAIN" == "1" ]]; then
  stamp "EVAL_COMPLETE summary=${FULL_OUTPUT}/checkpoint-final/mmeb_eval/summary.json"
fi
stamp "PIPELINE_COMPLETE output=${FULL_OUTPUT}"
