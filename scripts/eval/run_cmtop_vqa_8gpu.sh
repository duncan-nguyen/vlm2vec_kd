#!/bin/bash
# Evaluate CM-Merge FastVLM VQA on all six MMEB benchmarks while keeping all
# eight GPUs useful: six student subset jobs plus two teacher embedding jobs.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tensara/work/vlm_2_vec}"
CKPT="${CKPT:-training/CMTop/fastvlm_vqa/cmmerge_seed42/checkpoint-final}"
MMEB_EVAL_DIR="${MMEB_EVAL_DIR:-/mnt/models/vlm_2_vec_storage/eval_images}"
STUDENT_OUT="${STUDENT_OUT:-$CKPT/mmeb_vqa}"
TEACHER_OUT="${TEACHER_OUT:-runs/teacher_emb_vqa}"
TOPOLOGY_OUT="${TOPOLOGY_OUT:-runs/topo_cmmerge_seed42_vqa.json}"
TOPOLOGY_BATCH_SIZE="${TOPOLOGY_BATCH_SIZE:-128}"
HF_HOME="${HF_HOME:-/mnt/models/vlm_2_vec_storage/hf_cache}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/home/tensara/.hf_token_r3}"

SUBSETS=(OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W)
TEACHER_GROUP_6=(OK-VQA A-OKVQA DocVQA)
TEACHER_GROUP_7=(InfographicsVQA ChartQA Visual7W)

cd "$PROJECT_DIR"
source .venv/bin/activate
ulimit -Sn 65535 2>/dev/null || true
export HF_HOME
export TOKENIZERS_PARALLELISM=false
if [[ -z "${HF_TOKEN:-}" ]] && [[ -r "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  export HF_TOKEN
fi

mkdir -p logs "$STUDENT_OUT" "$TEACHER_OUT" "$(dirname "$TOPOLOGY_OUT")"
PIPELINE_LOG=logs/cmtop_vqa_eval_8gpu.log
DOWNLOAD_LOG=logs/cmtop_vqa_eval_download.log
TOPOLOGY_LOG=logs/cmtop_vqa_eval_topology.log

stamp() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "$PIPELINE_LOG"
}

stamp "EVAL_PIPELINE_START checkpoint=${CKPT}"
stamp "DOWNLOAD_START target=${MMEB_EVAL_DIR}"
download_status=1
for attempt in 1 2 3 4; do
  stamp "DOWNLOAD_ATTEMPT ${attempt}/4"
  if (( attempt < 4 )); then
    # Standard Xet concurrency is more stable on this host than
    # HF_XET_HIGH_PERFORMANCE's 100+ simultaneous range requests.
    if env -u HF_XET_HIGH_PERFORMANCE python -u scripts/data/download_mmeb.py \
        --eval --eval-out "$MMEB_EVAL_DIR" >>"$DOWNLOAD_LOG" 2>&1; then
      download_status=0
      break
    fi
  else
    # Last resort: preserve the same incomplete file but bypass the Xet client.
    if env -u HF_XET_HIGH_PERFORMANCE HF_HUB_DISABLE_XET=1 \
        python -u scripts/data/download_mmeb.py --eval \
        --eval-out "$MMEB_EVAL_DIR" >>"$DOWNLOAD_LOG" 2>&1; then
      download_status=0
      break
    fi
  fi
  stamp "DOWNLOAD_ATTEMPT_FAILED ${attempt}/4; retrying with preserved partial"
  sleep $((attempt * 10))
done
if (( download_status == 0 )); then
  stamp "DOWNLOAD_COMPLETE"
else
  stamp "DOWNLOAD_FAILED attempts=4"
  tail -n 80 "$DOWNLOAD_LOG" >>"$PIPELINE_LOG" || true
  exit 1
fi

student_eval() {
  local gpu="$1"
  local subset="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python tools/eval_mmeb.py \
    --model_name "$CKPT" \
    --model_backbone llava_qwen2 \
    --encode_output_path "$STUDENT_OUT" \
    --lora --lora_r 64 --lora_alpha 64 \
    --pooling eos --normalize True --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "$subset" --dataset_split test \
    --per_device_eval_batch_size 16 \
    --image_dir "$MMEB_EVAL_DIR" --image_resolution 448 \
    --tgt_prefix_mod
}

teacher_eval() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" python tools/eval_mmeb.py \
    --model_name raghavlite/B3_Qwen2_2B \
    --model_backbone qwen2_vl \
    --encode_output_path "$TEACHER_OUT" \
    --lora --lora_r 8 \
    --pooling eos --normalize True --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "$@" --dataset_split test \
    --per_device_eval_batch_size 16 \
    --image_dir "$MMEB_EVAL_DIR" --image_resolution 448 \
    --tgt_prefix_mod
}

stamp "ENCODE_START student_gpus=0-5 teacher_gpus=6-7"
pids=()
labels=()
for gpu in 0 1 2 3 4 5; do
  subset="${SUBSETS[$gpu]}"
  student_eval "$gpu" "$subset" >"logs/eval_student_${subset}.log" 2>&1 &
  pids+=("$!")
  labels+=("student:${subset}:gpu${gpu}")
done
teacher_eval 6 "${TEACHER_GROUP_6[@]}" >logs/eval_teacher_gpu6.log 2>&1 &
pids+=("$!")
labels+=("teacher:group1:gpu6")
teacher_eval 7 "${TEACHER_GROUP_7[@]}" >logs/eval_teacher_gpu7.log 2>&1 &
pids+=("$!")
labels+=("teacher:group2:gpu7")

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    stamp "ENCODE_JOB_COMPLETE ${labels[$index]}"
  else
    status=$?
    stamp "ENCODE_JOB_FAILED ${labels[$index]} exit=${status}"
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi
stamp "ENCODE_COMPLETE"

stamp "TOPOLOGY_START"
if python tools/eval_topology.py \
    --teacher_embeddings "$TEACHER_OUT" \
    --student_embeddings "$STUDENT_OUT" \
    --subsets "${SUBSETS[@]}" \
    --batch_size "$TOPOLOGY_BATCH_SIZE" \
    --output "$TOPOLOGY_OUT" >"$TOPOLOGY_LOG" 2>&1; then
  stamp "TOPOLOGY_COMPLETE output=${TOPOLOGY_OUT}"
else
  status=$?
  stamp "TOPOLOGY_FAILED exit=${status}"
  tail -n 80 "$TOPOLOGY_LOG" >>"$PIPELINE_LOG" || true
  exit "$status"
fi

python - "$STUDENT_OUT" "${SUBSETS[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
scores = {}
for subset in sys.argv[2:]:
    with (root / f"{subset}_score.json").open() as handle:
        scores[subset] = json.load(handle)["acc"]
scores["macro_average"] = sum(scores.values()) / len(scores)
output = root / "summary.json"
output.write_text(json.dumps(scores, indent=2) + "\n")
print(json.dumps(scores, indent=2))
PY

stamp "EVAL_PIPELINE_COMPLETE summary=${STUDENT_OUT}/summary.json"
