#!/bin/bash
#
# Build the teacher embedding cache that the rkd / uld / cmtop launchers reuse.
# Run once per cell; every variant and seed afterwards then trains with no
# teacher model in the process at all.
#
#   TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls bash scripts/data/precompute_teacher_embeddings.sh
#   TASK=vqa STUDENT=llava_onevision TEACHER_CACHE=cache/b3_qwen2_2b_llava_onevision_vqa \
#     bash scripts/data/precompute_teacher_embeddings.sh
#
# The teacher and data settings below are written into the cache's meta.json and
# re-checked at training time, so a cache built for another subset list or image
# resolution is refused rather than silently used. That is why TASK and STUDENT
# exist: the subset list and --image_resolution are part of the cache identity,
# so each (student, task) cell needs its own cache directory.
#
# Only the teacher runs here (include_student=False in
# tools/precompute_teacher_embeddings.py); STUDENT selects the image resolution
# that student's launcher trains at, not a model that gets loaded.
set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
TEACHER_CACHE="${TEACHER_CACHE:?set TEACHER_CACHE to the directory to build}"
# Only the teacher runs here and only its pooled output is kept, so the batch
# can be larger than a training step's.
BATCH_SIZE="${BATCH_SIZE:-32}"

# Which MMEB split to encode. Must match the launcher's --subset_name exactly,
# order included: the list decides the index -> sample mapping the cache is
# keyed by.
TASK="${TASK:-cls}"
case "$TASK" in
  cls) SUBSETS=("ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397") ;;
  vqa) SUBSETS=("OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W") ;;
  *)   echo "unknown TASK '$TASK'; expected cls or vqa" >&2; exit 1 ;;
esac

# Which student the cache is for. It is not loaded; it only fixes the image
# resolution, which changes the pixels the teacher sees and therefore its
# embeddings. Table 6: FastVLM trains at 448, LLaVA-OneVision at 336.
STUDENT="${STUDENT:-fastvlm}"
case "$STUDENT" in
  fastvlm)
    MODEL_NAME="apple/FastVLM-0.5B"
    MODEL_BACKBONE="llava_qwen2"
    DEFAULT_RESOLUTION="448" ;;
  llava_onevision)
    MODEL_NAME="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    MODEL_BACKBONE="llava_onevision"
    DEFAULT_RESOLUTION="336" ;;
  *)   echo "unknown STUDENT '$STUDENT'; expected fastvlm or llava_onevision" >&2; exit 1 ;;
esac
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-$DEFAULT_RESOLUTION}"

echo "task=$TASK student=$STUDENT resolution=$IMAGE_RESOLUTION -> $TEACHER_CACHE"

torchrun --nproc_per_node="$NUM_GPUS_PER_NODE" tools/precompute_teacher_embeddings.py \
    --model_name "$MODEL_NAME" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --teacher_lora True \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --teacher_normalize True \
    --model_backbone "$MODEL_BACKBONE" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --percent_data 1.0 \
    --image_dir "$MMEB_TRAIN_DIR" \
    --image_resolution "$IMAGE_RESOLUTION" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --dataloader_num_workers "${NUM_WORKERS:-4}" \
    --teacher_embedding_cache "$TEACHER_CACHE" \
    --output_dir "/tmp/precompute_teacher_embeddings" \
    "$@"
# No --bf16: MMEBModel.load() already loads the teacher in bfloat16 and the tool
# autocasts its own forward, so the flag would only add a TrainingArguments
# device check this tool never needs. --output_dir is likewise unused;
# TrainingArguments simply requires one.
#
# percent_data is part of the fingerprint too, so a smoke-test cache built with
# a forwarded  --percent_data 0.01  is refused by a full run rather than reused.
