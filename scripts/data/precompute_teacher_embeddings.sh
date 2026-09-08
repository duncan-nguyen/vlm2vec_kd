#!/bin/bash
#
# Build the teacher embedding cache the CMTop launchers can reuse. Run once; the
# whole 6-variant x 3-seed ablation then trains without a teacher model.
#
#   TEACHER_CACHE=cache/b3_qwen2_2b_cls bash scripts/data/precompute_teacher_embeddings.sh
#
# The teacher and data settings below must match the ones in fastvlm_cls.sh --
# they are written into the cache's meta.json and re-checked at training time,
# so a mismatch is refused rather than silently used.
set -euo pipefail

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
MMEB_TRAIN_DIR="${MMEB_TRAIN_DIR:-./vlm2vec_train/MMEB-train}"
TEACHER_CACHE="${TEACHER_CACHE:?set TEACHER_CACHE to the directory to build}"
# Only the teacher runs here and only its pooled output is kept, so the batch
# can be larger than a training step's.
BATCH_SIZE="${BATCH_SIZE:-32}"

torchrun --nproc_per_node="$NUM_GPUS_PER_NODE" tools/precompute_teacher_embeddings.py \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --teacher_lora True \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --teacher_normalize True \
    --model_backbone "llava_qwen2" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --percent_data 1.0 \
    --image_dir "$MMEB_TRAIN_DIR" \
    --image_resolution "448" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --dataloader_num_workers "${NUM_WORKERS:-4}" \
    --teacher_embedding_cache "$TEACHER_CACHE" \
    --output_dir "/tmp/precompute_teacher_embeddings"
# No --bf16: MMEBModel.load() already loads the teacher in bfloat16 and the tool
# autocasts its own forward, so the flag would only add a TrainingArguments
# device check this tool never needs. --output_dir is likewise unused;
# TrainingArguments simply requires one.
