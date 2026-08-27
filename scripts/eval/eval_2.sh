# Nơi chứa ảnh MMEB-eval. Ghi đè: MMEB_EVAL_DIR=/duong/dan/khac bash scripts/eval/eval_2.sh
# Mặc định khớp với `python scripts/data/download_mmeb.py --eval`.
MMEB_EVAL_DIR="${MMEB_EVAL_DIR:-./eval_images}"

python  tools/eval_mmeb.py  --model_name /workspace/ComfyUI/models/gligen/VLM_Embed/training/deepspeed_projector_grounding/checkpoint-epoch1 \
                    --encode_output_path  ./MMEB-evaloutputs/ret_v1/ \
                    --pooling  eos \
                    --model_backbone  llava_qwen2 \
                    --lora --lora_r 64 --lora_alpha 64 \
                    --normalize   True  --bf16  \
                    --dataset_name  TIGER-Lab/MMEB-eval \
                    --subset_name MSCOCO \
                    --dataset_split  test  \
                    --per_device_eval_batch_size  16 \
                    --image_dir  "$MMEB_EVAL_DIR" \
                    --tgt_prefix_mod