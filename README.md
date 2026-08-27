# VLMEmbed
## Set up env
```bash
apt-get update
apt-get upgrade -y
cd VLM_Embed
python -m venv vlm
source vlm/bin/activate
```
## Set up
```
pip install -r requirements.txt
```
## Download dataset
1. Download the eval image file zip from huggingface (`optional`) 
```bash
cd VLM_Embed
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
```
2. Download train image, it can take > 1 hour to download
```bash
cd VLM_Embed
bash scripts/data/download_traindata.sh
bash scripts/data/download_traindata_2.sh
```
3. Fix some line code 

Because of the error of code in **Transformers library**, run the following script to find the error and comment some lines: 

Just comment the following code, from line 140 to 143 in file **/vlm/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py**: 
```python
if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
    raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
else:
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
```
Or run `fix_lib.py` to fix:
```bash
python tools/misc/fix_lib.py
```


## Repository layout

Everything is run **from the repo root**; the entrypoints under `tools/` put the
root on `sys.path` themselves, so no `pip install` step is needed.

```
configs/
  deepspeed/     ds_config*.json
  projector/     projector_config*.json
  data/          train_image.yaml
docs/assets/     figures used by this README
scripts/
  data/          dataset download + encoding
  train/         training launchers
    rebuttal/    the rebuttal sweep
  eval/          evaluation launchers
tools/           python entrypoints
  train_distill_ddp.py         DDP distillation trainer (the one in use)
  train_distillation.py        DeepSpeed variant
  train_distill_no_deepspeed.py
  train_vlm2vec.py             VLM2Vec baseline trainer (was train.py)
  eval_mmeb.py  eval_mmeb_simple.py  prepare_data.py  visualizer.py
  eval_baselines/              CLIP / BLIP / SigLIP / OpenCLIP baselines
  misc/                        download, push_to_hub, fix_lib, test_load_model
src/                           the library
  arguments.py  distiller.py  profiling.py  utils.py
  criterions/                  KD losses (+ text_spans, vision_clustering)
  data/                        datasets and collators
  model/                       MMEBModel and the vendored VLM backbones
  evaluation/                  shared eval helpers
```

## Training

All launchers live under `scripts/train/` (and `scripts/train/rebuttal/` for the
rebuttal sweep). Run them **from the repo root**:

```bash
bash scripts/train/train_RKD.sh
bash scripts/train/train_distill_propose_V.sh
bash scripts/train/rebuttal/rebuttal_hierd_grounding.sh
```
## Profiling the training loop

The training loop is instrumented with a step profiler that is a no-op unless it
is switched on:

```bash
VLM2VEC_PROFILE=1 VLM2VEC_PROFILE_STEPS=100 bash scripts/train/rebuttal/rebuttal_hierd_grounding.sh
```

It prints a per-step breakdown (`data_wait`, `to_device`, `forward` ->
`teacher_fwd` / `student_fwd` / `spacy_spans` / `text_span_loss` /
`vision_cluster_loss` -> `vision_cluster` / `cross_modal_loss`, `backward`,
`optimizer`) so you can see where a step actually goes before optimising it.

| variable | default | meaning |
| --- | --- | --- |
| `VLM2VEC_PROFILE` | `0` | enable the profiler |
| `VLM2VEC_PROFILE_SYNC` | `1` | `torch.cuda.synchronize()` around each section, so CUDA async execution does not misattribute time. Adds overhead, so profiled steps are slower than real ones |
| `VLM2VEC_PROFILE_EVERY` | `50` | print a report every N optimizer steps |
| `VLM2VEC_PROFILE_STEPS` | `0` | stop after N steps (0 = never) |

### Other performance switches

| variable | default | meaning |
| --- | --- | --- |
| `VLM2VEC_SPAN_CACHE` | unset | directory to persist the spaCy span cache across runs. Unset keeps it in memory only |
| `VLM2VEC_SPACY_PROCESSES` | `1` | processes for `nlp.pipe`. Values > 1 were a large net loss at batch scale |
| `VLM2VEC_FORCE_EAGER` | `0` | force eager attention on both models (fallback if SDPA misbehaves) |
| `VLM2VEC_NO_MERGE_LORA` | `0` | keep the frozen teacher's LoRA adapters unmerged |
| `VLM2VEC_FULL_LOGITS` | `0` | run `lm_head` over the whole sequence again instead of the last position |

Dataloader workers are set with the standard HF flag, `--dataloader_num_workers`
(the training scripts pass `8`).

## Inference & Evaluation
1. To evaluate our model on an MMEB dataset (e.g., MSCOCO_i2t), run:
```bash
bash scripts/eval/eval.sh
```

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]) and [B3](https://github.com/raghavlite/B3)
