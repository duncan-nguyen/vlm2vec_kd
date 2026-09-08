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

`scripts/data/download_mmeb.py` fetches archives in parallel, extracts each one
while the next is still downloading, resumes interrupted transfers, and skips
subsets that are already extracted — so re-running it is free.

**Download only what the run needs.** MMEB-train is 47.2 GB in total, but no
single experiment uses all of it. Point the tool at a launcher and it reads the
`--subset_name` list straight out of it:

```bash
python scripts/data/download_mmeb.py --for scripts/train/hierd/fastvlm_cls.sh
```

or pick a preset:

| preset | subsets | size |
| --- | --- | --- |
| `grounding` | MSCOCO | 3.6 GB |
| `cls` | ImageNet_1K, N24News, HatefulMemes, VOC2007, SUN397 | 11.9 GB |
| `ret` | VisDial, CIRR, VisualNews_{i2t,t2i}, MSCOCO_{i2t,t2i}, NIGHTS, WebQA | 14.0 GB |
| `vqa` | OK-VQA, A-OKVQA, DocVQA, InfographicsVQA, ChartQA, Visual7W | 16.6 GB |
| `all` | everything | 47.2 GB |

```bash
python scripts/data/download_mmeb.py --preset cls
python scripts/data/download_mmeb.py --preset all --dry-run   # show the plan only
python scripts/data/download_mmeb.py --eval                   # MMEB-eval images, 7.1 GB
```

`bash scripts/data/download_traindata.sh` still works and is equivalent to
`--preset all`.

**Two things worth setting before a large download:**

```bash
export HF_TOKEN=hf_...                        # the Hub throttles anonymous traffic
pip install "huggingface_hub[hf_xet]"         # MMEB is on Xet storage: parallel chunks
```

`--workers N` (default 4) controls how many archives download at once. It only
helps if one stream does not already saturate the link — check with `--workers 1`
first if you are unsure.
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
  data/          download_mmeb.py, precompute_teacher_embeddings.sh + encoding
  train/         <method>/<student>_<task>.sh, one per main-table cell
  eval/          cls.sh / vqa.sh (Table 1 benchmarks)
tools/           python entrypoints
  train_distill_ddp.py         DDP distillation trainer (the one in use)
  train_distillation.py        DeepSpeed variant
  train_distill_no_deepspeed.py
  train_vlm2vec.py             VLM2Vec baseline trainer (was train.py)
  eval_mmeb.py  eval_mmeb_simple.py  prepare_data.py  visualizer.py
  eval_topology.py             teacher-vs-student topology + neighborhood metrics
  precompute_teacher_embeddings.py  build a frozen-teacher embedding cache
  eval_baselines/              CLIP / BLIP / SigLIP / OpenCLIP baselines
  misc/                        download, push_to_hub, fix_lib, test_load_model
src/                           the library
  arguments.py  distiller.py  profiling.py  utils.py  topology.py  teacher_cache.py
  criterions/                  KD losses (+ text_spans, vision_clustering)
  data/                        datasets and collators
  model/                       MMEBModel and the vendored VLM backbones
  evaluation/                  shared eval helpers
```

## Training

Image locations are not baked into the launchers. Both default to what
`scripts/data/download_mmeb.py` produces, and either can be overridden per run:

```bash
MMEB_TRAIN_DIR=/data/MMEB-train bash scripts/train/hierd/fastvlm_cls.sh
MMEB_EVAL_DIR=/data/eval_images  bash scripts/eval/cls.sh
```

| variable | default | used by |
| --- | --- | --- |
| `MMEB_TRAIN_DIR` | `./vlm2vec_train/MMEB-train` | every `scripts/train/**` launcher, `prepare_encoded_data.sh` |
| `MMEB_EVAL_DIR` | `./eval_images` | `scripts/eval/cls.sh`, `vqa.sh` |

`scripts/train/<method>/<student>_<task>.sh` — one launcher per cell of the
paper's main table (Table 1). Run them **from the repo root**:

```
scripts/train/
  hierd/   fastvlm_cls.sh  fastvlm_vqa.sh  llava_onevision_cls.sh  llava_onevision_vqa.sh
  rkd/     fastvlm_cls.sh
  emkd/    fastvlm_cls.sh  llava_onevision_cls.sh  llava_onevision_vqa.sh
  emo/     llava_onevision_cls.sh
  cmtop/   fastvlm_cls.sh + README.md   (research line, not a Table 1 cell)
```

```bash
bash scripts/train/hierd/fastvlm_cls.sh
```

Every launcher is pinned to the paper's configuration — Table 6 (all methods but
EM-KD), Table 7 (EM-KD), Table 8 (loss weights), Table 9 (layer selection) and
Table 4 (DBSCAN `min_samples`). Those tables are transcribed into a checker so
the launchers cannot drift:

```bash
python tools/check_paper_settings.py         # non-zero exit on any mismatch
```

HieRD is `--kd_loss_type span_propose_attn`. The student is decided by
`--model_name`: `MMEBModel.build()` reads the backbone out of that checkpoint's
config and overwrites whatever `--model_backbone` says.

**Image resolution.** Pass an explicit pixel budget (`448`, `336`, `128`), not
the `high`/`mid`/`low` presets: the presets disagree between the training and
evaluation code paths (`low` is 448 in `src/distiller.py` and 128 in
`src/data/dataset/mmeb_dataset.py`), so a preset silently changes preprocessing
between the two.

### Cross-modal topological distillation (CMTop)

`--kd_loss_type cmtop` distils the persistent topology of the query-candidate
retrieval *relation* rather than the geometry of the two point clouds. It is a
research line on top of the paper's table, not one of its cells; the launcher
still pins every shared hyperparameter to Table 6 so the ablation is clean.

```bash
# optional but recommended: encode the frozen teacher once, then every variant
# and seed trains with no teacher model in the process at all
TEACHER_CACHE=cache/b3_qwen2_2b_cls bash scripts/data/precompute_teacher_embeddings.sh

TEACHER_CACHE=cache/b3_qwen2_2b_cls VARIANT=cmtop_h0 SEED=42 bash scripts/train/cmtop/fastvlm_cls.sh
python tools/misc/test_cmtop.py          # self-checks, no GPU or model download
python tools/misc/test_teacher_cache.py
```

`--teacher_embedding_cache` also works for `contrastive_rkd` and
`universal_logit` — any criterion that reads only the teacher's final embedding.
Criteria that need its hidden states are refused rather than served wrong data.

`VARIANT` selects one row of the ablation (`student_only`, `endpoint`, `vsp`,
`pointcloud_h0`, `cmtop_h0`, `cmtop_h0_h1`). Full runbook — environment, data,
smoke test, the ablation loop and evaluation:
[scripts/train/cmtop/README.md](scripts/train/cmtop/README.md). Design and flag
reference: [docs/cmtop_implementation.md](docs/cmtop_implementation.md); the
research brief it implements:
[docs/cross_modal_topological_distillation.md](docs/cross_modal_topological_distillation.md).

**Coverage.** Table 1 is 7 methods x 2 students x 2 tasks = 28 cells. Nine exist
here. Missing: every MSE, CKD and SFT cell (no criterion in `src/criterions/`),
plus RKD and EMO outside their single CLS cell, and EM-KD/FastVLM/VQA.

## Evaluation

```bash
bash scripts/eval/cls.sh training/hierd_fastvlm_cls/checkpoint-final
CKPT=training/hierd_llava_onevision_vqa/checkpoint-final \
  BACKBONE=llava_onevision IMAGE_RESOLUTION=336 bash scripts/eval/vqa.sh
```

`cls.sh` and `vqa.sh` cover the 5 and 6 Table 1 datasets respectively and report
Precision@1. `IMAGE_RESOLUTION` must match the value the checkpoint was trained
at, and `BACKBONE` must match its student.

| variable | default |
| --- | --- |
| `BACKBONE` | `llava_qwen2` (FastVLM) |
| `IMAGE_RESOLUTION` | `448` |
| `OUT` | `<checkpoint>/mmeb_<task>` |
| `MMEB_EVAL_DIR` | `./eval_images` |

## Profiling the training loop

The training loop is instrumented with a step profiler that is a no-op unless it
is switched on:

```bash
VLM2VEC_PROFILE=1 VLM2VEC_PROFILE_STEPS=100 bash scripts/train/hierd/fastvlm_cls.sh
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
bash scripts/eval/cls.sh
```

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]) and [B3](https://github.com/raghavlite/B3)
