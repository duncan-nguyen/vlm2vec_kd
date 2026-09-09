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
  train/         <method>/<student>_<task>.sh, the full method x student x task
                 matrix + README.md
  eval/          cls.sh / vqa.sh (Table 1 benchmarks)
tools/           python entrypoints
  train_distill_ddp.py         DDP trainer, no autocast (emkd/emo/hierd/pdtw/pproj)
  train_distill_no_deepspeed.py DDP trainer under bf16 autocast (cmtop/rkd/uld/talas)
  train_distillation.py        DeepSpeed variant; no launcher uses it
  train_vlm2vec.py             VLM2Vec baseline trainer (was train.py)
  eval_mmeb.py  eval_mmeb_simple.py  prepare_data.py  visualizer.py
  eval_topology.py             teacher-vs-student topology + neighborhood metrics
  precompute_teacher_embeddings.py  build a frozen-teacher embedding cache
  eval_baselines/              CLIP / BLIP / SigLIP / OpenCLIP baselines
  misc/                        download, push_to_hub, fix_lib, test_*.py self-checks
src/                           the library
  arguments.py  distiller.py  profiling.py  utils.py  topology.py  teacher_cache.py
  criterions/                  KD losses
    registry.py                the one table of methods; add a method here
    base.py                    DistillCriterion: contrastive + gather + teacher lookup
    span_common.py             HieRD span helpers: padding-side-aware extraction,
                               cluster pooling, what may be detached
  training/                    the training loop, shared by every method
    entrypoint.py              argument parsing, model/optimizer setup
    loop.py                    DistillTrainer
    dataloader.py              worker/prefetch/pinning policy
    checkpoint.py              saving
  data/                        datasets, collators, image decoding
  model/                       MMEBModel and the vendored VLM backbones
  evaluation/                  shared eval helpers
```

Both training entrypoints are three lines over `src/training/entrypoint.py`;
they differ only in whether the forward runs under bf16 autocast. The loop never
learns a method's name — see [docs/adding_a_method.md](docs/adding_a_method.md).

### Self-checks

No GPU, no model download, a few seconds each:

```bash
python tools/misc/test_training_stack.py   # registry, criterion base, loop, images
python tools/misc/test_cmtop.py            # persistence primitives + the CMTop criterion
python tools/misc/test_talas.py            # TALAS's two losses, ASAM, the two-pass step
python tools/misc/test_teacher_cache.py    # cache format and what it refuses
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

`scripts/train/<method>/<student>_<task>.sh` — one launcher per
`method × student × task` cell, for both students (`fastvlm`,
`llava_onevision`) and both tasks (`cls`, `vqa`). Run them **from the repo
root**:

```
scripts/train/                   README.md  the matrix + compatibility notes
  rkd/           contrastive_rkd
  uld/           universal_logit
  cmtop/         cmtop                      + README.md, the 6-variant ablation
  talas/         talas                      + README.md, the 5-variant ablation
  emkd/          em_kd | em_kd_llava_ov     one criterion per student
  emo/           emo_loss
  hierd/         span_propose_attn          the paper's HieRD
  pdtw/          proposal_dtw
  pproj/         proposal_proj
```

Each directory holds `fastvlm_cls.sh`, `fastvlm_vqa.sh`,
`llava_onevision_cls.sh` and `llava_onevision_vqa.sh`.

```bash
bash scripts/train/hierd/fastvlm_cls.sh
```

[scripts/train/README.md](scripts/train/README.md) is the operational half: the
full matrix, the per-student settings each table fixes, the teacher-cache
recipes, and — the thing to read before launching a method on a student it has
not been run on — which criteria are safe on which student's token layout.

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
TASK=cls STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls \
  bash scripts/data/precompute_teacher_embeddings.sh

TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls VARIANT=cmtop_h0 SEED=42 \
  bash scripts/train/cmtop/fastvlm_cls.sh
python tools/misc/test_cmtop.py          # self-checks, no GPU or model download
python tools/misc/test_teacher_cache.py
```

`--teacher_embedding_cache` also works for `contrastive_rkd`,
`universal_logit` and `talas` — any criterion that reads only the teacher's final
embedding. Criteria that need its hidden states are refused rather than served
wrong data.

`VARIANT` selects one row of the ablation (`student_only`, `endpoint`, `vsp`,
`pointcloud_h0`, `cmtop_h0`, `cmtop_h0_h1`). Full runbook — environment, data,
smoke test, the ablation loop and evaluation:
[scripts/train/cmtop/README.md](scripts/train/cmtop/README.md). Design and flag
reference: [docs/cmtop_implementation.md](docs/cmtop_implementation.md); the
research brief it implements:
[docs/cross_modal_topological_distillation.md](docs/cross_modal_topological_distillation.md).

### TALAS

`--kd_loss_type talas` is the TALAS baseline
([docs/baseline methods/TALAS.pdf](docs/baseline%20methods/TALAS.pdf)): anchor
the student's top `K` layers to the teacher's final embedding through one
learnable projection each (`L_TAMD`), propagate that geometry down the student
itself by aligning adjacent layers' batch relation matrices (`L_LASD`), and
optimise with adaptive sharpness-aware minimization.

```bash
TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls VARIANT=talas SEED=42 \
  bash scripts/train/talas/fastvlm_cls.sh
python tools/misc/test_talas.py           # self-checks, no GPU or model download
```

ASAM is `--sharpness_aware asam` and is **not** TALAS-specific — it wraps the
optimizer, so any `--kd_loss_type` can use it. It costs two forward/backward
passes per optimizer step.

Two deliberate departures from the paper, both documented in
[docs/talas_implementation.md](docs/talas_implementation.md): the paper's
unsupervised SimCSE term is replaced by this repo's supervised in-batch
contrastive loss (and left at weight 1 rather than the paper's 0.001, so the
baseline stays comparable to the others), and the launchers use Table 6's shared
per-student settings rather than TALAS's own text-training recipe. Runbook:
[scripts/train/talas/README.md](scripts/train/talas/README.md).

**Coverage.** Every method has all four `student × task` launchers: 9 methods x
2 students x 2 tasks = 36, all checked by `tools/check_paper_settings.py`. Table
1's MSE, CKD and SFT rows are still missing because no criterion implements them.
`span_propose` and `span_propose_attn_only_phrase` are HieRD ablations rather
than methods and have no launchers of their own — run them by passing
`--kd_loss_type` to a `hierd/` launcher.

Completeness is not the same as validity. `contrastive_rkd`, `universal_logit`,
`cmtop` and `talas` read only pooled embeddings; EM-KD has one criterion per student; the
span criteria (HieRD) derive their offsets from the student's padding side in
`src/criterions/span_common.py`. The remaining three — `emo_loss`,
`proposal_dtw`, `proposal_proj` — still slice hidden states by a position
convention that holds for one student and not the other. Those launchers carry a
`# !` note in their header naming the assumption, and
[scripts/train/README.md](scripts/train/README.md#model-pair-compatibility) has
the table.

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

### Making a run faster

Roughly in order of how much they buy, for any method:

1. **Precompute the teacher's embeddings.** The teacher is frozen and the data
   is not augmented, so its embedding for a sample is a pure function of the
   dataset index. Methods that read nothing else from the teacher —
   `cmtop`, `contrastive_rkd`, `universal_logit` — can then run with no teacher
   model in the process: no teacher forward (the larger of the two models), no
   teacher-side image preprocessing, and its weights out of GPU memory, which is
   what lets the batch grow.

   ```bash
   TASK=cls STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls \
     bash scripts/data/precompute_teacher_embeddings.sh
   TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls bash scripts/train/cmtop/fastvlm_cls.sh
   ```

   One cache serves every cache-compatible method, variant and seed *at that
   cell*. It is fingerprinted against the teacher, the subset list and the image
   settings and refuses to open under a configuration it was not built for, so
   the four cache-capable cells (2 students x 2 tasks) each need their own —
   `TASK` and `STUDENT` select them.

2. **Dataloader workers.** `--dataloader_num_workers` (standard HF flag). The
   input pipeline — JPEG decode, resize, tokenise, four processor passes per
   batch — is heavy enough that running it on the process driving the GPU leaves
   the device idle. With the flag unset it now defaults to 4 rather than 0;
   `--dataloader_num_workers -1` is the opt-out. Workers are persistent and
   prefetch 4 batches ahead.

3. **Check where the time actually goes** before tuning anything else:
   `VLM2VEC_PROFILE=1 VLM2VEC_PROFILE_STEPS=100`. A large `data_wait` means the
   input pipeline; raise the worker count. A large `forward` means the models,
   and the teacher cache above is the lever.

4. **`--logging_steps`.** Every log line reads the loss components back from the
   GPU, which is a synchronisation point. The launchers pass `1`; at `20` the
   progress bar is just as useful and the loop never stalls for it.

5. **Attentions.** `output_attentions=True` pins a backbone to the eager
   attention kernel and keeps a `(B, heads, L, L)` tensor per layer alive — on
   the student side through the backward pass too. Each method declares which
   side it actually reads in `src/criterions/registry.py`, and everything else
   runs on SDPA. A new method that leaves those fields at their conservative
   default is correct but slow.

Image decoding gives libjpeg the target resolution up front, so a source headed
for `--image_resolution 448` is decoded at a DCT-scaled size instead of in full
and resized down afterwards — same geometry, a fraction of the decode. Note that
the resize itself squashes to a square rather than preserving the aspect ratio;
that is what every run in this repo has done, and `--image_keep_aspect_ratio`
switches it off without changing the default under existing command lines.

## Inference & Evaluation
1. To evaluate our model on an MMEB dataset (e.g., MSCOCO_i2t), run:
```bash
bash scripts/eval/cls.sh
```

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]) and [B3](https://github.com/raghavlite/B3)
