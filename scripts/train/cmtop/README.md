# CMTop — how to run it

Cross-modal topological distillation: distil the persistent topology of the
query–candidate *retrieval relation* rather than the geometry of the image and
text point clouds taken separately.

* the research brief: [docs/cross_modal_topological_distillation.md](../../../docs/cross_modal_topological_distillation.md)
* the design, the maths and the flag reference: [docs/cmtop_implementation.md](../../../docs/cmtop_implementation.md)

This file is the operational half: what to run, in what order.

---

## 0. Environment

The repo's checked-in `.venv` is empty. On the training box:

```bash
python -m venv .venv && source .venv/bin/activate    # .python-version pins 3.13
pip install -r requirements.txt
pip install "huggingface_hub[hf_xet]"
export HF_TOKEN=hf_...                                # the Hub throttles anonymous traffic
```

If `transformers/models/qwen2_vl/image_processing_qwen2_vl.py` raises, the repo
ships `tools/misc/fix_lib.py` to comment out the offending lines — but it
hard-codes `./vlm/lib/python3.11/...`, so point it at the real site-packages of
your env before running it.

## 1. Data

```bash
python scripts/data/download_mmeb.py --for scripts/train/cmtop/fastvlm_cls.sh   # 11.9 GB, the 5 CLS subsets
python scripts/data/download_mmeb.py --eval                                     # 7.1 GB, MMEB-eval images
```

`--for` reads the `--subset_name` list straight out of the launcher, so the
download always matches what will be trained on. Images elsewhere:
`export MMEB_TRAIN_DIR=... MMEB_EVAL_DIR=...`.

## 2. Smoke test — do not skip this

The self-checks need no GPU, no model download and no dataset:

```bash
python tools/misc/test_cmtop.py           # 39 checks: persistence maths + the criterion
python tools/misc/test_teacher_cache.py   # 34 checks: the cache, mostly what it refuses
```

Then a real but tiny training run. Anything after the script name is forwarded
to the trainer, and a repeated argparse option overrides the launcher's value:

```bash
VARIANT=cmtop_h0 bash scripts/train/cmtop/fastvlm_cls.sh --percent_data 0.01
```

Watch `kd_loss` in the progress bar: it must be non-zero and finite. If it sits
at exactly 0 for a `cmtop_*` variant, the topological term is not contributing
and `--cmtop_weight` is the first thing to look at.

## 3. Precompute the teacher (recommended)

```bash
TEACHER_CACHE=cache/b3_qwen2_2b_cls bash scripts/data/precompute_teacher_embeddings.sh
```

The teacher is frozen and the data is not augmented, so its embedding for a
sample is a pure function of the dataset index. Encoding it once means every
variant and seed afterwards runs with **no teacher model in the process at
all** — no teacher forward (~60% of a step's FLOPs), no teacher image
preprocessing, no teacher weights on the device. For the 6×3 ablation that is
one pass over the data instead of eighteen.

Costs about 6 KB per sample on disk (`(N, 2, dim)` float16).

**`percent_data` is part of the cache fingerprint.** A cache built with
`--percent_data 0.01` is only valid for runs at 0.01 and will be *refused*, not
silently misused, by a full run. Build a separate cache for the smoke test.

## 4. Train

```bash
# one configuration
TEACHER_CACHE=cache/b3_qwen2_2b_cls VARIANT=cmtop_h0 SEED=42 \
  bash scripts/train/cmtop/fastvlm_cls.sh

# the full ablation from section 4 of the brief
for v in student_only endpoint vsp pointcloud_h0 cmtop_h0 cmtop_h0_h1; do
  for s in 42 43 44; do
    TEACHER_CACHE=cache/b3_qwen2_2b_cls VARIANT=$v SEED=$s \
      bash scripts/train/cmtop/fastvlm_cls.sh
  done
done
```

Checkpoints land in `training/CMTop/<variant>_seed<seed>/`, as
`checkpoint-epoch1` and `checkpoint-final`.

Omit `TEACHER_CACHE` to run the teacher live; everything else is identical.

## 5. Evaluate

MMEB accuracy is only half the claim. The other half — that a gain comes with
better preservation of the teacher's retrieval structure — needs
`tools/eval_topology.py`, which reuses the embedding dumps the MMEB eval already
writes.

```bash
# (a) student accuracy; embeddings land in <ckpt>/mmeb_cls
bash scripts/eval/cls.sh training/CMTop/cmtop_h0_seed42/checkpoint-final

# (b) teacher embeddings on the same eval set — run once, reused by every variant
python tools/eval_mmeb.py \
    --model_name raghavlite/B3_Qwen2_2B --model_backbone qwen2_vl \
    --encode_output_path runs/teacher_emb \
    --lora --lora_r 8 --pooling eos --normalize True --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval --dataset_split test \
    --subset_name ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
    --per_device_eval_batch_size 16 --image_dir "${MMEB_EVAL_DIR:-./eval_images}" \
    --image_resolution 448 --tgt_prefix_mod

# (c) topology discrepancy + neighborhood preservation
python tools/eval_topology.py \
    --teacher_embeddings runs/teacher_emb \
    --student_embeddings training/CMTop/cmtop_h0_seed42/checkpoint-final/mmeb_cls \
    --subsets ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
    --batch_size 16 --output runs/topo_cmtop_h0_seed42.json
```

Subset names differ between the two splits — `ImageNet_1K` with an underscore in
MMEB-train, `ImageNet-1K` with a hyphen in MMEB-eval. That is MMEB's naming, not
a typo.

What to read in the report: `cross_modal_h0` is the quantity the method targets;
`query_cloud_h0` / `candidate_cloud_h0` are the ordinary point-cloud references
that the "is relation topology better than point-cloud topology?" question needs;
`recall@k` and `spearman` are measured against the *teacher*, so they are
fidelity, not accuracy.

---

## The six variants

Every row of the experiment plan is the same criterion under different flags, so
nothing else varies between them.

| `VARIANT` | brief's row | KD-side flags |
| --- | --- | --- |
| `student_only` | student only | `--kd_weight 0 --cmtop_weight 0 --cmtop_endpoint_kd none` |
| `endpoint` | standard endpoint KD | `--cmtop_weight 0` |
| `vsp` | pairwise / VSP-style geometry KD | `--cmtop_weight 0 --cmtop_geometry_weight 1` |
| `pointcloud_h0` | ordinary point-cloud H0 KD | `--cmtop_mode point_cloud` |
| `cmtop_h0` | **cross-modal relation H0 (the proposal)** | `--cmtop_mode cross_modal` |
| `cmtop_h0_h1` | + lightweight H1-birth | `--cmtop_mode cross_modal --cmtop_h1_weight 0.1` |

`student_only` is the one variant that does not declare a projector: it never
uses the endpoint term, and registering an unused module makes DDP abort on
multi-GPU.

## Environment variables

`scripts/train/cmtop/fastvlm_cls.sh`:

| variable | default | meaning |
| --- | --- | --- |
| `VARIANT` | `cmtop_h0` | which ablation row to run |
| `SEED` | `42` | random seed; also part of the output directory |
| `TEACHER_CACHE` | *(unset)* | path to a precomputed cache; unset runs the teacher live |
| `MMEB_TRAIN_DIR` | `./vlm2vec_train/MMEB-train` | where MMEB-train images live |
| `OUTPUT_DIR` | `training/CMTop/<variant>_seed<seed>` | checkpoint directory |
| `CMTOP_WEIGHT` | `1.0` | λ_CMTop |
| `KD_WEIGHT` | `0.3` | weight on the endpoint KD term |

`scripts/data/precompute_teacher_embeddings.sh`: `TEACHER_CACHE` (required),
`MMEB_TRAIN_DIR`, `NUM_GPUS_PER_NODE`, `BATCH_SIZE` (default 32),
`NUM_WORKERS` (default 4).

Both scripts forward extra arguments to the underlying Python entrypoint.

## Things worth knowing before you tune

**Batch size is the lever.** The filtration is built on the batch, so
`--per_device_train_batch_size` × world size decides how much of the retrieval
relation the loss can see at all. This is the parameter most likely to decide
whether CMTop beats endpoint KD. With the teacher cache in use its weights are
no longer on the device, so there is room to raise it.

Raising it makes `python tools/check_paper_settings.py` report a mismatch
against Table 6. That is expected — CMTop is a research line, not a cell of the
paper's main table — and the ablation stays internally valid as long as all six
variants share the same batch size.

**`--cmtop_weight` is the first knob to turn.** The topological term is a squared
difference of cosine distances, so it is naturally small next to the contrastive
loss.

**Verify the cache once.** Run a few steps with and without `TEACHER_CACHE` at
the same seed; the losses should agree to float16 precision. A larger gap means
the cache is wrong, not that the run is noisy.
