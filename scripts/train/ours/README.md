# Ours — runbook

Ours distils the identity-preserving merge hierarchy of the query-candidate
retrieval relation. “Identity” here means teacher/student row correspondence,
not an extra dataset class-label input. Read the [research brief](../../../docs/cross_modal_topological_distillation.md)
and [implementation notes](../../../docs/ours_implementation.md) before changing
the objective or batch semantics.

## 1. Verify the local implementation

```bash
python tools/misc/test_ours.py
python tools/misc/test_training_stack.py
python tools/misc/test_teacher_cache.py
```

The first command checks the minimax-connectivity definition, witness gradients,
the permutation counterexample, task validation and candidate dedup. It needs no
model, dataset or GPU.

## 2. Prepare data and the teacher cache

```bash
python scripts/data/download_mmeb.py --for scripts/train/ours/fastvlm_cls.sh
python scripts/data/download_mmeb.py --eval

TASK=cls STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls \
  bash scripts/data/precompute_teacher_embeddings.sh
```

Set `MMEB_TRAIN_DIR` and `MMEB_EVAL_DIR` if the images are elsewhere. The cache
is optional but recommended: Ours reads only the frozen teacher's final
embedding, so every row and seed can reuse it without loading the teacher.
The cache fingerprint includes subsets, their order, `percent_data` and image
settings; incompatible caches are refused.

## 3. Smoke train

```bash
TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls VARIANT=ours \
  bash scripts/train/ours/fastvlm_cls.sh \
  --percent_data 0.01 --push_to_hub False --eval_after_train False
```

Watch `topo_loss`, `candidate_unique_fraction` and the total `loss`. They must
stay finite; `topo_loss` should be non-zero for a non-identical student.

Task-homogeneous sampling requires at least one full global batch inside a
task. A tiny smoke percentage may violate that; lower
`--per_device_train_batch_size` for the smoke run, not for the final experiment.

## 4. Main grid

```bash
for v in ours student_only; do
  for s in 42 43 44; do
    TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls VARIANT=$v SEED=$s \
      bash scripts/train/ours/fastvlm_cls.sh
  done
done
```

The same interface is available for:

- `fastvlm_vqa.sh`
- `llava_onevision_cls.sh`
- `llava_onevision_vqa.sh`
- `fastvlm_ret.sh`, `llava_onevision_ret.sh` (retrieval; about 3.2× the data of CLS)
- `fastvlm_grounding.sh`, `llava_onevision_grounding.sh` (visual grounding)

Retrieval and grounding use the same configuration; their subsets and eval groups are in [docs/datasets.md](../../../docs/datasets.md). Build their caches with `TASK=ret` or `TASK=grounding`.

| `VARIANT` | purpose |
| --- | --- |
| `ours` | **correspondence-aware merge-hierarchy method** |
| `student_only` | retrieval objective only, everything else identical |
| `topo_only` | L_topo only (`--ours_retrieval_loss False`), λ unchanged |

`student_only` is the same criterion with `--ours_weight 0`, so it keeps the
task-homogeneous sampler, the candidate deduplication and the batch
construction. The loss is the only thing that changes between the two rows.

Ours needs no teacher-to-student projector at any width: it compares
distances, not embeddings. Declaring one would leave unused DDP parameters.

Comparing against a different `--kd_loss_type` is *not* like-for-like as the
repo stands. `TaskHomogeneousSampler` is enabled for `ours` only, so such a run
also changes the in-batch negatives. Enable the sampler for the other method too,
or report the difference.

## 5. Structural ablations

Run these after the main grid:

```bash
# Full U versus only its query-candidate block
VARIANT=ours bash scripts/train/ours/fastvlm_cls.sh --ours_merge_block cross

# Verify that canonical candidate vertices matter
VARIANT=ours bash scripts/train/ours/fastvlm_cls.sh \
  --ours_deduplicate_candidates False
```

The main method keeps `--ours_merge_block all`, candidate deduplication and
task-homogeneous batching enabled. It has one tunable loss coefficient:
`OURS_WEIGHT`, passed as `--ours_weight`.

## 6. Loss-term sensitivity

The objective has two terms, `L = L_ret + λ_topo · L_topo`. With λ held at
`OURS_WEIGHT` (1.0), the sweep trains one row per term and one with both:

| `VARIANT` | loss |
| --- | --- |
| `student_only` | `L_ret` |
| `topo_only` | `λ_topo · L_topo` |
| `ours` | `L_ret + λ_topo · L_topo` |

```bash
CELLS="fastvlm_cls" SEEDS="42 43 44" NUM_GPUS_PER_NODE=8 \
  bash scripts/train/ours/sensitivity/loss.sh
```

The runs go into the same `training/ours/<cell>/<variant>_seed<seed>` directories
as the main grid. A run whose `checkpoint-final` already exists is skipped, so
`student_only` and `ours` rows from the main grid are reused, not retrained. At
the end the script prints the MMEB averages per cell and variant (mean ± std
over seeds) from each run's `mmeb_eval/summary.json`. `DRY_RUN=1` prints the
plan without launching anything. The λ sweep is not part of this script.

## 7. Evaluate

```bash
# Task accuracy and student embedding dumps
bash scripts/eval/cls.sh training/ours/fastvlm_cls/ours_seed42/checkpoint-final

# Teacher embedding dumps; run once for the same subsets
python tools/eval_mmeb.py \
  --model_name raghavlite/B3_Qwen2_2B --model_backbone qwen2_vl \
  --encode_output_path runs/teacher_emb \
  --lora --lora_r 8 --pooling eos --normalize True --bf16 \
  --dataset_name TIGER-Lab/MMEB-eval --dataset_split test \
  --subset_name ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
  --per_device_eval_batch_size 16 --image_dir "${MMEB_EVAL_DIR:-./eval_images}" \
  --image_resolution 448 --tgt_prefix_mod

# Correspondence-aware structure and neighborhood fidelity
python tools/eval_topology.py \
  --teacher_embeddings runs/teacher_emb \
  --student_embeddings training/ours/fastvlm_cls/ours_seed42/checkpoint-final/mmeb_cls \
  --subsets ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
  --batch_size 128 --output runs/ours_seed42_structure.json
```

Read `cross_modal_merge_l1` as the primary structural metric. `recall@k` and
`spearman` are fidelity to the teacher; MST-edge recall and component ARI test
identity-aligned connectivity without reusing the training scalar. MMEB accuracy
is performance against ground truth; report both.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `VARIANT` | `ours` | `ours`, `student_only` or `topo_only` |
| `SEED` | `42` | run seed |
| `TEACHER_CACHE` | unset | compatible precomputed teacher embeddings |
| `MMEB_TRAIN_DIR` | `./vlm2vec_train/MMEB-train` | training images |
| `NUM_GPUS_PER_NODE` | `1` | DDP world size on one node |
| `BATCH_SIZE` | `16` FastVLM / `8` OneVision | per-device micro-batch size |
| `OUTPUT_DIR` | launcher-specific | checkpoint directory |
| `OURS_WEIGHT` | `1.0` | `lambda_merge`, the only loss coefficient |

The effective relation batch is local batch size times world size. Use 128–256
when memory permits and keep it identical across rows. Because the graph is
the micro-batch, gradient accumulation does not enlarge its topology.
