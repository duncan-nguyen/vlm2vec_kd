# CM-Merge — runbook

CM-Merge distils the labelled merge hierarchy of the query-candidate retrieval
relation. Read the [research brief](../../../docs/cross_modal_topological_distillation.md)
and [implementation notes](../../../docs/cmtop_implementation.md) before changing
the objective or batch semantics.

## 1. Verify the local implementation

```bash
python tools/misc/test_cmtop.py
python tools/misc/test_training_stack.py
python tools/misc/test_teacher_cache.py
```

The first command checks the minimax-connectivity definition, witness gradients,
the permutation counterexample, task validation and candidate dedup. It needs no
model, dataset or GPU.

## 2. Prepare data and the teacher cache

```bash
python scripts/data/download_mmeb.py --for scripts/train/cmtop/fastvlm_cls.sh
python scripts/data/download_mmeb.py --eval

TASK=cls STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls \
  bash scripts/data/precompute_teacher_embeddings.sh
```

Set `MMEB_TRAIN_DIR` and `MMEB_EVAL_DIR` if the images are elsewhere. The cache
is optional but recommended: CM-Merge reads only the frozen teacher's final
embedding, so every variant and seed can reuse it without loading the teacher.
The cache fingerprint includes subsets, their order, `percent_data` and image
settings; incompatible caches are refused.

## 3. Smoke train

```bash
TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls VARIANT=cmmerge \
  bash scripts/train/cmtop/fastvlm_cls.sh \
  --percent_data 0.01 --push_to_hub False --eval_after_train False
```

Watch `cmmerge_loss`, `candidate_unique_fraction` and the total `loss`. They must
stay finite; `cmmerge_loss` should be non-zero for a non-identical student.

Task-homogeneous sampling requires at least one full global batch inside a
task. A tiny smoke percentage may violate that; lower
`--per_device_train_batch_size` for the smoke run, not for the final experiment.

## 4. Main grid

```bash
for v in student_only endpoint vsp pointcloud_h0 barcode_h0 critical_edges cmmerge; do
  for s in 42 43 44; do
    TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls VARIANT=$v SEED=$s \
      bash scripts/train/cmtop/fastvlm_cls.sh
  done
done
```

The same interface is available for:

- `fastvlm_vqa.sh`
- `llava_onevision_cls.sh`
- `llava_onevision_vqa.sh`

| `VARIANT` | purpose |
| --- | --- |
| `student_only` | retrieval objective only |
| `endpoint` | conventional projected endpoint KD |
| `vsp` | dense query-candidate relation-matrix KD |
| `pointcloud_h0` | ordinary topology of each modality |
| `barcode_h0` | bipartite H0 barcode without node correspondence |
| `critical_edges` | correspondence-aware MST-edge control |
| `cmmerge` | **main labelled merge-hierarchy method** |
| `barcode_h0_h1` | optional legacy barcode control |

Only `endpoint` creates the teacher-to-student projector. The other variants do
not use endpoint KD, so declaring it would leave unused DDP parameters.

The launchers still accept `cmtop_h0` and `cmtop_h0_h1` as aliases for the two
legacy barcode variants. Do not report `cmtop_h0` as the main method.

## 5. Structural ablations

Run these after the main seven-row grid:

```bash
# Full U versus only its query-candidate block
VARIANT=cmmerge bash scripts/train/cmtop/fastvlm_cls.sh --cmtop_merge_block cross

# Verify that canonical candidate vertices matter
VARIANT=cmmerge bash scripts/train/cmtop/fastvlm_cls.sh \
  --cmtop_deduplicate_candidates False
```

The main method keeps `--cmtop_merge_block all`, candidate deduplication and
task-homogeneous batching enabled. It has one tunable loss coefficient:
`CMTOP_WEIGHT`, passed as `--cmtop_weight`.

## 6. Evaluate

```bash
# Task accuracy and student embedding dumps
bash scripts/eval/cls.sh training/CMTop/cmmerge_seed42/checkpoint-final

# Teacher embedding dumps; run once for the same subsets
python tools/eval_mmeb.py \
  --model_name raghavlite/B3_Qwen2_2B --model_backbone qwen2_vl \
  --encode_output_path runs/teacher_emb \
  --lora --lora_r 8 --pooling eos --normalize True --bf16 \
  --dataset_name TIGER-Lab/MMEB-eval --dataset_split test \
  --subset_name ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
  --per_device_eval_batch_size 16 --image_dir "${MMEB_EVAL_DIR:-./eval_images}" \
  --image_resolution 448 --tgt_prefix_mod

# Labelled structure and neighborhood fidelity
python tools/eval_topology.py \
  --teacher_embeddings runs/teacher_emb \
  --student_embeddings training/CMTop/cmmerge_seed42/checkpoint-final/mmeb_cls \
  --subsets ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
  --batch_size 128 --output runs/cmmerge_seed42_structure.json
```

Read `cross_modal_merge_l1` as the primary structural metric. The H0/H1
diagram distances are permutation-blind controls. `recall@k` and `spearman` are
fidelity to the teacher; MST-edge recall and component ARI test labelled
connectivity without reusing the training scalar. MMEB accuracy is performance
against ground truth; report both.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `VARIANT` | `cmmerge` | ablation row |
| `SEED` | `42` | run seed |
| `TEACHER_CACHE` | unset | compatible precomputed teacher embeddings |
| `MMEB_TRAIN_DIR` | `./vlm2vec_train/MMEB-train` | training images |
| `NUM_GPUS_PER_NODE` | `1` | DDP world size on one node |
| `BATCH_SIZE` | `16` FastVLM / `8` OneVision | per-device micro-batch size |
| `OUTPUT_DIR` | launcher-specific | checkpoint directory |
| `CMTOP_WEIGHT` | `1.0` | main `lambda_merge` |
| `KD_WEIGHT` | `0.3` | endpoint-only baseline weight |

The effective relation batch is local batch size times world size. Use 128–256
when memory permits and keep it identical across variants. Because the graph is
the micro-batch, gradient accumulation does not enlarge its topology.
