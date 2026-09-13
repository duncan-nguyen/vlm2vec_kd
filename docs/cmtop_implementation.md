# CM-Merge: implementation notes

This is the code-level companion to
[cross_modal_topological_distillation.md](cross_modal_topological_distillation.md).
The registered criterion name remains `--kd_loss_type cmtop` for checkpoint and
launcher compatibility; the default method is now CM-Merge.

## Code map

| file | role |
| --- | --- |
| [src/topology.py](../src/topology.py) | bipartite MST, identity-preserving merge witnesses/matrix |
| [src/criterions/cross_modal_topology.py](../src/criterions/cross_modal_topology.py) | the CM-Merge objective |
| [src/distiller.py](../src/distiller.py) | stable task/candidate identities and collation |
| [src/training/dataloader.py](../src/training/dataloader.py) | task-homogeneous global batching under DDP |
| [src/evaluation/topology_metrics.py](../src/evaluation/topology_metrics.py) | merge discrepancy and neighborhood fidelity |
| [tools/eval_topology.py](../tools/eval_topology.py) | evaluation CLI over existing embedding dumps |
| [tools/misc/test_cmtop.py](../tools/misc/test_cmtop.py) | definition-level maths, gradients and criterion checks |
| [scripts/train/cmtop/](../scripts/train/cmtop/) | matched launchers |

## Exact computation

For a query-candidate distance matrix `D`, define a complete weighted bipartite
graph and

```text
U[a,b] = min over paths a→b (maximum edge distance on the path).
```

`U[a,b]` is exactly the first filtration threshold where the two indexed
vertices share a connected component. It can be recovered from any MST: it is
the largest edge on the unique MST path between `a` and `b`.

`bipartite_merge_witnesses` computes all entries with a Kruskal pass. Whenever
an edge joins components A and B, that edge is the bottleneck witness for every
pair in `A × B`. `bipartite_merge_matrix` then gathers the corresponding values
from the original PyTorch distance matrix. The MST routing is discrete and
detached; gradients through its selected values are exact almost everywhere,
away from distance ties.

The main objective is only

```text
L = L_retrieval + cmtop_weight · mean_{a<b} |U_teacher[a,b] - U_student[a,b]|.
```

That is the whole objective: no endpoint term, no projector, no auxiliary loss.
`--cmtop_merge_block cross` restricts the comparison to query-candidate entries
for the paper's block ablation; `all` is the default.

## Why identities are part of the implementation

A sorted H0 barcode keeps merge heights but discards which nodes merged. A
candidate-column permutation can therefore give zero barcode loss while
changing retrieval at rank 1. CM-Merge compares the identity-aligned matrix `U`, so the
same permutation changes the loss.

Two data constraints make those identities unambiguous:

- `TaskHomogeneousSampler` forms every *global* DDP batch inside one dataset
  task. All ranks receive disjoint slices of the same task batch before the
  criterion gathers their embeddings.
- `DistillationDataset` emits a deterministic int64 candidate id from
  `(task, positive text, positive image path)`. The criterion retains all query
  rows but selects the first embedding for each unique candidate id. This
  preserves many-query-to-one relations without duplicate candidate vertices.

The criterion validates task ids and logs `num_unique_candidates` plus
`candidate_unique_fraction`. A missing task id, an id of `-1` or a mixed task
batch all fail loudly; pass `--cmtop_task_homogeneous False` to run on a
mixed-task relation deliberately.

## Runs

Two rows, one criterion. The control is the same criterion with the CM-Merge
term switched off, so it runs under exactly the same sampler, batch
construction and candidate deduplication -- the only difference is the loss.

| `VARIANT` | target | main flags |
| --- | --- | --- |
| `cmmerge` | **identity-preserving merge hierarchy** | `--cmtop_weight 1.0` |
| `student_only` | no teacher signal | `--cmtop_weight 0.0` |

```bash
for v in cmmerge student_only; do
  for s in 42 43 44; do
    VARIANT=$v SEED=$s bash scripts/train/cmtop/fastvlm_cls.sh
  done
done
```

Structural ablations of the method itself, after the main grid:

```bash
# identity-aligned query-candidate entries only
VARIANT=cmmerge bash scripts/train/cmtop/fastvlm_cls.sh --cmtop_merge_block cross

# demonstrate why canonical candidates matter
VARIANT=cmmerge bash scripts/train/cmtop/fastvlm_cls.sh --cmtop_deduplicate_candidates False
```

Comparing CM-Merge against a *different* `--kd_loss_type` is not an apples-to-apples
comparison as the repo stands: `TaskHomogeneousSampler` is enabled for `cmtop`
only (see [src/training/dataloader.py](../src/training/dataloader.py)), so such a
run differs in its in-batch negatives as well as in its loss. Either enable the
sampler for the other method too, or report the difference.

## Evaluation

`tools/eval_topology.py` aligns query identities across teacher/student and
constructs an independent unique candidate pool. Query and candidate counts may
differ. This is important for classification, where thousands of queries share
a small label vocabulary.

```bash
python tools/eval_topology.py \
  --teacher_embeddings runs/teacher/emb \
  --student_embeddings training/CMTop/cmmerge_seed42/checkpoint-final/mmeb_cls \
  --subsets ImageNet-1K N24News \
  --batch_size 128 \
  --output runs/cmmerge_seed42_structure.json
```

The primary structural metric is `cross_modal_merge_l1`. `recall@k` overlap and
Spearman correlation measure fidelity to the teacher over the full canonical
candidate pool; `teacher_mst_edge_recall` and component `ARI` at teacher
distance quantiles provide correspondence-aware diagnostics that are not the
optimized merge loss. Report these next to MMEB task accuracy, never as
replacements for it.

## Verification

```bash
python tools/misc/test_cmtop.py
python tools/misc/test_training_stack.py
python tools/misc/test_teacher_cache.py
```

The CM-Merge checks independently verify:

- the MST result equals Floyd-Warshall minimax connectivity;
- every indexed vertex pair has a valid bottleneck witness;
- symmetry, zero diagonal and the L-infinity stability bound;
- finite non-zero gradients through witness distances;
- equivariance to a simultaneous permutation of teacher/student identities;
- the candidate-permutation counterexample separating CM-Merge from a barcode;
- candidate deduplication and mixed-task / missing-metadata rejection;
- disjoint, task-consistent DDP sampler slices.

## Operational notes

The graph is built from the DDP-gathered micro-batch, so the effective topology
batch is `per_device_train_batch_size × world_size`; target 128–256 when memory
permits. Task ranges smaller than one full global batch are dropped, and the
sampler rejects a dataset where no task can form one.

MST selection runs on CPU through SciPy and returns an `O((|Q|+|C|)^2)` merge
matrix. This is modest at the intended batch size but should be profiled before
going substantially above 256 indexed vertices per side.

CM-Merge reads only final teacher embeddings and therefore supports
`--teacher_embedding_cache`. Precompute once and reuse the same cache across
seeds:

```bash
TASK=cls STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls \
  bash scripts/data/precompute_teacher_embeddings.sh
TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls VARIANT=cmmerge \
  bash scripts/train/cmtop/fastvlm_cls.sh
```

CM-Merge compares distances, never embeddings, so it needs no teacher-to-student
projector at any teacher/student width. Declaring one would create unused
trainable parameters and can break DDP.

Gradients move the weights of the edges currently in the student's MST; nothing
pushes a *different* edge into that tree. The loss therefore has a floor set by
the tree structure the student starts from -- in the self-check it converges to
~35% of its initial value with 5 of 15 MST edges shared with the teacher. Read
`teacher_mst_edge_recall` alongside the loss curve rather than expecting it to
reach zero.
