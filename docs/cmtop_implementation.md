# CMTop: implementation notes

Implementation of [cross_modal_topological_distillation.md](cross_modal_topological_distillation.md).
This file covers what the code does, why the topology is cheap enough to sit
inside a training step, and how to run the ablation and the evaluation.

| file | role |
| --- | --- |
| [src/topology.py](../src/topology.py) | persistence primitives: bipartite/point-cloud MST, H0 barcode, H1 births, Wasserstein |
| [src/criterions/cross_modal_topology.py](../src/criterions/cross_modal_topology.py) | the criterion, registered as `--kd_loss_type cmtop` |
| [src/evaluation/topology_metrics.py](../src/evaluation/topology_metrics.py) | topology discrepancy and neighborhood preservation |
| [tools/eval_topology.py](../tools/eval_topology.py) | CLI over the embedding dumps `tools/eval_mmeb.py` already writes |
| [src/teacher_cache.py](../src/teacher_cache.py) | memmapped store of the frozen teacher's embeddings |
| [tools/precompute_teacher_embeddings.py](../tools/precompute_teacher_embeddings.py) | builds that store; sharded under `torchrun` |
| [tools/misc/test_cmtop.py](../tools/misc/test_cmtop.py) | self-checks; no model download, no GPU |
| [tools/misc/test_teacher_cache.py](../tools/misc/test_teacher_cache.py) | cache self-checks, mostly about what it refuses |
| [scripts/train/cmtop/fastvlm_cls.sh](../scripts/train/cmtop/fastvlm_cls.sh) | the six-variant ablation launcher |

## The two shortcuts

The filtration is the bipartite graph `G_eps = (Q u C, {(q_i, c_j) : D_ij <= eps})`
with `D_ij = 1 - cos(q_i, c_j)`. Nothing here is approximated:

* **H0 is a minimum spanning tree.** Components of `G_eps` merge exactly at the
  MST edge weights, and every vertex is born at `eps = 0`, so the finite barcode
  *is* the sorted vector of the `|Q| + |C| - 1` MST weights. One `scipy` MST call
  per side, no filtration sweep.
* **H1 is the complement of that tree.** A bipartite graph has no triangles, so
  the flag complex has no 2-simplices and no 1-cycle is ever filled in. Each of
  the `E - V + 1` non-tree edges opens one independent cycle when it enters, and
  that cycle never dies. "H1-birth only" is therefore not a cheap stand-in for
  H1 on this filtration -- it is all of H1, and the brief's `lambda_1` term is
  exact rather than heuristic. (This is the one place where the bipartite
  framing buys something the point-cloud framing cannot.)

`tools/misc/test_cmtop.py` re-derives both from a union-find sweep over the
sorted edges and compares against the fast path, along with the exact diagram
Wasserstein against brute-force enumeration of all partial matchings.

Gradients reach the *coordinates* of the diagram -- the distances sitting on the
selected edges -- while the combinatorics (which edges the MST picked) are held
constant. That is exact, not a relaxation: the MST edge set is piecewise
constant in the embeddings, so it has zero gradient almost everywhere.

## Running the ablation

Every row of section 4 of the brief is the same criterion under different flags:

| brief's row | `VARIANT` | flags |
| --- | --- | --- |
| student only | `student_only` | `--kd_weight 0 --cmtop_weight 0 --cmtop_endpoint_kd none` |
| standard endpoint KD | `endpoint` | `--cmtop_weight 0` |
| pairwise / VSP-style geometry KD | `vsp` | `--cmtop_weight 0 --cmtop_geometry_weight 1` |
| ordinary point-cloud H0 KD | `pointcloud_h0` | `--cmtop_mode point_cloud` |
| cross-modal relation H0 (main) | `cmtop_h0` | `--cmtop_mode cross_modal` |
| + lightweight H1-birth | `cmtop_h0_h1` | `--cmtop_mode cross_modal --cmtop_h1_weight 0.1` |

```bash
for v in student_only endpoint vsp pointcloud_h0 cmtop_h0 cmtop_h0_h1; do
  for s in 42 43 44; do VARIANT=$v SEED=$s bash scripts/train/cmtop/fastvlm_cls.sh; done
done
```

Full flag list: `cmtop_*` in [src/arguments.py](../src/arguments.py). The ones
worth turning first are `--cmtop_weight` (lambda_CMTop), `--cmtop_h1_weight`
(lambda_1) and `--cmtop_endpoint_kd` (`cosine`/`mse`/`none`, scaled by
`--kd_weight`). `--cmtop_mode` also accepts `union` (both modalities as a single
cloud) and `+`-joined combinations such as `cross_modal+point_cloud`.

## Evaluating

MMEB accuracy comes from `tools/eval_mmeb.py` as usual. The structural half of
the claim -- "the gain is accompanied by better preservation of the teacher
retrieval structure" -- comes from `tools/eval_topology.py`, which reuses the
embedding dumps that eval already wrote:

```bash
python tools/eval_topology.py \
    --teacher_embeddings runs/teacher/emb \
    --student_embeddings runs/student/emb \
    --subsets ImageNet-1K N24News \
    --batch_size 16 \
    --output runs/student/topology_report.json
```

It reports, per subset:

* `topology_discrepancy`: exact `W_2^2` between the teacher's and the student's
  diagrams, for the cross-modal relation (`cross_modal_h0`, `cross_modal_h1_birth`)
  *and* for the ordinary point clouds (`query_cloud_h0`, `candidate_cloud_h0`,
  `union_cloud_h0`). Averaged over random batches of `--batch_size`, because a
  batch-sized filtration is the only one the loss ever shapes.
* `neighborhood_preservation`: `recall@k` overlap between the teacher's and the
  student's top-k, plus mean Spearman correlation of the full score vectors.
  Measured against the *teacher*, so it is fidelity, not accuracy.

Two deliberate differences from the training loss, so the evaluation is not
grading the loss with its own ruler:

* it uses the exact diagram distance (diagonal matches included,
  `wasserstein2_diagram`) rather than the bijective sorted matching the loss
  optimises;
* it reports the point-cloud discrepancies alongside the relational one, which
  is what the "is relation topology better than ordinary point-cloud topology?"
  question needs. On synthetic data the gap is stark: two independently sampled
  Gaussian clouds have near-identical *separate* H0 barcodes while their
  cross-modal relation is completely different -- exactly the blind spot the
  method claims to close.

## Cost

The topological term is negligible next to a VLM forward/backward. Measured
per step on CPU, forward + backward of the loss term alone:

| batch | H0 | H0 + H1-birth |
| --- | --- | --- |
| 16 | 0.4 ms | 0.6 ms |
| 64 | 1.1 ms | 1.3 ms |
| 128 | 2.3 ms | 2.8 ms |
| 256 | 10.0 ms | 11.4 ms |

The MST runs on CPU via scipy on a detached copy of the distance matrix, so it
costs one `(B, B)` device-to-host transfer per side per step. The dense
`(V, V)` scipy path is what makes B=256 jump to 10 ms; if the gathered batch
ever goes past a few hundred, replace it with a sparse or on-device MST.

**The loss is not where the time goes.** Per step the criterion runs four
encoder passes -- teacher query, teacher target, student query, student target.
With a 2B-parameter teacher against a 0.5B student, and backward counted at
twice forward, the frozen teacher is roughly 60% of the step's FLOPs.

### Precomputing the teacher (`--teacher_embedding_cache`)

The teacher is frozen and the dataset applies no augmentation, so its embedding
for a sample is a pure function of the dataset index, and CMTop reads nothing
else. Compute them once:

```bash
TEACHER_CACHE=cache/b3_qwen2_2b_cls bash scripts/data/precompute_teacher_embeddings.sh
TEACHER_CACHE=cache/b3_qwen2_2b_cls VARIANT=cmtop_h0 bash scripts/train/cmtop/fastvlm_cls.sh
```

Every variant and seed then trains with **no teacher model in the process at
all**: no teacher forward, no teacher image resize or tokenisation in the
dataloader, and no teacher weights on the device. For the 6-variant x 3-seed
plan that is one pass over the data instead of eighteen.

`--teacher_embedding_cache` is accepted by any criterion in
`TEACHER_EMBEDDING_ONLY_CRITERIONS` (`cmtop`, `contrastive_rkd`,
`universal_logit`); anything that reads hidden states or attentions is refused
with an explanation rather than served wrong data. All three now obtain teacher
embeddings through `Distiller.encode_teacher`, which is the one place that knows
whether a cache is in use.

The store is a `(N, 2, dim)` float16 memmap plus a `meta.json` fingerprint of
everything that determines what the teacher produces -- model, checkpoint,
pooling, normalisation, LoRA rank, dataset, split, subset list *and its order*,
`percent_data`, image dir, resolution, `max_len`. Loading refuses a cache whose
fingerprint does not match the run, because a stale cache is otherwise silently
wrong rather than loudly broken. An interrupted build is never marked usable,
and the merge step requires every dataset index exactly once.

float16 rather than bfloat16 is deliberate: the embeddings are produced in bf16
and are close to unit norm, so fp16 stores them with more mantissa than they
were computed with, at the same file size.

### What did *not* work: skipping the full-vocabulary lm_head and the hidden states

Two optimisations that look obvious from the outside do not apply here, and the
reason is the same in both cases: `src/model/vlm_backbone/` **vendors** its own
copies of the backbone classes, and they differ from the HuggingFace originals.

* **`logits_to_keep` for the qwen teachers.** HF's `Qwen2VLForConditionalGeneration`
  accepts it, so passing it would avoid running a 1536x151936 lm_head over the
  whole sequence. The vendored class does not take the argument -- its forward
  ends in `*args, **kwargs`, which swallows it silently -- and it has its
  `logits = self.lm_head(hidden_states)` line commented out anyway, so there was
  never a lm_head cost on the teacher to remove. The vendored `qwen2_5_vl` *does*
  compute full logits, but the kwarg cannot reach it either; recovering that
  would mean editing the vendored file, which is a decision for whoever owns it.
  `_LOGITS_TO_KEEP_BACKBONES` therefore stays at the two backbones whose vendored
  forwards genuinely take the argument.
* **Reading `last_hidden_state` off the base module** to avoid
  `output_hidden_states=True`. For HF's Qwen2-VL this is exact:
  `model.model(...).last_hidden_state` is bit-identical to
  `model(..., output_hidden_states=True).hidden_states[-1]`. For the vendored
  class it is not even close. The vendored `Qwen2VLForConditionalGeneration`
  performs the visual merge itself -- it takes `pixel_values` as a *list*,
  concatenates it, runs the vision tower and scatters the result into
  `inputs_embeds` -- before calling `self.model`, which is a text-only decoder.
  Worse, that decoder's forward also ends in `*args, **kwargs`, so handing it
  `pixel_values` would raise nothing and simply return **text-only embeddings
  with the images dropped**. A try/except fallback does not help against a
  failure that does not raise. The path was removed rather than gated.

The lesson worth keeping: in this repo a backbone name identifies a vendored
class, so anything reasoned from the HuggingFace implementation of the same name
has to be checked against `src/model/vlm_backbone/` before it is trusted.

## Gradient scaling under DDP

The student reps are gathered with `dist_utils.dist_gather` (autograd-aware),
so each rank contributes the gradient of its own slice and DDP then *averages*
across ranks -- leaving the gradient a factor of `world_size` smaller than the
gathered loss implies. `src/loss.py` compensates for this with `scale_loss`;
`em_kd` and this criterion do not. It is a uniform rescale of the whole
objective, equivalent to dividing the learning rate by the number of GPUs, and
it is consistent with the other criteria here, so it is left alone -- but it
matters if you compare a multi-GPU run against a single-GPU one at the same LR.

## Choices worth revisiting

* **Sorted vs. exact matching in the loss.** Training uses the bijective
  sorted-to-sorted matching, which is the exact 1-D optimal transport on the
  death coordinate but ignores the option of matching a bar to the diagonal.
  With equal cardinalities and comparable scales this is the optimum; it can
  differ when one side has a bar that is much shorter than its counterpart.
  `wasserstein2_diagram` is the exact version if this turns out to matter.
* **The batch is the filtration.** Topology is computed on the (DDP-gathered)
  batch, so `--per_device_train_batch_size` times the world size sets how much of
  the retrieval relation the loss can see. This is the parameter most likely to
  decide whether the method beats endpoint KD.
* **Duplicate nodes.** Classification subsets point many queries at the same
  label embedding; duplicated candidates sit at distance 0 and add zero-length
  bars that say more about the sampling than about the model. `eval_topology.py`
  de-duplicates before measuring; training does not, so a batch drawn from a
  single classification subset will have a partly degenerate barcode.
* **`mse` endpoint KD against a normalised student.** With `--normalize True`
  the student embedding is unit-norm while the projected teacher is not, so
  `--cmtop_endpoint_kd mse` asks the student to match a length it cannot have.
  `cosine` (the default) is the right choice under normalisation.
* **`--cmtop_reduction`** defaults to `mean` (per bar) so that lambda does not have
  to be retuned when the batch size changes; `sum` is the textbook `W_2^2`.
