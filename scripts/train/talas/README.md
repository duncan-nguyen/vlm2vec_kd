# TALAS — how to run it

Teacher-Anchored Layer Alignment with Adaptive Sharpness-aware minimization:
anchor the student's top `K` layers to the teacher's final embedding, propagate
that geometry down the student itself, and optimise the result with ASAM.

* the paper: [docs/baseline methods/TALAS.pdf](../../../docs/baseline%20methods/TALAS.pdf)
* the design, the maths, and every place this differs from the paper:
  [docs/talas_implementation.md](../../../docs/talas_implementation.md)

This file is the operational half.

---

## 1. Smoke test — do not skip this

No GPU, no model download, no dataset:

```bash
python tools/misc/test_talas.py           # the criterion, ASAM, the two-pass loop
python tools/misc/test_teacher_cache.py   # cache format + registry consistency
```

Then a real but tiny run — anything after the script name is forwarded to the
trainer:

```bash
bash scripts/train/talas/fastvlm_cls.sh --percent_data 0.01
```

## 2. Build the teacher cache first

TALAS reads nothing from the teacher but its final pooled embedding — appendix C
of the paper runs the teacher exactly once, offline, for precisely this reason.
So there is no reason to run it live:

```bash
TASK=cls STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls \
  bash scripts/data/precompute_teacher_embeddings.sh

TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls bash scripts/train/talas/fastvlm_cls.sh
```

One cache per `(student, task)` cell, shared with `rkd`, `uld` and `cmtop`.
Leave `TEACHER_CACHE` unset to run the teacher live.

## 3. The variants

`VARIANT` selects one row of the paper's Table 2 / Figure 2 ablation:

| `VARIANT` | what it runs |
| --- | --- |
| `talas` *(default)* | the full method, ASAM |
| `no_asam` | the same three losses under plain AdamW |
| `sam` | non-adaptive SAM at `rho 0.05` |
| `no_lasd` | `--talas_lasd_weight 0` |
| `no_tamd` | `--talas_tamd_weight 0` — the paper's catastrophic row |

```bash
VARIANT=talas SEED=42 bash scripts/train/talas/fastvlm_cls.sh

for v in talas no_asam sam no_lasd no_tamd; do
  for s in 42 43 44; do
    VARIANT=$v SEED=$s bash scripts/train/talas/fastvlm_cls.sh
  done
done
```

The two layer-depth ablations are single flags:

```bash
for k in 1 2 3 4 6; do TAMD_LAYERS=$k bash scripts/train/talas/fastvlm_cls.sh; done
bash scripts/train/talas/fastvlm_cls.sh --talas_num_lasd_layers 6
```

## 4. Environment variables

On top of the shared `MMEB_TRAIN_DIR` / `OUTPUT_DIR` / `SEED` /
`NUM_GPUS_PER_NODE` / `TEACHER_CACHE`:

| variable | default | meaning |
| --- | --- | --- |
| `VARIANT` | `talas` | the ablation row, above |
| `TALAS_CONTRASTIVE_WEIGHT` | `1.0` | `λ₁`. The paper uses 0.001 — see below |
| `TAMD_WEIGHT` | `0.75` | `λ₂`, on `L_TAMD` |
| `LASD_WEIGHT` | `1.0` | `λ₃`, on `L_LASD` |
| `TAMD_LAYERS` | `2` | `K`, the number of teacher-anchored top layers |
| `SAM_RHO` | `0.5` | ASAM's radius, in units of `|w|` |

## 5. Two things that will surprise you

**`λ₁` is 1.0 here, not the paper's 0.001.** The paper's contrastive term is an
*unsupervised* SimCSE loss over two dropout views, a regulariser it deliberately
keeps small. This repo's contrastive term is the supervised query/positive loss —
the primary task objective, at weight 1 for every other method in
`scripts/train/`. Making TALAS the one baseline that trains at 0.001× the task
loss would not be a fair comparison. For the paper's balance:

```bash
TALAS_CONTRASTIVE_WEIGHT=0.001 bash scripts/train/talas/fastvlm_cls.sh
```

**ASAM doubles the step time.** Every optimizer step runs its whole accumulation
window twice: once to pick the perturbation, once at the perturbed weights to
get the gradient that is applied. The paper measures the same (Table 3: 195
ms/step without, 375 with). `VARIANT=no_asam` is the cheap configuration.

## 6. What it logs

`talas_tamd_loss` and `talas_lasd_loss` alongside `contrastive_loss`, `kd_loss`
and `loss`. If `talas_tamd_loss` sits near 1.0 and does not move, the `W_l`
projections are not learning — check that `--projector_lr` is reaching them
(the startup line `Criterion parameters added to optimizer at lr ...`) and that
no `--projector_config_path` was passed.
