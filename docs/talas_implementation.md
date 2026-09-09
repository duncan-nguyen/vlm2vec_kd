# TALAS in this repo

`--kd_loss_type talas`, plus `--sharpness_aware asam`.

The paper: [docs/baseline methods/TALAS.pdf](baseline%20methods/TALAS.pdf) —
*TALAS: Teacher-Anchored Layer Alignment with Adaptive Sharpness-Aware
Minimization for Embedding Distillation* (anonymous ACL submission).

The code:

| piece | where |
| --- | --- |
| `L_TAMD` + `L_LASD` | [src/criterions/talas.py](../src/criterions/talas.py) |
| ASAM / SAM | [src/training/sam.py](../src/training/sam.py) |
| flags | `TrainingArguments` in [src/arguments.py](../src/arguments.py) |
| launchers | [scripts/train/talas/](../scripts/train/talas/) |
| self-checks | `python tools/misc/test_talas.py` |

---

## 1. What the paper proposes

Three terms and an optimizer.

**`L_TAMD`** (eq. 3) — *teacher-anchored multi-layer distillation*. The teacher's
final embedding `e^T` is a **stationary anchor** for the student's top `K = k+1`
layers, each through its own learnable projection `W_l ∈ R^{d_S × d_T}`:

```
L_TAMD = 1/K · Σ_{l = L-K+1..L}  ( 1 − cos( e_l^S W_l , e^T ) )
```

Anchoring *several* upper layers rather than only the last is the paper's answer
to the capacity gap: the student injects the teacher's semantics early and
consolidates them by the output. Anchoring too many is worse than anchoring one
— their Table 6 peaks at `K = 2` and falls away past `K = 4`.

**`L_LASD`** (eq. 5) — *layer-aligned self-distillation*. Lower layers are never
shown the teacher. Instead each layer's batch relation matrix
`R_l = Ê_l Ê_lᵀ` (cosine similarities of the L2-normalised embeddings) is pulled
towards the layer above it, which acts as a guide:

```
L_LASD = 1/(L−1) · Σ_l  ‖ R_{l+1} − R_l ‖_F²
```

so the teacher's geometry, anchored at the top, propagates downwards without any
shallow layer being forced onto an abstract teacher representation.

**`L_SimCSE`** (eq. 7) — an unsupervised contrastive regulariser, two dropout
views of the same sentence, there to stop the student collapsing onto the
teacher's anisotropy.

**ASAM** — the whole objective is optimised with adaptive sharpness-aware
minimization: two forward/backward passes per step, the first choosing a
perturbation `w + ε`, the second producing the gradient that is actually
applied. Their Figure 2 shows ASAM beating both plain SAM and DISAM, and their
Table 2 shows it matters most where the capacity gap is largest.

Total: `L = λ₁ L_SimCSE + λ₂ L_TAMD + λ₃ L_LASD`, with
`λ₁ = 0.001, λ₂ = 0.75, λ₃ = 1` (appendix E).

---

## 2. What changes here, and why

TALAS is a **text** method: BERT-family students, LLM embedding teachers, an
unpaired sentence corpus. This repo distils **multimodal** embedding models on
MMEB query/candidate pairs. Four things had to be decided.

### 2.1 `L_SimCSE` → the repo's in-batch contrastive loss

TALAS's corpus has no pairs, so it manufactures two views of each sentence with
dropout and contrasts those. This repo *has* pairs, and
`DistillCriterion.forward` already computes the in-batch softmax over them —
the same role (keep the space uniform, don't collapse onto the teacher), on a
supervision signal that actually exists here. Running a second dropout forward
pass on top would double the student's cost to re-derive a weaker version of a
loss the batch already supports.

`--talas_contrastive_weight` is that term's `λ₁`. **The default is 1.0, not the
paper's 0.001.** 0.001 is calibrated for the dropout objective on unpaired text;
here the contrastive term is the primary task loss and every other method in the
repo gives it weight 1, which is what makes the baselines comparable. Pass
`--talas_contrastive_weight 0.001` for the paper's balance.

### 2.2 `W_l` lives on the criterion, not in a projector config

Eq. 3 wants `K` projections, one per anchored layer, sized from
`--talas_num_tamd_layers`. The repo's shared projectors are configured by a JSON
file (`--projector_config_path`) and go the other way, teacher → student. Rather
than couple `K` to a config file, `TALASLoss` builds its own.

That needed one general addition to the framework:
`DistillCriterion.build_parameters(distiller)`, a no-op for every other method.
`src.training.entrypoint.attach_criterion` calls it after the `Distiller` exists
and registers the criterion on the distiller, so its weights reach the device,
are covered by DDP, get a parameter group at `--projector_lr`, and are written
to the checkpoint as `kd_criterion.pth`. Nothing in it branches on a method
name.

**TALAS launchers must not pass `--projector_config_path`** — a `t2s` projector
registered and never used makes DDP abort with "expected to have finished
reduction".

### 2.3 Two streams, not one

Each batch has a query side and a positive side. `L_TAMD` and `L_LASD` are
computed on each and averaged, which keeps their magnitude the same as a
one-stream run and keeps `R_l` a within-modality relation matrix. (Mixing query
and candidate rows into one matrix would make it a cross-modal claim, which is
CMTop's subject, not TALAS's.)

### 2.4 Per-layer pooling is layout-agnostic

`e_l^S` is obtained by running the student's own `_pooling` over
`hidden_states[l]` with the batch's own attention mask — the identical call the
model makes for its final embedding. So TALAS is exactly as correct as the
model's own pooling on **both** students, and needs none of the
`[vision][text][pad]` versus `[pad][vision][text]` handling that EM-KD and the
span criteria do. `hidden_states[0]` is the embedding layer's output and is
skipped; `e_1^S … e_L^S` are `hidden_states[1:]`.

The per-layer pooled stack is all-gathered across ranks as one tensor, so the KD
terms see the same widened batch as the contrastive term and `R_l` is built over
the global batch on multi-GPU.

### 2.5 Two judgment calls the paper leaves open

**The `L_LASD` guide is detached** (`--talas_lasd_detach_guide`, default true).
Eq. 5 is written symmetrically, but §3.2 describes propagating knowledge
"sequentially from top to bottom" with the upper layer as a "dynamic guide".
Without the detach the term is a symmetric smoothness penalty and the
teacher-anchored top gets dragged *down* toward the untrained bottom, fighting
`L_TAMD`. Pass `--talas_lasd_detach_guide false` for eq. 5 read literally.

**`L_LASD` sums** (`--talas_lasd_reduction`, default `sum`). Eq. 5 spells out
`‖A‖_F² = Σ_ij A_ij²`, which makes the term grow with `N²`. `mean` divides by
`N²` and is batch-size independent; use it if the term dominates at a batch size
the paper never ran.

### 2.6 Training configuration

The paper's own setup (appendix E, Table 4) is 5 epochs, lr 2e-5, batch 32. The
launchers here use **Table 6 of the HieRD paper** — 1 epoch, lr 1e-4, batch
16/8, the settings every other baseline in this repo uses and that
`tools/check_paper_settings.py` enforces. TALAS's numbers are meant to be
comparable to the other baselines here, not to the TALAS paper's text results.

---

## 3. ASAM

`src/training/sam.py`, behind `--sharpness_aware {none,sam,asam}`. Nothing about
it is TALAS-specific — any `--kd_loss_type` can be trained with it.

```
eps = rho · T_w² ∇L / ‖ T_w ∇L ‖₂
```

`T_w = 1` for SAM; `T_w = |w| + eta` for ASAM, applied (following the reference
implementation) only to parameters whose name contains `weight`, so biases and
1-d parameters keep a plain L2 ball. The radius means different things in the
two: ASAM measures it in units of `|w|` and wants 0.5–2.0, SAM wants ~0.05.

The training loop drives the two steps, because the second one has to replay the
whole gradient accumulation window at the perturbed weights and only the loop
knows what that window was. `DistillTrainer` therefore buffers the window's
micro-batches when — and only when — `--sharpness_aware` is on; at
`--gradient_accumulation_steps 1` that is the one batch already alive.

Three details that are easy to get wrong and were checked:

* **the weights are restored from a saved copy**, not by subtracting `ε` again.
  The trainable weights here are bf16, where `(w + ε) − ε` does not reliably
  return `w`, and the drift would accumulate over a run. The copy is the same
  size `ε` would have been.
* **the logged loss is the clean one**, at `w`. The second pass is evaluated at
  the perturbed weights and its loss is not what the run is minimising.
* **a step whose gradient is zero or non-finite skips the perturbation** and
  takes the ordinary descent step rather than dividing by it.

Cost: two forward/backward passes per optimizer step, i.e. roughly double the
step time. The paper says the same (its Limitations section, and Table 3: 195
ms/step without ASAM, 375 with).

---

## 4. Flags

| flag | default | paper | meaning |
| --- | --- | --- | --- |
| `--talas_contrastive_weight` | `1.0` | 0.001 | `λ₁`, on the contrastive term |
| `--talas_tamd_weight` | `0.75` | 0.75 | `λ₂`, on `L_TAMD` |
| `--talas_lasd_weight` | `1.0` | 1 | `λ₃`, on `L_LASD` |
| `--talas_num_tamd_layers` | `2` | 2 | `K`: top layers anchored to the teacher |
| `--talas_num_lasd_layers` | `0` (all) | all | adjacent pairs in `L_LASD`, from the top |
| `--talas_lasd_detach_guide` | `true` | — | top-down propagation vs. a symmetric penalty |
| `--talas_lasd_reduction` | `sum` | sum | `‖·‖_F²` vs. its per-entry mean |
| `--sharpness_aware` | `none` | asam | `none` / `sam` / `asam` |
| `--sam_rho` | `0.5` | — | perturbation radius |
| `--asam_eta` | `0.01` | 0.01 | the `|w| + η` floor |

`--kd_weight` is **not** read by TALAS: `λ₂` and `λ₃` weight the two KD terms
directly.

Logged every step: `talas_tamd_loss`, `talas_lasd_loss`, `kd_loss`,
`contrastive_loss`, `loss`.

---

## 5. The ablations

`VARIANT` on any launcher covers Table 2 and Figure 2:

| `VARIANT` | what it is |
| --- | --- |
| `talas` | the full method (ASAM) |
| `no_asam` | same losses, plain AdamW |
| `sam` | non-adaptive SAM at `rho 0.05` |
| `no_lasd` | `--talas_lasd_weight 0` |
| `no_tamd` | `--talas_tamd_weight 0` — the paper's catastrophic row |

The two layer-depth ablations (Tables 6 and 7) are single flags:

```bash
for k in 1 2 3 4 6; do TAMD_LAYERS=$k bash scripts/train/talas/fastvlm_cls.sh; done
for m in 2 4 6 12; do bash scripts/train/talas/fastvlm_cls.sh --talas_num_lasd_layers $m; done
```

## 6. Checking it

```bash
python tools/misc/test_talas.py          # the criterion, ASAM, the two-pass loop
python tools/misc/test_teacher_cache.py  # cache compatibility
python tools/check_paper_settings.py     # launcher drift
bash scripts/train/talas/fastvlm_cls.sh --percent_data 0.01   # end-to-end
```
