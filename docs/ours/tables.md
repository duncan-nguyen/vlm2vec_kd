# Experiment tables

Every table and figure the Ours paper needs, with the runs that fill it. The story and the research questions are in [cross_modal_topological_distillation.md](cross_modal_topological_distillation.md). Subsets, IOD/OOD splits and candidate counts are in [../datasets.md](../datasets.md).

## Shared protocol

- **Cells.** `F` = FastVLM-0.5B, `L` = LLaVA-OneVision-0.5B. Tasks are CLS, VQA, RET and GD. The teacher is `raghavlite/B3_Qwen2_2B` unless a table says otherwise.
- **Batching.** Within a cell, every run uses the same global batch and the task-homogeneous sampler: 16 × 8 GPUs = 128 for `F`, 8 × 8 = 64 for `L`. The graph is the micro-batch, so gradient accumulation cannot replace GPUs.
- **Seeds.** Use 42, 43, 44 and report mean ± std. Ablations start with seed 42 only; add seeds for rows that enter the final table.
- **Metric.** Precision@1 (`acc` in `mmeb_eval/summary.json`).
- **Baseline naming.**
  - `student_only` is `L_ret` with the same sampler, and is the SFT row in every rerun table.
  - "Row-wise KL" is bidirectional KL between the teacher's and student's in-batch query→candidate and candidate→query softmax distributions (MobileCLIP / WeMM-Embedding recipe).
- **Status.** `ready` runs today. `needs C<n>` depends on an item in [Code prerequisites](#code-prerequisites).

## Main text

### T1: Classification (RQ1)

Cells `F-CLS` and `L-CLS`, IOD and OOD columns as in [../latex/tables/main_cls.tex](../latex/tables/main_cls.tex).

| Block | Method | Seeds | Runs | Status |
| --- | --- | --- | --- | --- |
| Reported by HieRD (batch 16, no sampler) | Teacher, SFT, MSE, RKD, CKD, EMO, EM-KD, HieRD | — | 0 | copied; fix the EM-KD / LLaVA IN-1K and Avg typo (53.43, 63.59) |
| Rerun under the shared protocol | `student_only` | 3 | 6 | ready |
| | RKD | 3 | 6 | needs C1 |
| | Row-wise KL | 3 | 6 | needs C1, C2 |
| | HieRD (`F` only) | 3 | 3 | needs C1 |
| | **Ours** | 3 | 6 | ready |

OOD columns need eval only (`scripts/eval/cls_ood.sh`), including for the reruns.

| Method | F IOD Avg | F OOD Avg | L IOD Avg | L OOD Avg |
| --- | --- | --- | --- | --- |
| `student_only` | — | — | — | — |
| RKD | — | — | — | — |
| Row-wise KL | — | — | — | — |
| HieRD | — | — | n/a | n/a |
| **Ours** | — | — | — | — |

### T2: Visual question answering (RQ1)

Same design as T1, on `F-VQA` and `L-VQA`, with columns as in [../latex/tables/main_vqa.tex](../latex/tables/main_vqa.tex). This is 27 runs. OOD eval uses `scripts/eval/vqa_ood.sh`.

| Method | F IOD Avg | F OOD Avg | L IOD Avg | L OOD Avg |
| --- | --- | --- | --- | --- |
| `student_only` | — | — | — | — |
| RKD | — | — | — | — |
| Row-wise KL | — | — | — | — |
| HieRD | — | — | n/a | n/a |
| **Ours** | — | — | — | — |

### T3: Retrieval (RQ1)

There are no published numbers for this setting, so every row is a rerun.

- **Columns:**
  - IOD: VisDial, CIRR, VisualNews_t2i, VisualNews_i2t, MSCOCO_t2i, MSCOCO_i2t, NIGHTS, WebQA, plus Avg.
  - OOD: OVEN, FashionIQ, EDIS, Wiki-SS-NQ, plus Avg.
- **Cells:** `F-RET` is required; `L-RET` if compute allows.

| Method | Seeds | Runs (`F` / `+L`) | Status |
| --- | --- | --- | --- |
| `student_only` | 3 | 3 / +3 | needs C5 |
| RKD | 3 | 3 / +3 | needs C1, C5 |
| Row-wise KL | 3 | 3 / +3 | needs C1, C2, C5 |
| HieRD (optional) | 3 | 3 | needs C1, C5 |
| **Ours** | 3 | 3 / +3 | needs C5 |

IOD RET is about 595K training samples, 3.2× CLS. If it is subsampled, use one per-subset cap for every method and for the teacher cache.

| Method | F IOD Avg | F OOD Avg | L IOD Avg | L OOD Avg |
| --- | --- | --- | --- | --- |
| `student_only` | — | — | — | — |
| RKD | — | — | — | — |
| Row-wise KL | — | — | — | — |
| **Ours** | — | — | — | — |

### T4: Ablation A, what to distil (RQ2)

Cells `F-CLS` and `F-VQA`. Every row uses the bipartite query--candidate relation, the sampler and the same batch; only the target changes.

| Row | Tests | Seeds | Runs | Status |
| --- | --- | --- | --- | --- |
| Full $U$ (Ours) | — | 3 | 0 (reuses T1/T2) | ready |
| L1 on $D$ | structure rather than distances | 3 | 6 | needs C3 |
| $H_0$ barcode, $W_2$ | correspondence | 3 | 6 | needs C3 (recover from `9769122`) |
| $U$ truncated to small scales (teacher quantile 10% / 50%) | multiscale | 3 | 6 | needs C3 |
| Rank-normalised $U$ | scale sensitivity ($U$ is only monotone-equivariant) | 3 | 6 | needs C3 |
| *(P2)* TopoAE-style L1 on MST edges | nearest correspondence-based topological loss | 3 | 6 | needs C3 |

λ tuning for the $D$ and barcode rows: 3 values × 1 seed on `F-CLS`, 6 runs. Total: 30–36 runs.

| Target | F-CLS IOD Avg | F-VQA IOD Avg |
| --- | --- | --- |
| Full $U$ (Ours) | — | — |
| L1 on $D$ | — | — |
| $H_0$ barcode | — | — |
| $U$ truncated | — | — |
| Rank-normalised $U$ | — | — |

### T5: Ablation B, where to build it and what is needed (RQ2)

Cell `F-CLS`. Each row changes one axis of Ours.

| Axis | Row | Flag | Seeds | Runs | Status |
| --- | --- | --- | --- | --- | --- |
| Filtration | union point cloud | `--ours_filtration union` | 3 | 3 | needs C4 |
| | per-modality point clouds | `--ours_filtration per_modality` | 3 | 3 | needs C4 |
| Block | query--candidate only | `--ours_merge_block cross` | 3 | 3 | ready |
| | within-side only | `--ours_merge_block within` | 3 | 3 | needs C4 |
| Loss | without `L_ret` | `VARIANT=topo_only` | 3 | 3 | ready (`sensitivity/loss.sh`) |
| Batch | no candidate dedup | `--ours_deduplicate_candidates False` | 3 | 3 | ready |
| | no sampler, Ours | `--ours_task_homogeneous False` | 3 | 3 | ready |
| | no sampler, `student_only` | `VARIANT=student_only --ours_task_homogeneous False` | 3 | 3 | ready |

Total: 24 runs. Start with the 8 seed-42 runs.

| Axis | Row | F-CLS IOD Avg | Δ vs Ours |
| --- | --- | --- | --- |
| — | Ours | — | 0 |
| Filtration | union | — | — |
| | per-modality | — | — |
| Block | cross | — | — |
| | within | — | — |
| Loss | w/o `L_ret` | — | — |
| Batch | w/o dedup | — | — |
| | w/o sampler | — | — |
| | `student_only` w/o sampler | — | — |

Report NIGHTS (I → I, no modality gap) per task if T3 exists: the bipartite-vs-union gap is expected to be smallest there.

### A1: Structural analysis (RQ3)

No training. The inputs are the checkpoints from T1–T4 and teacher embedding dumps for the same subsets (`tools/eval_topology.py`).

| Output | Content | Status |
| --- | --- | --- |
| Table | teacher MST-edge recall, component ARI, recall@k and Spearman for every T1/T2 method | ready |
| Table rows | the same metrics for the $D$ and barcode rows of T4; a barcode student with low loss but low MST-edge recall supports the correspondence argument | needs T4 |
| Scatter | structure fidelity vs MMEB accuracy, one point per method × seed × cell | needs C6 |
| Diagnostic | teacher vs student kNN overlap against Pearson correlation of distances | needs C6 |

| Method | MST-edge recall | ARI | recall@10 | Spearman | IOD Avg |
| --- | --- | --- | --- | --- | --- |
| `student_only` | — | — | — | — | — |
| RKD | — | — | — | — | — |
| Row-wise KL | — | — | — | — | — |
| HieRD | — | — | — | — | — |
| **Ours** | — | — | — | — | — |

### F1: Toy counterexample (Section 2 figure)

CPU only. On synthetic query--candidate embeddings with a permuted student:

- the barcode loss is zero while $\mathcal L_{\mathrm{topo}}$ is not;
- gradient descent on each loss yields different Precision@1.

Needs C6.

## Appendix

| ID | Content | Cells | Runs | Status |
| --- | --- | --- | --- | --- |
| A2 | λ sweep {0.1, 0.3, 1, 3, 10} (figure) | `F-CLS`, `F-VQA`, 1 seed | 8 (1.0 reused) | ready |
| A3 | Global batch {32, 64, 128, 256}, **Ours and `student_only` at every size** (figure) | `F-CLS`, 1 seed | 6 | ready |
| A4 | Cost: time per step, peak VRAM, MST time vs $B$; Ours, RKD, row-wise KL, HieRD; with and without teacher cache | — | 0 | logs plus a micro-benchmark |
| A5 | Recall@3 and MRR for T1–T3 (HieRD reports them) | — | 0 | needs C7; eval currently computes only `acc` |
| A6 | Per-dataset results for T4 and T5 | — | 0 | — |
| A7 | Training dynamics: `topo_loss` floor and MST-edge recall over steps | — | 0 | logs |
| A8 | Grounding (IOD `MSCOCO`; OOD `Visual7W-Pointing`, `RefCOCO`, `RefCOCO-Matching`): `student_only`, RKD, row-wise KL, Ours | `F-GD`, 1–3 seeds | 4–12 | needs C1, C2, C5 |
| A9 | Capacity gap with B3-Qwen2-7B teacher: `student_only`, RKD, row-wise KL, Ours | `F-CLS`, 1–3 seeds | 4–12 | needs C1, C2, a 7B teacher cache |

A8 is low priority. HieRD Table 14 shows the B3-Qwen2-2B teacher below SFT on `MSCOCO` and `Visual7W-Pointing`.

## Code prerequisites

| ID | Work | Unblocks |
| --- | --- | --- |
| C1 | Enable the task-homogeneous sampler for any criterion. Today it is gated on `kd_loss_type == ours` in `src/training/dataloader.py`. | T1–T3 reruns, A8, A9 |
| C2 | Bidirectional row-wise KL criterion that accepts `--teacher_embedding_cache` | T1–T3, A8, A9 |
| C3 | `--ours_target {merge, distance, barcode, truncated, rank, mst_edge}` | T4 |
| C4 | `--ours_filtration {bipartite, union, per_modality}` and `--ours_merge_block within` | T5 |
| C5 | RET / GD wiring: per-method launchers, `TASK=ret\|grounding` in `precompute_teacher_embeddings.sh`, eval groups in `src/evaluation/benchmarks.py`, `tools/check_paper_settings.py` | T3, A8 |
| C6 | Pearson on distances in `neighborhood_preservation`, a structure-vs-accuracy aggregation script, the toy script | A1, F1 |
| C7 | Recall@k and MRR in `tools/eval_mmeb.py` | A5 |

Every new option needs a check in `tools/misc/test_ours.py`.

## Order and budget

| Step | Work | Training runs |
| --- | --- | --- |
| 0 | C1–C4, C6 | — |
| 1 | **Go / no-go** on `F-CLS` seed 42: Ours, `student_only`, row-wise KL, RKD, L1 on $D$, barcode | 6 |
| 2 | T1, T2 | 54 |
| 3 | T4 and T5 at seed 42, then seeds for the rows kept | 22 → 54 |
| 4 | A1, F1 | 0 |
| 5 | C5, then T3 | 12–24 |
| 6 | Appendix A2, A3, A8, A9 | 22–40 |

Main text is about 145 training runs; the appendix adds 20–40.

- **Stop condition.** If Ours does not beat row-wise KL and L1 on $D$ in step 1, revisit the story before step 2.
- **If compute runs short, cut in this order:**
  1. `L-RET`
  2. A8 and A9
  3. seeds on T5
- **Never cut T4.** It carries the motivation.
