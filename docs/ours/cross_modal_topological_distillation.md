# Research Brief: Cross-Modal Topological Distillation for VLM2Vec

> **Working name:** Ours
> **Core claim:** for universal multimodal embedders, the structure a student should inherit is the **topology of the query--candidate retrieval relation** -- its multiscale connectivity -- compared under teacher--student row correspondence, rather than the geometry or topology of each modality's point cloud.

Citations marked *[verify]* come from a literature survey (2026-09-13) and must be checked against the paper before they enter `references.bib`.

## 1. Motivation

The *Universal Geometry of Embeddings* paper (Jha et al., arXiv 2505.12540) suggests that embeddings from different architectures and dimensions can share a common latent geometric structure even when their coordinates are incompatible. Its vector-space-preservation objective is essential *[verify against its ablation]*, indicating that **relational geometry is a key transferable signal**.

This connects naturally to VLM2Vec / VLM2Vec-V2, which express every task -- classification, VQA, retrieval, grounding -- as ranking a candidate set for an instructed multimodal query. What a compact student has to inherit from its teacher is therefore how queries relate to candidates, not the coordinates of either.

**Structure rather than distances.** vec2vec's preservation term matches dot-product values, so on its own it motivates relation-level transfer, not a topological target over plain distance matching. That second step rests on evidence that independently trained models agree on neighbourhood structure far more than on metric distances. After calibrating similarity metrics for model width and depth, global/metric convergence largely disappears while local neighbourhood relationships persist (Gröger, Wen & Brbić, *Revisiting the Platonic Representation Hypothesis: An Aristotelian View*, ICML 2026, arXiv 2602.14486 *[verify]*).

Coordinate matching (MSE with a projector) and distance matching (RKD, similarity-matrix losses) therefore over-constrain a much smaller student. The same intuition underlies relation- or order-level distillation such as DIST (Huang et al., NeurIPS 2022, arXiv 2205.10536), Margin-MSE (Hofstätter et al., arXiv 2010.02666) and RankDistil (Reddi et al., AISTATS 2021).

**Relation, not point clouds.** Topology is a natural way to transfer structure without coordinates, but existing topological distillation and alignment methods build their filtrations on point clouds -- per modality, per layer, or over a union:

- TopKD (Kim et al., ICML 2024) and PsHD (Fan et al., NeurIPS 2024): persistence diagrams / images of feature clouds;
- ToMCLIP (You et al., arXiv 2510.10889 *[verify]*): sliced Wasserstein on H0 diagrams of an embedding cloud;
- RTD / RTD-AE (Barannikov et al., ICML 2022, arXiv 2201.00058; Trofimov et al., ICLR 2023, arXiv 2302.00136): cross-barcodes of two clouds with correspondence;
- HC (Zhang et al., NeurIPS 2024) and ToMA (You et al., arXiv 2604.26370 *[verify]*): per-modality complexes linked through the image--text pairing.

These describe the shape of each representation space, not how queries connect to candidates. We instead distil the topology of the cross-modal relation itself: the filtration of the bipartite query--candidate distance graph. A working hypothesis to test (RQ2): under the modality gap (Liang et al., NeurIPS 2022), a union-cloud filtration is dominated by within-side distances, whereas the bipartite graph makes within-side connectivity arise only through the cross-modal relation.

**Correspondence follows from that choice.** Teacher and student embed the same queries and candidates, so their relational structure must be compared under row-wise correspondence (Section 2). This point is already made for point clouds by RTD-AE, which notes that Wasserstein distance between diagrams is permutation-invariant although a natural one-to-one correspondence exists. We use it as a design requirement, not as the headline contribution. "Identity" means that teacher row *i* and student row *i* are the same sample; it is **not** an extra class-label annotation.

**Practical property (not a novelty claim).** Only final normalised teacher embeddings are read: no hidden states, attention maps, parser, clustering or teacher-to-student projector. The teacher can be precomputed once and cached. This contrasts with HieRD (ICML 2026), which needs teacher attention and hidden states, per-layer projectors, DBSCAN and a syntactic parser. Other black-box, projector-free similarity-distribution losses exist (TinyCLIP, MobileCLIP, UniME, WeMM-Embedding), so this property is not claimed as new.

## 2. Why the Target Must Be Correspondence-Aware

Let a batch contain indexed query and candidate vertices

$$
Q^M = \{q_i^M\}_{i=1}^{n_q}, \qquad
C^M = \{c_j^M\}_{j=1}^{n_c}, \qquad M\in\{T,S\},
$$

where $q_i^T$ corresponds to $q_i^S$ and $c_j^T$ to $c_j^S$. Define $D^M_{ij}=1-\cos(q_i^M,c_j^M)$ and the bipartite filtration

$$
G^M_\varepsilon=
\left(Q^M\cup C^M,
\{(q_i,c_j):D^M_{ij}\le\varepsilon\}\right).
$$

For $H_0$, the finite barcode is the sorted multiset of the $n_q+n_c-1$ minimum-spanning-tree edge weights. It records **when some components merge**, not **which query and candidate identities merge**.

### Candidate-permutation counterexample

Let $P$ be a non-identity permutation of the candidate columns and set $D^S=D^TP$. The two weighted bipartite graphs are isomorphic after permuting candidates, so their $H_0$ barcodes (and $H_1$ birth summaries) are identical and any diagram loss is zero. The correct candidate attached to each query has nevertheless changed, so Precision@1 can fall from 1 to 0.

Scope of the argument: it applies to diagram-level objectives (TopKD, PsHD, the diagram term of ToMCLIP, and a barcode loss on the relation itself). It does **not** apply to correspondence-based topological losses such as TopoAE, RTD, HC or ToMA, which must be compared empirically instead.

### Note on $H_1$

A bipartite graph has no triangles, so its flag complex has no 2-simplices and every $H_1$ class born by a non-MST edge persists forever. On this filtration, $H_1$-birth is therefore all of $H_1$, not an approximation. Matching those births still only matches statistics of non-MST distances, and MST / non-MST edge alignment already appears in vision--language alignment (ToMA), so $H_1$ terms are baselines, not part of the method.

## 3. Method: Correspondence-Aware Persistent Connectivity

### 3.1 Cross-modal merge times

For vertices $a,b\in V=Q\cup C$, the merge time under model $M$ is

$$
U^M_{ab}
=
\min_{\pi:a\rightsquigarrow b}
\max_{e\in\pi}D^M_e
=
\inf\{\varepsilon:a\text{ and }b\text{ are connected in }G^M_\varepsilon\},
$$

where $\pi$ ranges over alternating query--candidate paths. $U^M$ is the cophenetic (minimax, single-linkage) ultrametric of the bipartite graph and can be read off its MST. Its blocks:

- $U_{q_i,c_j}$: the scale at which a query and a candidate become connected, directly or through semantic bridges;
- $U_{q_i,q_k}$: when two queries become connected through shared candidates;
- $U_{c_j,c_l}$: when two candidates become connected through shared queries.

The within-side blocks contain no within-modality distances; they are induced entirely by the cross-modal relation.

### 3.2 Objective

$$
\mathcal L_{\mathrm{topo}}
=
\frac{2}{|V|(|V|-1)}
\sum_{a<b}
\left|U^T_{ab}-U^S_{ab}\right|,
\qquad
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{ret}}
+
\lambda_{\mathrm{topo}}\mathcal L_{\mathrm{topo}}
}
$$

$\mathcal L_{\mathrm{ret}}$ is the standard in-batch contrastive loss. There is one distillation term and one coefficient.

### 3.3 Multiscale interpretation and the relaxation view

For one aligned pair,

$$
\left|U^T_{ab}-U^S_{ab}\right|
=
\int
\left|
\mathbf 1[U^T_{ab}\le\varepsilon]
-
\mathbf 1[U^S_{ab}\le\varepsilon]
\right|
\,d\varepsilon ,
$$

so $\mathcal L_{\mathrm{topo}}$ is the integrated disagreement between teacher and student connectivity partitions over all filtration scales.

$U$ is also a many-to-one coarsening of $D$: it depends only on the MST's bottleneck weights. Matching $U$ is thus a principled **relaxation** of matching $D$. It keeps connectivity at every scale and leaves the non-bottleneck geometry free, which is the level of structure the motivation argues transfers across models.

## 4. Properties and Limitations

Statable with little effort:

| property | statement | source |
| --- | --- | --- |
| stability | $\max_{a,b}\lvert U^1_{ab}-U^2_{ab}\rvert \le \max_{a,b}\lvert D^1_{ab}-D^2_{ab}\rvert$ | Carlsson & Mémoli, JMLR 2010, Lemma 15. The proof only perturbs chains and never uses the triangle inequality, so it holds for cosine distance on a bipartite edge set; say so explicitly. |
| every-scale partitions | $a,b$ share a component at level $r$ iff $U_{ab}\le r$ | ibid., Remark 13 |
| canonicity | single linkage is the unique hierarchical clustering method satisfying their axioms | ibid., Theorem 18. Stated for metric spaces; use as motivation only. |
| monotone equivariance | $U_{f\circ D}=f\circ U_D$ for non-decreasing $f$; the merge order is invariant, the values are not | elementary. Raw L1 on $U$ is therefore **not** scale- or temperature-invariant; do not claim it is. |
| nearest-neighbour value | $\min_c U_{qc}=\min_c D_{qc}$ for every query | elementary |
| identifiability | barcode losses vanish under candidate permutation; $\mathcal L_{\mathrm{topo}}$ does not | Section 2 |

Limitations to state rather than let reviewers find:

- **$U$ does not determine Recall@1.** With $D(q_1,c_1)=0.2$, $D(q_1,c_2)=0.5$, $D(q_2,c_2)=0.1$, $D(q_2,c_1)=0.15$, the path $q_1\!-\!c_1\!-\!q_2\!-\!c_2$ gives $U(q_1,c_2)=0.2=U(q_1,c_1)$, a tie between the correct and the wrong candidate. Never claim that matching $U$ preserves retrieval.
- **Sparse supervision.** $U$ depends only on the MST, so gradients reach about $n_q+n_c-1$ edges per batch and nothing pushes a different edge into the tree. The loss has a floor (see `docs/ours_implementation.md`), and $\mathcal L_{\mathrm{ret}}$ is required.
- **Relation to existing objects.** $U$ is the induced ultra-matrix of a labelled merge tree (Gasparovic et al., arXiv 1908.00063 *[verify]*), so $\mathcal L_{\mathrm{topo}}$ is an L1 analogue of a labelled merge-tree distance. Its zero set may coincide with that of symmetric $H_0$-RTD; this derivation is unchecked and must be verified. Position both explicitly.

## 5. Positioning and Claims

| transferred object | representatives | black-box | correspondence | multiscale |
| --- | --- | --- | --- | --- |
| features + projector | MSE, TALAS, jina-embeddings-v5, HieRD $L_{hid}$ | partly | yes | no |
| tokens / spans / clusters (white-box) | HieRD, EM-KD, EMO | no | yes | no |
| full pairwise distances | RKD (inside HieRD's base loss), SP, Topology Distillation (KDD 2021) | yes | yes | no |
| row-wise similarity distributions | TinyCLIP, MobileCLIP, UniME, WeMM-Embedding (arXiv 2608.24053 *[verify]*) | yes | yes | no |
| diagram-level topology | TopKD, PsHD, ToMCLIP $L_{ta}$ | yes | no | yes |
| correspondence-based topology on point clouds | TopoAE, RTD, HC, ToMA | yes | yes | partly |
| **Ours** | single-linkage ultrametric of the bipartite query--candidate relation | yes | yes | yes |

**Claims we can make:**

1. A distillation objective that matches the correspondence-aware single-linkage ultrametric of the cross-modal retrieval relation. No prior work found matches cophenetic / minimax ultrametrics between two models, nor builds topological KD on the bipartite query--candidate graph. This is absence of evidence, so word it accordingly.
2. The properties in Section 4, with its limitations stated.
3. Empirically, under identical batching, the objective outperforms distance matching, similarity-distribution KD and topological baselines, and is competitive with or better than white-box HieRD at lower cost. This claim depends on results.
4. Students trained with it preserve the teacher's connectivity better on metrics that are not the training loss, and that preservation tracks accuracy. This claim depends on results.

**Claims to avoid:**

| claim | contradicted by |
| --- | --- |
| first topological / persistent-homology KD | TopKD, PsHD, ToMCLIP |
| first correspondence-aware topological loss | TopoAE, RTD, HC, ToMA |
| first relational or black-box projector-free KD for multimodal embedders | RKD in HieRD; MobileCLIP, WeMM-Embedding |
| task-homogeneous batching as a contribution | standard in GTE, VLM2Vec-V2, WeMM-Embedding (which reports −3.4 MMEB-V2 without it) |
| "small students lack capacity" as a premise | Stanton et al., NeurIPS 2021; WeMM-Embedding distils 9B→2B with row-wise KL. Test it instead (RQ1, capacity gap). |
| scale or temperature invariance | $U$ is only monotone-equivariant (Section 4) |
| gradients through MST witness edges as a contribution | TopoAE, Chierchia & Perret (NeurIPS 2019), HC, ToMA route gradients the same way |
| matching $U$ preserves Recall@1 | counterexample in Section 4 |

## 6. Key Research Questions

| # | question | desired evidence |
| --- | --- | --- |
| RQ1 | Does distilling the retrieval-relation topology help beyond existing teacher signals? | Main tables: gains over `student_only` under the same sampler and global batch, and over RKD, row-wise KL and HieRD, consistent across both students and both tasks. |
| RQ2 | Which design choices produce the gain? | One ablation table on `fastvlm_cls`. Each row changes a single axis of Ours and keeps everything else fixed. **Filtration:** bipartite relation vs per-modality clouds vs union cloud, which tests "relation, not point clouds". **Target:** $U$ vs L1 on $D$ vs $H_0$ barcode, which tests "structure rather than distances" and "correspondence". **Block:** all vs query--candidate only. |
| RQ3 | Does the student actually inherit the teacher's structure, and does that track accuracy? | Teacher MST-edge recall, component ARI and kNN overlap, which are **not** the training loss, measured for Ours and for the baselines, together with their correlation with MMEB accuracy. Add a diagnostic showing that teacher and student agree more on neighbourhoods than on distances. |

Each part of the motivation is tested inside RQ2 rather than as a separate question. That keeps the main text to one main table, one ablation table and one structural analysis. λ and batch-size sensitivity, cost and OOD results go to the appendix.

Black-box access is a property by construction, not a research question. Report it as cost instead: time per step and VRAM against HieRD, with and without the teacher cache.

The earlier "universal topology" extension (a Wasserstein barycenter of several teachers' diagrams) is dropped. It is built on identity-agnostic diagrams; at most it is future work.

## 7. Experimental Requirements

**Controlled comparison.** Every baseline number in `docs/latex/tables/` is copied from HieRD's Table 1: batch 16 on one A100, no task-homogeneous sampler. Ours runs at a global batch of 128 with the sampler, which is enabled only for `--kd_loss_type ours` (`src/training/dataloader.py`). Those numbers cannot carry the main comparison. `student_only` at the same batch is the floor, and at least SFT, RKD and HieRD must be rerun at the same global batch with the same sampler.

**Baselines needed under that setup, each with its own tuned coefficient:**

- L1 / Huber on $D$ (RQ2, target)
- bidirectional row-wise KL on the in-batch similarity matrix (MobileCLIP / WeMM-Embedding recipe)
- $H_0$ barcode on the relation, optionally with $H_1$ births (RQ2, target)
- point-cloud / union $H_0$ (TopKD-style) and $U$ on point clouds (RQ2, filtration)
- TopoAE-style MST-edge distances or $H_0$-RTD

The barcode, point-cloud and distance variants existed in the CMTop criterion at commit `9769122` and can be recovered from there: `h0_deaths`, `h1_births`, `wasserstein2_sorted`, `point_cloud_mst_edges`, the `cross_modal` / `point_cloud` / `union` modes and the `vsp` control. Drop its endpoint-KD term when rerunning them.

**Ablations and sensitivity:** `--ours_merge_block all|cross`, `--ours_deduplicate_candidates`, the $\lambda_{\mathrm{topo}}$ sweep, the global batch size (32 / 64 / 128 / 256) and the loss-term rows (`scripts/train/ours/sensitivity/loss.sh`). Use 3 seeds wherever the gap is below about 0.5 points.

**Scope:**

- **Tasks.** Fill the OOD columns, and add at least one MMEB retrieval cell, since the method is pitched on the retrieval relation.
- **Capacity gap.** Add a B3-Qwen2-7B teacher; HieRD reports this setting.

**First run to do:** Ours vs bidirectional row-wise KL on `fastvlm_cls` with the teacher cache. If Ours does not beat it under identical batching, the story has to change before anything else is run.
