# Research Brief: Correspondence-Aware Cross-Modal Topological Distillation

> **Working name:** CM-Merge
> **Core claim:** for multimodal retrieval, the transferable topological object is not an unlabeled persistence barcode, but the **labeled multiscale connectivity hierarchy of the query--candidate relation**.

## 1. Motivation

The *Universal Geometry of Embeddings* line of work suggests that models with different architectures and embedding dimensions can share relational structure even when their coordinates are incompatible. This makes relation-level distillation attractive for VLM2Vec / VLM2Vec-V2, whose tasks are all expressed as ranking a candidate set for an instructed multimodal query.

Existing distillation objectives transfer one of three weaker objects:

1. endpoint features, which require a projector and assume coordinate compatibility;
2. the full pairwise similarity matrix, which preserves all metric detail but does not isolate multiscale structure;
3. an ordinary persistence diagram of each modality or of a union point cloud, which does not directly represent the retrieval relation.

A natural first attempt is to construct a weighted bipartite graph from cross-modal distances and match its teacher and student persistence diagrams. However, **diagram-level matching is insufficient for retrieval** because a persistence diagram discards the identities of the queries and candidates participating in each topological event.

The proposed method therefore distills the complete labeled $H_0$ connectivity evolution of the cross-modal relation. It remains black-box and dimension-agnostic: only the final normalized teacher embeddings are required, and no teacher hidden states, attention maps, logits head, or learned teacher-to-student projector are used.

## 2. Why Unlabeled Diagram Matching Is Not Enough

Let a batch contain labeled query and candidate vertices

$$
Q^M = \{q_i^M\}_{i=1}^{n_q}, \qquad
C^M = \{c_j^M\}_{j=1}^{n_c}, \qquad M\in\{T,S\},
$$

where vertex $q_i^T$ corresponds to $q_i^S$ and $c_j^T$ corresponds to $c_j^S$. Define

$$
D^M_{ij}=1-\cos(q_i^M,c_j^M).
$$

The associated bipartite filtration is

$$
G^M_\varepsilon=
\left(Q^M\cup C^M,
\{(q_i,c_j):D^M_{ij}\le\varepsilon\}\right).
$$

For $H_0$, the finite persistence barcode is only the sorted multiset of the $n_q+n_c-1$ minimum-spanning-tree edge weights. It records **when some components merge**, but not **which labeled queries and candidates merge**.

### Candidate-permutation counterexample

Let $P$ be a non-identity permutation of the candidate columns and set

$$
D^S=D^TP.
$$

The two weighted bipartite graphs are isomorphic after relabeling the candidate vertices. Their $H_0$ persistence diagrams, and the birth-only $H_1$ summaries of their 1-skeletons, are identical. An unlabeled diagram loss is therefore zero. Nevertheless, the correct candidate attached to each query has changed, so Precision@1 can fall from perfect retrieval to zero.

This exposes the central identifiability problem:

> A retrieval distillation objective must preserve both multiscale connectivity and the correspondence of the vertices that participate in that connectivity.

### Why birth-only $H_1$ is not the main solution

In a bipartite 1-skeleton there are no 2-simplices, so every cycle that is born persists forever. The non-MST edges are therefore all $H_1$ birth edges. Matching the earliest such births mainly matches low-order statistics of non-MST distances; it does not preserve cycle membership or finite persistence. Recent vision--language alignment work already uses MST/$H_0$-death and non-MST/$H_1$-birth edges, so this construction should be treated as a baseline rather than the main novelty.

## 3. Main Method: Labeled Persistent Connectivity

### 3.1 Cross-modal merge times

For vertices $a,b\in V=Q\cup C$, define their merge time under model $M$ as

$$
U^M_{ab}
=
\min_{\pi:a\rightsquigarrow b}
\max_{e\in\pi}D^M_e,
$$

where $\pi$ ranges over alternating query--candidate paths in the complete weighted bipartite graph.

Equivalently,

$$
U^M_{ab}
=
\inf\{\varepsilon:a\text{ and }b\text{ are connected in }G^M_\varepsilon\}.
$$

$U^M$ is the cophenetic/minimax ultrametric induced by the cross-modal minimum spanning tree. Its entries have a retrieval-specific interpretation:

- $U_{q_i,c_j}$ is the scale at which a query and candidate become connected directly or through semantic bridges;
- $U_{q_i,q_k}$ measures when two queries become connected through shared candidate neighborhoods;
- $U_{c_j,c_l}$ measures when two candidates become connected through shared queries.

The within-query and within-candidate blocks therefore contain no within-modality distances: they are induced entirely by the cross-modal relation.

### 3.2 A single topology distillation term

The proposed loss compares corresponding entries of the labeled merge matrices:

$$
\mathcal L_{\mathrm{Merge}}
=
\frac{2}{|V|(|V|-1)}
\sum_{a<b}
\left|U^T_{ab}-U^S_{ab}\right|.
$$

The complete training objective is

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{retrieval}}
+
\lambda_{\mathrm{Merge}}\mathcal L_{\mathrm{Merge}}
}
$$

where $\mathcal L_{\mathrm{retrieval}}$ is the standard VLM2Vec contrastive objective. The main method has one distillation term and one coefficient. Endpoint KD, full-matrix geometry KD, and $H_1$ terms are comparison baselines, not components of the proposed objective.

### 3.3 Multiscale interpretation

For one labeled vertex pair,

$$
\left|U^T_{ab}-U^S_{ab}\right|
=
\int
\left|
\mathbf 1[U^T_{ab}\le\varepsilon]
-
\mathbf 1[U^S_{ab}\le\varepsilon]
\right|
\,d\varepsilon.
$$

Thus $\mathcal L_{\mathrm{Merge}}$ is the integrated disagreement between teacher and student labeled connectivity partitions across every filtration scale. This is the topological object that the ordinary $H_0$ barcode compresses away.

## 4. Theoretical Properties to Establish

The paper should contain short proofs of the following statements.

### Proposition 1: recovery of the labeled filtration

For every threshold $\varepsilon$,

$$
a\sim_\varepsilon b
\quad\Longleftrightarrow\quad
U_{ab}\le\varepsilon.
$$

Consequently, $U$ determines the complete labeled connected-component partition at every scale. The ordinary $H_0$ barcode is a function of $U$, but $U$ is not a function of the barcode.

### Proposition 2: permutation sensitivity

An unlabeled persistence-diagram distance is invariant to arbitrary vertex relabeling. In contrast, $\mathcal L_{\mathrm{Merge}}$ is invariant only to a simultaneous relabeling applied to both teacher and student. A candidate permutation applied to the student alone is detected unless it is an automorphism of the labeled teacher hierarchy.

### Proposition 3: stability

For two distance matrices on the same labeled bipartite vertex set,

$$
\|U(D)-U(D')\|_\infty
\le
\|D-D'\|_\infty.
$$

The proof follows because the maximum edge weight of every fixed path changes by at most $\|D-D'\|_\infty$, and taking the minimum over paths preserves the bound.

### Proposition 4: exact almost-everywhere gradients

For distinct edge weights, every $U_{ab}$ equals the weight of the unique bottleneck edge on the MST path between $a$ and $b$. The MST combinatorics and the bottleneck identity are locally constant away from edge-weight ties. Gradients can therefore be routed exactly to the selected cross-modal distances almost everywhere without a soft topology relaxation.

### Complexity

1. Compute the bipartite MST from $n_qn_c$ edges.
2. Process its edges in Kruskal order. When an edge merges components $A$ and $B$, assign that edge as the merge witness for all pairs in $A\times B$.
3. Gather the differentiable student distances at those witness indices.

Every unordered vertex pair is assigned once, so constructing the labeled merge matrix after the MST costs $O(|V|^2)$ time and memory. The method adds no encoder forward pass.

## 5. Relation Construction and Training Protocol

These are part of defining a valid estimator of the retrieval topology, not additional loss terms.

### 5.1 Task-conditioned graphs

MMEB contains heterogeneous candidate spaces and task instructions. Cross-task edges are usually easy negatives whose distances reflect task or modality identity rather than useful semantic neighborhoods. Each topology graph must therefore contain samples from one dataset/task.

Under DDP, all ranks must follow the same task schedule at a given step before their embeddings are gathered. Every baseline must use the same task-conditioned batches.

### 5.2 Unique candidate vertices

Repeated class labels or answers must be represented by one canonical candidate vertex within the global topology batch. Deduplicate candidates by their canonical text/image identity before constructing $D$. Preserve the many-query-to-one-candidate positive mapping separately.

This prevents duplicated labels from creating zero-length or frequency-driven topological events.

### 5.3 Sufficient topology batch size

The topology batch must be large enough to contain nontrivial shared neighborhoods and semantic bridges. Target an effective task-homogeneous topology batch of 128--256, implemented with GradCache or chunked encoding if it does not fit in memory. Ordinary gradient accumulation is insufficient if the topology loss is still computed separately on each microbatch.

Teacher embeddings should be precomputed. This keeps the method black-box, removes teacher forward passes, and makes large topology batches practical.

## 6. Experimental Questions and Controls

### 6.1 Primary decision

Does correspondence-aware merge distillation improve OOD multimodal retrieval beyond strong endpoint and relation-matrix distillation while remaining projector-free and computationally lightweight?

Primary metric:

- official MMEB OOD macro average, reported by meta-task and dataset.

Guardrails:

- MMEB ID performance;
- fraction of datasets improved;
- training time and peak memory;
- teacher top-$k$ neighborhood fidelity;
- seed variance and sensitivity to $\lambda_{\mathrm{Merge}}$.

### 6.2 Main comparison table

| Variant | Transferred object | Role |
| --- | --- | --- |
| Student only | no teacher signal | task baseline |
| Endpoint KD | individual final embeddings | pointwise baseline |
| Relation-matrix KD / VSP | all labeled cross-distances | strongest geometric baseline |
| Point-cloud $H_0$ KD | separate modality topology | non-relational topology control |
| Bipartite diagram $H_0$ | sorted MST weights | original unlabeled proposal |
| Persistence-pairing / critical-edge KD | labeled selected edges | correspondence-aware topology baseline |
| **CM-Merge** | complete labeled multiscale connectivity | proposed method |

Do not combine multiple KD baselines in the main CM-Merge row. Any combination experiment belongs in the appendix and must not define the claimed method.

### 6.3 Required settings

- Full MMEB evaluation across classification, VQA, retrieval, and visual grounding, with ID and OOD reported separately.
- MMEB-V2 video and visual-document tasks if compute permits; these are a strong test of modality-independent relation transfer.
- At least three teacher--student pairs:
  - one same-family capacity gap;
  - one cross-architecture pair;
  - one pair with different embedding dimensions.
- Three to five seeds for the main baseline, strongest competing KD method, and CM-Merge.
- Matched training data, number of updates, global batch size, sampler, augmentation, and compute budget across methods.

### 6.4 Isolating ablations

Run each as a one-variable comparison:

1. sorted barcode versus labeled merge matrix;
2. cross-modal relation versus query/candidate point clouds;
3. task-mixed versus task-conditioned topology batches;
4. duplicated versus unique candidate vertices;
5. topology batch size $\{32,64,128,256\}$;
6. query--candidate block of $U$ versus the full induced $U$;
7. live teacher versus cached fp16 teacher embeddings as a numerical-equivalence check.

Only tune $\lambda_{\mathrm{Merge}}$ on a held-out development set. Report a coarse log-scale sensitivity range rather than per-dataset coefficient search.

## 7. Diagnostics That Test the Claim

Retrieval accuracy alone cannot establish that the proposed mechanism worked.

### Identifiability stress test

Apply controlled candidate permutations to a teacher distance matrix. Show that:

- the bipartite persistence-diagram distance remains zero;
- relation-matrix and CM-Merge discrepancies increase;
- CM-Merge identifies the changed connectivity assignments across scales.

### Structural fidelity

Report metrics not identical to the training loss:

- teacher--student top-$k$ candidate overlap;
- Spearman correlation of complete candidate rankings;
- teacher MST-edge recall under student distances;
- adjusted Rand agreement of connected-component partitions at teacher distance quantiles;
- correlation across datasets between retrieval gain and reduction in merge distortion.

### Batch stability

For fixed checkpoints, repeatedly subsample topology batches and report the variance of barcode discrepancy and labeled merge discrepancy. The proposed object should be more informative without becoming less stable.

### Cross-teacher motivation study

Before multi-teacher training, measure labeled merge-hierarchy agreement among several strong teachers on the same examples. Compare against candidate-permuted and random-model controls. This tests the motivating claim that a transferable cross-model relational backbone exists without adding a multi-teacher loss to the method.

## 8. Success and Stop Criteria

Proceed to a full ICLR-scale submission only if the following hold:

1. CM-Merge improves over the strongest relation-matrix or correspondence-aware topology baseline on the OOD macro average by more than run-to-run noise.
2. Improvements are distributed across datasets and at least two meta-tasks rather than driven by one classification benchmark.
3. The gain repeats across at least two teacher--student pairs, including the cross-architecture pair.
4. The candidate-permutation experiment validates the identifiability argument.
5. Added training time is small relative to the student forward/backward pass and inference cost is unchanged.

If CM-Merge only beats student-only or endpoint KD but not full relation-matrix KD, the evidence does not support an ICLR-level topological contribution. If it reduces merge discrepancy without improving retrieval, the topological object is measurable but not task-useful.

## 9. Optional Higher-Order Extension

Do not use birth-only $H_1$ of the bipartite 1-skeleton as the main extension. If $H_0$ merge distillation saturates and a higher-order signal is empirically necessary, use a relation-native Dowker complex:

$$
\mathcal D_Q^\varepsilon(D)
=
\left\{
\sigma\subseteq Q:
\exists c_j\in C,
\max_{q_i\in\sigma}D_{ij}\le\varepsilon
\right\}.
$$

A simplex now means that a group of queries shares a candidate witness. Higher-dimensional simplices can fill cycles, so $H_1$ has finite births and deaths and genuinely represents many-to-many neighborhood overlap. Any student--teacher objective must still preserve the identities of critical simplices or witnesses; plain Dowker diagram matching alone does not solve the correspondence problem.

This extension should enter the main paper only if it yields a clean gain over CM-Merge with a single fixed formulation. Otherwise it belongs in future work.

## 10. Intended ICLR Contribution

> Persistence diagrams summarize how many topological events occur and at which scales, but multimodal retrieval also depends on which queries and candidates participate in those events. CM-Merge distills the labeled persistent connectivity of the task-conditioned query--candidate relation, providing projector-free structural supervision across model architectures and embedding dimensions.

The intended contributions are:

1. an identifiability analysis showing why unlabeled diagram matching can be exactly blind to catastrophic retrieval permutations;
2. a correspondence-aware topological distillation objective based on the labeled merge hierarchy of a bipartite retrieval filtration;
3. stability, recovery, complexity, and almost-everywhere differentiability results;
4. broad MMEB/MMEB-V2 evidence against strong pointwise, geometric, and topology-aware baselines.

## 11. Closest Work to Position Against

- [Topological Autoencoders](https://proceedings.mlr.press/v119/moor20a.html): shows why persistence values without persistence pairings can produce meaningless label permutations and motivates correspondence-aware topological signatures.
- [Do Topological Characteristics Help in Knowledge Distillation?](https://proceedings.mlr.press/v235/kim24aj.html): establishes persistence-based KD on latent point clouds, but not the labeled cross-modal retrieval relation.
- [Topological Alignment of Shared Vision-Language Embedding Space](https://arxiv.org/abs/2510.10889): persistence-diagram alignment for multilingual vision--language embeddings.
- [Topology-Aware Representation Alignment for Semi-Supervised Vision-Language Learning](https://arxiv.org/abs/2604.26370): aligns $H_0$-death and $H_1$-birth critical-edge directions across paired modalities; this makes MST/non-MST edge selection a necessary baseline rather than the novelty.
- [Relational Persistent Homology for Multispecies Data](https://arxiv.org/abs/2308.06205): establishes Dowker and witness complexes as topology-native constructions for relations between heterogeneous entity types.
- [VLM2Vec](https://arxiv.org/abs/2410.05160) and [VLM2Vec-V2](https://arxiv.org/abs/2507.04590): provide the universal multimodal retrieval setting, task taxonomy, and evaluation protocol.

## 12. Implementation Status and Next Experiments

Implemented in this repository:

1. task-homogeneous global sampling synchronized across DDP ranks;
2. stable candidate identities and canonical candidate deduplication;
3. exact MST merge witnesses, labeled merge matrix $U$, and gradient gather path;
4. relation-matrix, point-cloud $H_0$, unlabeled bipartite $H_0$, critical-edge, and CM-Merge modes behind one criterion;
5. definition-level minimax, stability, gradient, DDP sampler, and candidate-permutation tests;
6. rectangular query/candidate evaluation with labeled merge discrepancy and neighborhood fidelity.

The remaining work is empirical rather than additional method design: run the controlled pilot, choose one global $lambda_{mathrm{Merge}}$ on development data, and scale only after the success criteria in Section 8 are met.
