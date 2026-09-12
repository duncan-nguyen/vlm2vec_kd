# Research Brief: Correspondence-Aware Cross-Modal Topological Distillation

> **Working name:** CM-Merge
> **Core claim:** for multimodal retrieval, the transferable topological object is not an identity-agnostic persistence barcode, but the **correspondence-aware multiscale connectivity hierarchy of the query--candidate relation**.

## 1. Motivation

The *Universal Geometry of Embeddings* line of work suggests that models with different architectures and embedding dimensions can share relational structure even when their coordinates are incompatible. This makes relation-level distillation attractive for VLM2Vec / VLM2Vec-V2, whose tasks are all expressed as ranking a candidate set for an instructed multimodal query.

Existing distillation objectives transfer one of three weaker objects:

1. endpoint features, which require a projector and assume coordinate compatibility;
2. the full pairwise similarity matrix, which preserves all metric detail but does not isolate multiscale structure;
3. an ordinary persistence diagram of each modality or of a union point cloud, which does not directly represent the retrieval relation.

A natural first attempt is to construct a weighted bipartite graph from cross-modal distances and match its teacher and student persistence diagrams. However, **diagram-level matching is insufficient for retrieval** because a persistence diagram discards the identities of the queries and candidates participating in each topological event.

The proposed method therefore distills the complete identity-preserving $H_0$ connectivity evolution of the cross-modal relation. Here, identity means that teacher and student rows refer to the same query or candidate; it does **not** mean that CM-Merge consumes an additional class-label annotation. The method remains black-box and dimension-agnostic: only the final normalized teacher embeddings are required, and no teacher hidden states, attention maps, logits head, or learned teacher-to-student projector are used.

## 2. Why Identity-Agnostic Diagram Matching Is Not Enough

Let a batch contain indexed query and candidate vertices

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

For $H_0$, the finite persistence barcode is only the sorted multiset of the $n_q+n_c-1$ minimum-spanning-tree edge weights. It records **when some components merge**, but not **which query and candidate identities merge**.

### Candidate-permutation counterexample

Let $P$ be a non-identity permutation of the candidate columns and set

$$
D^S=D^TP.
$$

The two weighted bipartite graphs are isomorphic after permuting the candidate vertices. Their $H_0$ persistence diagrams, and the birth-only $H_1$ summaries of their 1-skeletons, are identical. An identity-agnostic diagram loss is therefore zero. Nevertheless, the correct candidate attached to each query has changed, so Precision@1 can fall from perfect retrieval to zero.

This exposes the central identifiability problem:

> A retrieval distillation objective must preserve both multiscale connectivity and the correspondence of the vertices that participate in that connectivity.

### Why birth-only $H_1$ is not the main solution

In a bipartite 1-skeleton there are no 2-simplices, so every cycle that is born persists forever. The non-MST edges are therefore all $H_1$ birth edges. Matching the earliest such births mainly matches low-order statistics of non-MST distances; it does not preserve cycle membership or finite persistence. Recent vision--language alignment work already uses MST/$H_0$-death and non-MST/$H_1$-birth edges, so this construction should be treated as a baseline rather than the main novelty.

## 3. Main Method: Correspondence-Aware Persistent Connectivity

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

The proposed loss compares corresponding entries of the identity-aligned merge matrices:

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

For one identity-aligned vertex pair,

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

Thus $\mathcal L_{\mathrm{Merge}}$ is the integrated disagreement between teacher and student identity-aligned connectivity partitions across every filtration scale. This is the topological object that the ordinary $H_0$ barcode compresses away.
