# Research Brief: Cross-Modal Topological Distillation for VLM2Vec

> **Core idea:** distill the persistent topology of the query–candidate retrieval relation, not only the geometry of separate image/text point clouds.

## 1. Motivation

The *Universal Geometry of Embeddings* paper suggests that embeddings from different architectures and dimensions can share a common latent geometric structure even when their coordinates are incompatible. Its vector-space-preservation objective is essential, indicating that **relational geometry is a key transferable signal**.

This connects naturally to VLM2Vec / VLM2Vec-V2, which provides a unified multimodal embedding space for retrieval and other vision-language tasks. However, a simple teacher-versus-student persistence-diagram loss would likely be too incremental because persistent-homology KD and multimodal topological alignment already exist.

## 2. Main Idea: Topology of the Cross-Modal Retrieval Relation

Let the teacher produce query and candidate embeddings:

$$Q_t = \{q^T_i \mid i = 1,\dots,B\}, \qquad C_t = \{c^T_j \mid j = 1,\dots,B\}$$

Define the teacher cross-modal distance matrix

$$D^T_{ij} = 1 - \cos(q^T_i, c^T_j)$$

and analogously $D^s_{ij}$ for the student. Construct a bipartite filtration

$$G^T_\varepsilon = (Q_t \cup C_t,\; E^T_\varepsilon), \qquad (q_i, c_j) \in E^T_\varepsilon \iff D^T_{ij} \le \varepsilon$$

**Interpretation:** the topology describes the *retrieval relation itself* — how queries and candidates connect across scales — rather than asking whether image and text clouds separately have similar shapes.

## 3. Proposed Objective

Start with cross-modal $H_0$ connectivity:

$$\mathcal{L}_{\text{CMTop},H_0} = W_2^2(D_T\_H_0,\; D^s\_H_0)$$

Optionally test lightweight $H_1$ **birth** structure rather than full $H_1$ persistence:

$$\mathcal{L}_{\text{CMTop}} = \mathcal{L}_{H_0} + \lambda_1 \mathcal{L}_{H_1\text{-birth}}$$

$$\mathcal{L} = \mathcal{L}_{\text{KD}} + \lambda_{\text{CMTop}} \mathcal{L}_{\text{CMTop}}$$

**Why $H_1$-birth only?** Full $H_1$ matching can be noisy and expensive. In a bipartite retrieval graph, $H_1$ births can encode many-to-many semantic structure, e.g., overlapping neighborhoods among several related queries and candidates.

## 4. Initial Experiment Plan

- Student only.
- Standard endpoint KD.
- Pairwise / VSP-style geometry KD.
- Ordinary point-cloud $H_0$ KD.
- Cross-modal relation $H_0$ (main proposal).
- Cross-modal relation $H_0$ + lightweight $H_1$-birth (optional extension).

**Evaluation:** MMEB / MMEB-V2 retrieval, topology discrepancy, neighborhood preservation, training cost, and multiple seeds. Use only final teacher embeddings if possible; this distinguishes the method from white-box approaches requiring hidden layers or attention maps.

## 5. Key Research Questions

| Question | Desired evidence |
| --- | --- |
| Does cross-modal topology help beyond endpoint KD? | Consistent retrieval gain across multiple teacher–student pairs. |
| Is relation topology better than ordinary point-cloud topology? | Cross-modal $H_0$ > separate image/text $H_0$. |
| Does the loss preserve the intended structure? | Lower topological discrepancy plus better neighborhood/retrieval preservation. |
| Can the method remain black-box? | Final teacher embeddings only; no teacher hidden states or attention maps. |

## 6. More Ambitious Extension: Universal Topology

Use multiple strong multimodal teachers $T_1,\dots,T_m$ and test whether their persistence structures share a common backbone. If so, compute a Wasserstein barycenter:

$$\bar{D}_k = \arg\min_D \sum_m w_m\, W_2^2\!\left(D,\; D_k^{\wedge}(T_m)\right)$$

$$\mathcal{L}_{\text{univ-top}} = W_2^2(D^s_k,\; \bar{D}_k)$$

**Research question:** Do strong multimodal embedding models share a persistent topological backbone that can be distilled into a much smaller student?

## 7. Target ICML Contribution

> For multimodal embedding models, the transferable structure is not only the geometry of each modality, but the **persistent topology of the cross-modal retrieval relation**.

**Recommended first step:** Implement cross-modal $H_0$ on VLM2Vec/MMEB, compare it against endpoint KD and ordinary point-cloud $H_0$, and verify that any performance gain is accompanied by better preservation of the teacher retrieval structure.
