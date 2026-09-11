"""Metrics for the CM-Merge evaluation protocol.

Section 4 of ``docs/cross_modal_topological_distillation.md`` asks for two
things beyond MMEB accuracy: a *topology discrepancy* between the teacher's and
the student's retrieval relation, and *neighborhood preservation*. A gain that
does not come with both is not evidence for the claim the paper wants to make.

The primary diagnostic is the labelled merge-matrix discrepancy, exactly the
structural object the main method targets.  Exact diagram distances remain as
controls (diagonal matches included), alongside full-pool neighborhood fidelity.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from src.topology import (
    bipartite_merge_matrix,
    bipartite_mst_edges,
    cosine_distance_matrix,
    h0_deaths,
    h1_births,
    point_cloud_h0_diagram,
    wasserstein2_diagram,
)


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _cross_modal_diagrams(dist_matrix, h1_topk=None):
    """H0 bars and H1 births of one relation, from a single MST.

    H0 and H1 are the tree and its complement, so they share one MST call.
    A B x B relation has ~B^2 H1 bars and the exact diagram distance is
    quadratic in the bar count, so the default keeps the same |Q| + |C|
    earliest births the training loss uses.
    """
    edges = bipartite_mst_edges(dist_matrix)
    topk = h1_topk if h1_topk else sum(dist_matrix.shape)
    return (
        h0_deaths(dist_matrix, edges).numpy(),
        h1_births(dist_matrix, edges, topk=topk).numpy(),
    )


def _component_labels(dist_matrix, threshold):
    """Connected-component labels of one bipartite threshold graph."""
    distances = dist_matrix.detach().cpu().numpy()
    n_q, n_c = distances.shape
    q_idx, c_idx = np.nonzero(distances <= threshold)
    rows = np.concatenate([q_idx, n_q + c_idx])
    cols = np.concatenate([n_q + c_idx, q_idx])
    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(n_q + n_c, n_q + n_c),
    )
    return connected_components(graph, directed=False, return_labels=True)[1]


def relation_topology_discrepancy(
    teacher_qry,
    teacher_cand,
    student_qry,
    student_cand,
    h1_topk=None,
    partition_quantiles=(0.01, 0.05, 0.1, 0.2),
):
    """Labelled merge discrepancy plus barcode controls for one batch.

    Args:
        teacher_qry/teacher_cand/student_qry/student_cand: Embedding arrays.
            Teacher/student rows must be aligned within each side. Query and
            candidate pool sizes may differ.
        h1_topk: how many earliest H1 births to compare; ``None`` uses
            ``|Q| + |C|``, matching the ``--cmtop_h1_topk`` training default.
        partition_quantiles: teacher-distance quantiles at which to compare the
            labelled component partitions with adjusted Rand index.

    Returns:
        The merge discrepancy, exact barcode controls, MST-edge recall,
        component agreement and ordinary point-cloud topology references.
    """
    t_q, t_c = _as_tensor(teacher_qry), _as_tensor(teacher_cand)
    s_q, s_c = _as_tensor(student_qry), _as_tensor(student_cand)
    if t_q.size(0) != s_q.size(0) or t_c.size(0) != s_c.size(0):
        raise ValueError(
            "teacher/student node labels must align within query and candidate "
            f"sides, got Q={t_q.size(0)}/{s_q.size(0)}, "
            f"C={t_c.size(0)}/{s_c.size(0)}"
        )
    if t_q.size(0) == 0 or t_c.size(0) == 0:
        raise ValueError("the relation needs at least one query and one candidate")
    if any(not 0.0 <= q <= 1.0 for q in partition_quantiles):
        raise ValueError("partition_quantiles must lie in [0, 1]")

    teacher_cross = cosine_distance_matrix(t_q, t_c)
    student_cross = cosine_distance_matrix(s_q, s_c)

    teacher_merge = bipartite_merge_matrix(teacher_cross)
    student_merge = bipartite_merge_matrix(student_cross)
    upper = torch.triu(
        torch.ones_like(teacher_merge, dtype=torch.bool), diagonal=1
    )

    teacher_h0, teacher_h1 = _cross_modal_diagrams(teacher_cross, h1_topk)
    student_h0, student_h1 = _cross_modal_diagrams(student_cross, h1_topk)
    teacher_edges = bipartite_mst_edges(teacher_cross)
    student_edges = bipartite_mst_edges(student_cross)
    n_c = teacher_cross.size(1)
    teacher_edge_ids = set(
        (teacher_edges[0] * n_c + teacher_edges[1]).cpu().tolist()
    )
    student_edge_ids = set(
        (student_edges[0] * n_c + student_edges[1]).cpu().tolist()
    )
    out = {
        "cross_modal_merge_l1": float(
            (teacher_merge - student_merge).abs()[upper].mean()
        ),
        "cross_modal_h0": wasserstein2_diagram(teacher_h0, student_h0),
        "cross_modal_h1_birth": wasserstein2_diagram(teacher_h1, student_h1),
        "teacher_mst_edge_recall": len(teacher_edge_ids & student_edge_ids)
        / max(len(teacher_edge_ids), 1),
    }

    teacher_values = teacher_cross.detach().cpu().numpy()
    for quantile in partition_quantiles:
        threshold = float(np.quantile(teacher_values, quantile))
        teacher_labels = _component_labels(teacher_cross, threshold)
        student_labels = _component_labels(student_cross, threshold)
        key = f"component_ari_q{int(round(100 * quantile)):02d}"
        out[key] = float(adjusted_rand_score(teacher_labels, student_labels))

    for name, teacher_points, student_points in (
        ("query_cloud_h0", t_q, s_q),
        ("candidate_cloud_h0", t_c, s_c),
        ("union_cloud_h0", torch.cat([t_q, t_c]), torch.cat([s_q, s_c])),
    ):
        out[name] = wasserstein2_diagram(
            point_cloud_h0_diagram(
                cosine_distance_matrix(teacher_points, teacher_points)
            ).numpy(),
            point_cloud_h0_diagram(
                cosine_distance_matrix(student_points, student_points)
            ).numpy(),
        )
    return out


def batched_topology_discrepancy(
    teacher_qry,
    teacher_cand,
    student_qry,
    student_cand,
    batch_size=64,
    num_batches=50,
    seed=0,
    h1_topk=None,
    partition_quantiles=(0.01, 0.05, 0.1, 0.2),
):
    """Average :func:`relation_topology_discrepancy` over random batches.

    The topology of a full 10k-example relation is neither cheap to compute nor
    the thing the loss shapes: training only ever sees a batch-sized filtration.
    Sampling batches of the training size measures the structure the loss
    actually acts on, and the spread over batches says how stable it is.
    """
    t_q, t_c = _as_tensor(teacher_qry), _as_tensor(teacher_cand)
    s_q, s_c = _as_tensor(student_qry), _as_tensor(student_cand)
    if t_q.size(0) != s_q.size(0) or t_c.size(0) != s_c.size(0):
        raise ValueError("teacher/student arrays must align within each modality")
    if num_batches < 1:
        raise ValueError("num_batches must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    q_batch_size = min(batch_size, t_q.size(0))
    c_batch_size = min(batch_size, t_c.size(0))

    rng = np.random.default_rng(seed)
    per_batch = []
    for _ in range(num_batches):
        q_idx = torch.as_tensor(
            rng.choice(t_q.size(0), size=q_batch_size, replace=False)
        )
        c_idx = torch.as_tensor(
            rng.choice(t_c.size(0), size=c_batch_size, replace=False)
        )
        per_batch.append(
            relation_topology_discrepancy(
                t_q[q_idx],
                t_c[c_idx],
                s_q[q_idx],
                s_c[c_idx],
                h1_topk=h1_topk,
                partition_quantiles=partition_quantiles,
            )
        )

    summary = {}
    for key in per_batch[0]:
        values = np.array([b[key] for b in per_batch], dtype=np.float64)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    summary["_config"] = {
        "query_batch_size": q_batch_size,
        "candidate_batch_size": c_batch_size,
        "num_batches": num_batches,
        "seed": seed,
        "partition_quantiles": list(partition_quantiles),
    }
    return summary


def _topk_indices(qry, cand, k, chunk=512):
    """Top-k candidates per query by cosine similarity, computed in chunks."""
    qry = torch.nn.functional.normalize(qry, p=2, dim=-1)
    cand = torch.nn.functional.normalize(cand, p=2, dim=-1)
    out = []
    for start in range(0, qry.size(0), chunk):
        scores = qry[start : start + chunk] @ cand.t()
        out.append(torch.topk(scores, k, dim=-1).indices)
    return torch.cat(out, dim=0)


def neighborhood_preservation(
    teacher_qry,
    teacher_cand,
    student_qry,
    student_cand,
    ks=(1, 5, 10, 50),
    rank_corr_queries=512,
    seed=0,
):
    """How much of the teacher's retrieval structure survives in the student.

    Two views of the same question, computed over the whole candidate pool:

    * ``recall@k``: fraction of the teacher's top-k candidates that are still in
      the student's top-k, averaged over queries. This is the neighborhood
      overlap the brief asks for -- it is measured against the *teacher*, not
      against ground truth, so it is a fidelity metric and not an accuracy one.
    * ``spearman``: rank correlation between the teacher's and the student's
      score vector for a query, averaged over a random subset of queries. Picks
      up reordering deep in the ranking that top-k overlap cannot see.
    """
    t_q, t_c = _as_tensor(teacher_qry), _as_tensor(teacher_cand)
    s_q, s_c = _as_tensor(student_qry), _as_tensor(student_cand)
    n_cand = t_c.size(0)

    out = {}
    for k in ks:
        if k > n_cand:
            continue
        teacher_top = _topk_indices(t_q, t_c, k)
        student_top = _topk_indices(s_q, s_c, k)
        overlap = [
            len(set(a.tolist()) & set(b.tolist())) / k
            for a, b in zip(teacher_top, student_top)
        ]
        out[f"recall@{k}"] = float(np.mean(overlap))

    rng = np.random.default_rng(seed)
    n_sample = min(rank_corr_queries, t_q.size(0))
    sample = rng.choice(t_q.size(0), size=n_sample, replace=False)
    t_qn = torch.nn.functional.normalize(t_q[sample], p=2, dim=-1)
    s_qn = torch.nn.functional.normalize(s_q[sample], p=2, dim=-1)
    t_cn = torch.nn.functional.normalize(t_c, p=2, dim=-1)
    s_cn = torch.nn.functional.normalize(s_c, p=2, dim=-1)

    correlations = []
    for i in range(n_sample):
        rho = spearmanr(
            (t_qn[i] @ t_cn.t()).numpy(), (s_qn[i] @ s_cn.t()).numpy()
        ).statistic
        if np.isfinite(rho):
            correlations.append(rho)
    out["spearman"] = float(np.mean(correlations)) if correlations else float("nan")
    out["_config"] = {
        "num_queries": int(t_q.size(0)),
        "num_candidates": int(n_cand),
        "rank_corr_queries": n_sample,
    }
    return out
