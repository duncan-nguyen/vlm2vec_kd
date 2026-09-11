"""Topology primitives for cross-modal relational distillation.

The research brief (``docs/cross_modal_topological_distillation.md``) distils the
persistent topology of the *retrieval relation* rather than the shape of the
image and the text point clouds taken separately. For a batch of queries Q and
candidates C the filtration is the bipartite graph

    G_eps = (Q u C, {(q_i, c_j) : D_ij <= eps}),    D_ij = 1 - cos(q_i, c_j)

The main CM-Merge objective compares the *labelled merge hierarchy* of the
teacher's and student's ``G_eps``.  For vertices ``a`` and ``b`` its merge time
is the minimax distance

    U[a, b] = min_path(a -> b) max_edge_on_path D[edge].

Equivalently, ``U[a, b]`` is the smallest threshold at which the two labelled
vertices become connected.  It is read exactly from the bottleneck edge on the
unique path between them in an MST.  Unlike a sorted persistence diagram, the
matrix retains which query/candidate participated in every merge and therefore
cannot be fooled by permuting candidate identities.

The older barcode primitives remain in this module as explicit ablation
baselines:

Two facts make that cheap enough to run inside a training step:

* **H0.** Components of G_eps merge exactly at the edge weights of a minimum
  spanning tree of the (complete, hence always connected) bipartite graph. Every
  vertex is born at eps = 0, so the finite part of the H0 barcode is precisely
  the sorted vector of MST edge weights -- V - 1 numbers from one MST call, no
  filtration sweep.
* **H1.** A bipartite graph has no triangles, so the flag complex over G_eps has
  no 2-simplices and no 1-cycle is ever filled in. Each of the E - V + 1 edges
  outside the MST opens one independent cycle when it enters, and that cycle
  never dies. "H1-birth only" is therefore not a cheap approximation of H1 here:
  it is the whole of H1, and it is the complement of the MST.

Gradients flow to the distances on the selected witness edges while the MST
combinatorics are treated as constant.  This is exact almost everywhere: the
MST and its path bottlenecks are piecewise constant away from distance ties.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import Tensor

# Edge weights are shifted above zero before being handed to scipy, whose sparse
# graph routines read an explicit 0 as "no edge". A positive constant shift
# leaves the MST untouched -- Kruskal only looks at the order of the weights --
# so the edge set that comes back is the one for the unshifted distances.
_SHIFT = 1.0


def cosine_distance_matrix(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """``1 - cos`` between every row of ``x`` and every row of ``y``.

    Returns a ``(len(x), len(y))`` matrix in ``[0, 2]``. Computed in float32:
    bf16 embeddings give distances whose spacing is far below the resolution the
    MST comparison needs. Clamped at 0 because a self-similarity of 1 + 1e-7 --
    routine in floating point -- would otherwise produce a negative "distance"
    and, through the diagonal cost in :func:`wasserstein2_diagram`, a negative
    contribution to a squared metric.
    """
    x = torch.nn.functional.normalize(x.float(), p=2, dim=-1, eps=eps)
    y = torch.nn.functional.normalize(y.float(), p=2, dim=-1, eps=eps)
    return (1.0 - x @ y.t()).clamp_min(0.0)


def _mst_edge_list(graph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices of the MST edges of a dense weighted adjacency block.

    ``graph`` holds strictly positive weights on the edges that exist and 0
    elsewhere; scipy treats it as undirected.
    """
    tree = minimum_spanning_tree(csr_matrix(graph)).tocoo()
    return tree.row, tree.col


def bipartite_mst_edges(dist_matrix: Tensor) -> tuple[Tensor, Tensor]:
    """MST of the complete bipartite graph K_{Bq,Bc} weighted by ``dist_matrix``.

    Args:
        dist_matrix: ``(Bq, Bc)`` cross-modal distances.

    Returns:
        ``(row, col)`` index tensors of length ``Bq + Bc - 1``, indexing
        ``dist_matrix`` directly. Not differentiable by construction.
    """
    n_q, n_c = dist_matrix.shape
    if n_q == 0 or n_c == 0:
        empty = torch.zeros(0, dtype=torch.long, device=dist_matrix.device)
        return empty, empty
    weights = dist_matrix.detach().float().cpu().numpy()

    graph = np.zeros((n_q + n_c, n_q + n_c), dtype=np.float64)
    graph[:n_q, n_q:] = weights - weights.min() + _SHIFT
    rows, cols = _mst_edge_list(graph)

    # Every edge joins one query node (index < n_q) to one candidate node, but
    # scipy is free to report either endpoint first.
    is_query_first = rows < n_q
    q_idx = np.where(is_query_first, rows, cols)
    c_idx = np.where(is_query_first, cols, rows) - n_q

    device = dist_matrix.device
    return (
        torch.as_tensor(q_idx, dtype=torch.long, device=device),
        torch.as_tensor(c_idx, dtype=torch.long, device=device),
    )


def bipartite_merge_witnesses(
    dist_matrix: Tensor,
    edges: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """MST bottleneck edge witnessing every labelled pair's merge time.

    The returned matrices have shape ``(Bq + Bc, Bq + Bc)``.  At an
    off-diagonal location ``(a, b)``, ``(query_idx[a,b], candidate_idx[a,b])``
    indexes the edge in ``dist_matrix`` at which vertices ``a`` and ``b`` first
    become connected.  Diagonal entries are ``-1``.

    Kruskal's algorithm gives all witnesses in one pass.  When an edge joins
    components A and B, that edge is the path bottleneck for every pair in
    ``A x B``.  The discrete witnesses are intentionally detached; gathering
    their distance values in :func:`bipartite_merge_matrix` remains fully
    differentiable.
    """
    if dist_matrix.ndim != 2:
        raise ValueError(
            f"dist_matrix must be 2-D, got shape {tuple(dist_matrix.shape)}"
        )
    n_q, n_c = dist_matrix.shape
    if n_q == 0 or n_c == 0:
        raise ValueError(
            "a labelled merge hierarchy needs at least one query and one candidate"
        )

    if edges is None:
        edges = bipartite_mst_edges(dist_matrix)
    q_edges, c_edges = edges
    expected = n_q + n_c - 1
    if q_edges.numel() != expected or c_edges.numel() != expected:
        raise ValueError(
            f"expected {expected} MST edges for K_{{{n_q},{n_c}}}, got "
            f"{q_edges.numel()} and {c_edges.numel()}"
        )

    # Kruskal needs the MST edges in non-decreasing weight order.  Do this on a
    # detached CPU copy because the result is discrete routing information.
    q_np = q_edges.detach().cpu().numpy().astype(np.int64, copy=False)
    c_np = c_edges.detach().cpu().numpy().astype(np.int64, copy=False)
    weights = dist_matrix.detach()[q_edges, c_edges].float().cpu().numpy()
    order = np.argsort(weights, kind="stable")

    n_vertices = n_q + n_c
    parent = np.arange(n_vertices, dtype=np.int64)
    size = np.ones(n_vertices, dtype=np.int64)
    members = [[i] for i in range(n_vertices)]
    witness_q = np.full((n_vertices, n_vertices), -1, dtype=np.int64)
    witness_c = np.full((n_vertices, n_vertices), -1, dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    for edge_pos in order:
        q_idx = int(q_np[edge_pos])
        c_idx = int(c_np[edge_pos])
        root_a = find(q_idx)
        root_b = find(n_q + c_idx)
        if root_a == root_b:
            # Defensive only: a valid MST never contains a cycle.
            continue

        a = np.asarray(members[root_a], dtype=np.int64)
        b = np.asarray(members[root_b], dtype=np.int64)
        witness_q[np.ix_(a, b)] = q_idx
        witness_q[np.ix_(b, a)] = q_idx
        witness_c[np.ix_(a, b)] = c_idx
        witness_c[np.ix_(b, a)] = c_idx

        # Union by size keeps `find` shallow; component member lists let this
        # pass assign all O(V^2) output entries in O(V^2) total work.
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]
        members[root_a].extend(members[root_b])
        members[root_b] = []

    off_diagonal = ~np.eye(n_vertices, dtype=bool)
    if np.any(witness_q[off_diagonal] < 0):
        raise RuntimeError("MST did not connect every vertex in the bipartite graph")

    device = dist_matrix.device
    return (
        torch.as_tensor(witness_q, dtype=torch.long, device=device),
        torch.as_tensor(witness_c, dtype=torch.long, device=device),
    )


def bipartite_merge_matrix(
    dist_matrix: Tensor,
    edges: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Exact labelled merge-time matrix of a complete bipartite filtration.

    The diagonal is zero.  Every off-diagonal entry is gathered from its MST
    bottleneck witness, so gradients flow to precisely the distances that
    support the current hierarchy.
    """
    witness_q, witness_c = bipartite_merge_witnesses(dist_matrix, edges)
    valid = witness_q >= 0
    values = dist_matrix[
        witness_q.clamp_min(0),
        witness_c.clamp_min(0),
    ]
    return torch.where(valid, values, torch.zeros_like(values))


def point_cloud_mst_edges(dist_matrix: Tensor) -> tuple[Tensor, Tensor]:
    """MST of the complete graph on one point cloud.

    Args:
        dist_matrix: ``(N, N)`` symmetric distances; only the strict upper
            triangle is read.

    Returns:
        ``(row, col)`` index tensors of length ``N - 1``.
    """
    n = dist_matrix.size(0)
    if n < 2:
        # One point has no MST edge, and `weights[upper].min()` below would be
        # taken over an empty selection.
        empty = torch.zeros(0, dtype=torch.long, device=dist_matrix.device)
        return empty, empty
    weights = dist_matrix.detach().float().cpu().numpy()

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    graph = np.zeros((n, n), dtype=np.float64)
    graph[upper] = weights[upper] - weights[upper].min() + _SHIFT
    rows, cols = _mst_edge_list(graph)

    device = dist_matrix.device
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(cols, dtype=torch.long, device=device),
    )


def h0_deaths(dist_matrix: Tensor, edges: tuple[Tensor, Tensor]) -> Tensor:
    """Finite H0 bars of the filtration, sorted ascending.

    All vertices are born at 0, so a bar is fully described by its death time,
    and the deaths are the MST edge weights. Differentiable in ``dist_matrix``.
    """
    rows, cols = edges
    return torch.sort(dist_matrix[rows, cols]).values


def h1_births(
    dist_matrix: Tensor,
    edges: tuple[Tensor, Tensor],
    topk: int | None = None,
) -> Tensor:
    """Earliest H1 birth times of the bipartite flag complex, sorted ascending.

    The complement of the MST is exactly the set of cycle-creating edges (see
    the module docstring), so the births are those weights. ``topk`` keeps only
    the earliest ones -- the late births are the far corners of the batch and
    carry little retrieval structure. ``None`` keeps all ``E - V + 1`` of them.
    """
    rows, cols = edges
    n_edges = dist_matrix.numel()
    n_cycles = n_edges - rows.numel()
    if n_cycles <= 0:
        return dist_matrix.new_zeros(0)

    in_tree = torch.zeros_like(dist_matrix, dtype=torch.bool)
    in_tree[rows, cols] = True
    # masked_fill is out-of-place, so the tree entries simply receive no
    # gradient instead of corrupting the graph with an in-place write.
    candidates = dist_matrix.masked_fill(in_tree, float("inf")).reshape(-1)

    k = n_cycles if topk is None else min(int(topk), n_cycles)
    births = torch.topk(candidates, k, largest=False).values
    return torch.sort(births).values


def wasserstein2_sorted(a: Tensor, b: Tensor, reduction: str = "mean") -> Tensor:
    """Squared 2-Wasserstein between two equal-size birth-0 diagrams.

    With every point on the ``birth = 0`` line the diagram distance collapses to
    1-D optimal transport on the death coordinate, whose optimum is the
    sorted-to-sorted matching. This restricts matchings to bijections and so
    ignores the diagonal, which is optimal whenever no pair is cheaper to kill
    against the diagonal than to match -- the regime a distillation loss runs in,
    since both diagrams have the same cardinality and comparable scale.
    :func:`wasserstein2_diagram` computes the exact value including diagonal
    matches; it is what evaluation reports.

    ``reduction="mean"`` divides by the number of bars so that the weight on the
    loss does not have to be retuned when the batch size changes;
    ``reduction="sum"`` is the textbook W_2^2.
    """
    if a.numel() != b.numel():
        raise ValueError(
            f"diagrams must have the same number of bars for a bijective "
            f"matching, got {a.numel()} and {b.numel()}"
        )
    if a.numel() == 0:
        return a.new_zeros(())
    sq = (a - b) ** 2
    return sq.mean() if reduction == "mean" else sq.sum()


def wasserstein2_diagram(
    deaths_a: np.ndarray,
    deaths_b: np.ndarray,
    reduction: str = "sum",
    max_bars: int = 8192,
) -> float:
    """Exact squared W_2 between two birth-0 persistence diagrams.

    Unlike :func:`wasserstein2_sorted` this allows points to be matched to the
    diagonal, so it also handles diagrams of different cardinality. Ground
    metric is Euclidean on the diagram: matching ``(0, d1)`` to ``(0, d2)`` costs
    ``(d1 - d2)^2`` and sending ``(0, d)`` to the diagonal costs ``d^2 / 2``.
    Solved as a linear assignment on the standard augmented cost matrix, so this
    is a reporting metric (numpy, no gradient), not a training loss. The
    augmented matrix is ``(n + m)^2``, which is why ``max_bars`` refuses inputs
    that would quietly try to allocate tens of gigabytes -- an H1 diagram of a
    B x B relation has ~B^2 bars, so cut it down with the ``topk`` of
    :func:`h1_births` before asking for an exact distance.
    """
    a = np.asarray(deaths_a, dtype=np.float64).ravel()
    b = np.asarray(deaths_b, dtype=np.float64).ravel()
    n, m = a.size, b.size
    if n == 0 and m == 0:
        return 0.0
    if n + m > max_bars:
        raise ValueError(
            f"exact diagram W2 needs a ({n + m})^2 cost matrix "
            f"({(n + m) ** 2 * 8 / 2**30:.1f} GiB); pass fewer bars (see topk in "
            f"h1_births) or raise max_bars deliberately"
        )

    diag_a = a**2 / 2.0
    diag_b = b**2 / 2.0
    # Forbidden entries need a finite stand-in: linear_sum_assignment rejects inf.
    # Any value above the cost of every admissible full matching works.
    worst_pair = max(
        (a.max() - b.min()) ** 2 if n and m else 0.0,
        (b.max() - a.min()) ** 2 if n and m else 0.0,
        diag_a.max() if n else 0.0,
        diag_b.max() if m else 0.0,
    )
    blocked = (worst_pair + 1.0) * (n + m + 1)

    cost = np.empty((n + m, n + m), dtype=np.float64)
    cost[:n, :m] = np.subtract.outer(a, b) ** 2
    cost[:n, m:] = blocked
    cost[:n, m:][np.arange(n), np.arange(n)] = diag_a
    cost[n:, :m] = blocked
    cost[n:, :m][np.arange(m), np.arange(m)] = diag_b
    cost[n:, m:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    total = float(cost[rows, cols].sum())
    if reduction == "mean":
        return total / max(n + m, 1)
    return total


def cross_modal_h0_diagram(dist_matrix: Tensor) -> Tensor:
    """Convenience wrapper: finite H0 bars of a cross-modal distance matrix."""
    return h0_deaths(dist_matrix, bipartite_mst_edges(dist_matrix))


def point_cloud_h0_diagram(dist_matrix: Tensor) -> Tensor:
    """Convenience wrapper: finite H0 bars of one point cloud."""
    return h0_deaths(dist_matrix, point_cloud_mst_edges(dist_matrix))
