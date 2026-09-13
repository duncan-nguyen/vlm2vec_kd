"""Merge-hierarchy primitives for cross-modal relational distillation.

The research brief (``docs/cross_modal_topological_distillation.md``) distils the
connectivity of the *retrieval relation* rather than the shape of the image and
the text point clouds taken separately. For a batch of queries Q and candidates
C the filtration is the bipartite graph

    G_eps = (Q u C, {(q_i, c_j) : D_ij <= eps}),    D_ij = 1 - cos(q_i, c_j)

CM-Merge compares the *identity-preserving merge hierarchy* of the teacher's and
the student's ``G_eps``.  For vertices ``a`` and ``b`` its merge time is the
minimax distance

    U[a, b] = min_path(a -> b) max_edge_on_path D[edge].

Equivalently, ``U[a, b]`` is the smallest threshold at which the two indexed
vertices become connected.  It is read exactly from the bottleneck edge on the
unique path between them in an MST: components of ``G_eps`` merge exactly at the
edge weights of a minimum spanning tree of the (complete, hence always
connected) bipartite graph, so one MST call gives the whole hierarchy -- no
filtration sweep.

Unlike a sorted persistence diagram, the matrix retains which query and which
candidate participated in every merge and therefore cannot be fooled by
permuting candidate identities.

Gradients flow to the distances on the selected witness edges while the MST
combinatorics are treated as constant.  This is exact almost everywhere: the
MST and its path bottlenecks are piecewise constant away from distance ties.
"""

import numpy as np
import torch
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
    routine in floating point -- would otherwise put a negative "distance" into
    the merge hierarchy.
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
    """MST bottleneck edge witnessing every indexed pair's merge time.

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
            "an identity-preserving merge hierarchy needs at least one query and one candidate"
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
    """Exact identity-preserving merge-time matrix of a complete bipartite filtration.

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
