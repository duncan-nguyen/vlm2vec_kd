"""Self-checks for CM-Merge and its persistence baselines.

Run from the repo root: `python tools/misc/test_cmtop.py`. Needs only torch,
numpy and scipy -- no model download, no GPU.

Each check re-derives a quantity from its definition (a filtration sweep, a
union-find Kruskal, a brute-force enumeration of diagram matchings) and compares
it against the fast path the training loss actually takes.
"""

import itertools
import os as _os
import sys as _sys

_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import numpy as np
import torch

from src.topology import (
    bipartite_merge_matrix,
    bipartite_merge_witnesses,
    bipartite_mst_edges,
    cosine_distance_matrix,
    h0_deaths,
    h1_births,
    point_cloud_mst_edges,
    wasserstein2_diagram,
    wasserstein2_sorted,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def kruskal_sweep(n_nodes, edges):
    """Reference Kruskal: returns (merge heights, cycle birth times).

    `edges` is a list of (weight, u, v). Sweeping the edges in weight order is
    the filtration itself: an edge that joins two components kills an H0 class at
    its weight, an edge that does not creates an independent 1-cycle.
    """
    uf = UnionFind(n_nodes)
    deaths, births = [], []
    for weight, u, v in sorted(edges):
        (deaths if uf.union(u, v) else births).append(weight)
    return deaths, births


def bipartite_edge_list(dist):
    n_q, n_c = dist.shape
    return [(float(dist[i, j]), i, n_q + j) for i in range(n_q) for j in range(n_c)]


def full_edge_list(dist):
    n = dist.shape[0]
    return [(float(dist[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]


def minimax_connectivity_reference(dist):
    """Definition-level all-pairs minimax paths (Floyd-Warshall)."""
    dist = np.asarray(dist, dtype=np.float64)
    n_q, n_c = dist.shape
    n = n_q + n_c
    merge = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(merge, 0.0)
    merge[:n_q, n_q:] = dist
    merge[n_q:, :n_q] = dist.T
    for k in range(n):
        merge = np.minimum(merge, np.maximum(merge[:, k, None], merge[k, None, :]))
    return merge


def brute_force_w2(a, b):
    """Exact squared W_2 by enumerating every partial matching. Tiny inputs only."""
    a, b = list(a), list(b)
    best = float("inf")
    for k in range(min(len(a), len(b)) + 1):
        for a_sub in itertools.combinations(range(len(a)), k):
            for b_sub in itertools.permutations(range(len(b)), k):
                cost = sum((a[i] - b[j]) ** 2 for i, j in zip(a_sub, b_sub))
                cost += sum(a[i] ** 2 / 2 for i in range(len(a)) if i not in a_sub)
                cost += sum(b[j] ** 2 / 2 for j in range(len(b)) if j not in b_sub)
                best = min(best, cost)
    return best


def load_criterion_module():
    """Import the CMTop criterion without going through `src.criterions`.

    That package's `__init__` pulls in spacy/numba/tslearn for the other
    criteria, none of which this check needs.
    """
    import importlib.util

    path = _os.path.join(
        _os.path.dirname(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        ),
        "src",
        "criterions",
        "cross_modal_topology.py",
    )
    spec = importlib.util.spec_from_file_location("cmtop_criterion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEncoder(torch.nn.Module):
    """Stands in for MMEBModel: a linear map onto a fixed batch of inputs."""

    def __init__(self, in_dim, out_dim, trainable):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim)
        for p in self.parameters():
            p.requires_grad_(trainable)

    def encode_input(self, batch):
        return self.proj(batch["x"]), None, None, None

    def compute_similarity(self, q, p):
        return q @ p.t()

    def eval(self):
        return self


class _FakeDistiller:
    """Mirrors the parts of `Distiller` a criterion touches.

    `encode_teacher` reproduces the real dispatch: cache first, live model
    otherwise. `tools/misc/test_teacher_cache.py` covers the cached branch.
    """

    def __init__(self, student, teacher, projectors=None, teacher_cache=None):
        self.student = student
        self.teacher = teacher
        self.temperature = 0.02
        self.projectors = projectors
        self.teacher_cache = teacher_cache

    def encode_teacher(self, input_data, side, dtype=None):
        if self.teacher_cache is not None:
            qry, pos = self.teacher_cache.get(
                input_data["sample_ids"],
                device=next(self.student.parameters()).device,
                dtype=dtype,
            )
            return qry if side == "qry" else pos
        with torch.no_grad():
            self.teacher.eval()
            output = self.teacher.encode_input(input_data["teacher_inputs"][side])
        return output[0] if isinstance(output, (tuple, list)) else output


def _args(**overrides):
    from types import SimpleNamespace

    base = dict(
        kd_weight=0.0,
        cmtop_weight=1.0,
        cmtop_mode="merge",
        cmtop_h0_weight=1.0,
        cmtop_h1_weight=0.0,
        cmtop_h1_topk=0,
        cmtop_endpoint_kd="none",
        cmtop_geometry_weight=0.0,
        cmtop_normalize_scale=False,
        cmtop_reduction="mean",
        cmtop_merge_block="all",
        cmtop_task_homogeneous=True,
        cmtop_deduplicate_candidates=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def criterion_checks():
    """End-to-end checks on CrossModalTopologyLoss with stand-in encoders."""
    print("\n--- criterion ---")
    module = load_criterion_module()
    torch.manual_seed(0)

    batch, in_dim, student_dim, teacher_dim = 12, 24, 16, 32
    student = _FakeEncoder(in_dim, student_dim, trainable=True)
    teacher = _FakeEncoder(in_dim, teacher_dim, trainable=False)
    projectors = torch.nn.ModuleDict(
        {"t2s": torch.nn.Sequential(torch.nn.Linear(teacher_dim, student_dim))}
    )
    distiller = _FakeDistiller(student, teacher, projectors)
    inputs = {
        "task_ids": torch.zeros(batch, dtype=torch.long),
        "candidate_ids": torch.arange(batch, dtype=torch.long),
        "student_inputs": {
            "qry": {"x": torch.randn(batch, in_dim)},
            "pos": {"x": torch.randn(batch, in_dim)},
        },
        "teacher_inputs": {
            "qry": {"x": torch.randn(batch, in_dim)},
            "pos": {"x": torch.randn(batch, in_dim)},
        },
    }

    for mode in (
        "merge",
        "critical_edges",
        "cross_modal",
        "point_cloud",
        "union",
        "merge+point_cloud",
    ):
        h1 = 0.1 if "cross_modal" in mode else 0.0
        out = module.CrossModalTopologyLoss(_args(cmtop_mode=mode, cmtop_h1_weight=h1))(
            distiller, inputs
        )
        finite = all(torch.isfinite(v).all() for v in out.values())
        check(f"criterion runs and stays finite (mode={mode})", finite)
        check(
            f"H1 term is active exactly when asked for (mode={mode})",
            (float(out["cmtop_h1_loss"].detach()) > 0) == (h1 > 0),
        )

    # H1 is only defined on the bipartite relation, so asking for it elsewhere
    # must fail rather than quietly contribute nothing.
    check(
        "H1 weight without a cross-modal mode is rejected",
        _raises(
            lambda: module.CrossModalTopologyLoss(
                _args(cmtop_mode="point_cloud", cmtop_h1_weight=0.1)
            ),
            ValueError,
        ),
    )
    check(
        "unknown --cmtop_reduction is rejected",
        _raises(lambda: module.CrossModalTopologyLoss(_args(cmtop_reduction="max"))),
    )

    check(
        "unknown --cmtop_mode is rejected",
        _raises(lambda: module.CrossModalTopologyLoss(_args(cmtop_mode="rips"))),
    )
    check(
        "unknown --cmtop_endpoint_kd is rejected",
        _raises(lambda: module.CrossModalTopologyLoss(_args(cmtop_endpoint_kd="l1"))),
    )

    # Missing projector with mismatched dims must fail loudly, not silently skip KD.
    check(
        "endpoint KD without a projector reports the dimension mismatch",
        _raises(
            lambda: module.CrossModalTopologyLoss(
                _args(
                    kd_weight=1.0,
                    cmtop_weight=0.0,
                    cmtop_endpoint_kd="cosine",
                )
            )(
                # Endpoint KD, unlike CM-Merge, needs the dimensional projector.
                _FakeDistiller(student, teacher, projectors=None), inputs
            ),
            ValueError,
        ),
    )

    criterion = module.CrossModalTopologyLoss(
        _args(cmtop_mode="merge", cmtop_geometry_weight=0.5)
    )
    out = criterion(distiller, inputs)
    student.zero_grad()
    out["loss"].backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    check(
        "student receives a finite non-zero gradient from the full objective",
        bool(grads)
        and all(torch.isfinite(g).all() for g in grads)
        and sum(float(g.abs().sum()) for g in grads) > 0,
    )
    check(
        "teacher stays frozen",
        all(p.grad is None for p in teacher.parameters()),
    )

    # Every ablation row of the brief has to be reachable from the flags.
    student_only = module.CrossModalTopologyLoss(
        _args(kd_weight=0.0, cmtop_weight=0.0, cmtop_endpoint_kd="none")
    )(distiller, inputs)
    check(
        "student-only flags leave the contrastive loss alone",
        torch.allclose(student_only["loss"], student_only["contrastive_loss"])
        and float(student_only["kd_loss"].detach()) == 0.0,
    )
    endpoint_only = module.CrossModalTopologyLoss(
        _args(kd_weight=1.0, cmtop_weight=0.0, cmtop_endpoint_kd="cosine")
    )(distiller, inputs)
    check(
        "endpoint-KD flags switch the topology term off",
        float(endpoint_only["cmtop_loss"].detach()) == 0.0
        and float(endpoint_only["endpoint_loss"].detach()) > 0.0,
    )

    identical = {
        "task_ids": inputs["task_ids"],
        "candidate_ids": inputs["candidate_ids"],
        "student_inputs": inputs["teacher_inputs"],
        "teacher_inputs": inputs["teacher_inputs"],
    }
    same_model = _FakeDistiller(teacher, teacher, projectors)
    out = module.CrossModalTopologyLoss(
        _args(kd_weight=0.0, cmtop_mode="merge", cmtop_endpoint_kd="none")
    )(same_model, identical)
    check(
        "distilling a model from itself gives zero topological loss",
        float(out["cmtop_loss"].detach()) == 0.0,
        f"{float(out['cmtop_loss'].detach()):.2e}",
    )

    duplicate_inputs = dict(inputs)
    duplicate_inputs["candidate_ids"] = torch.tensor(
        [0, 0] + list(range(2, batch)), dtype=torch.long
    )
    duplicate_out = module.CrossModalTopologyLoss(_args())(distiller, duplicate_inputs)
    check(
        "candidate identity dedup keeps every query but one canonical candidate",
        int(duplicate_out["num_unique_candidates"]) == batch - 1,
    )

    mixed_inputs = dict(inputs)
    mixed_inputs["task_ids"] = torch.tensor([0] * (batch - 1) + [1])
    check(
        "a mixed-task relation graph is rejected",
        _raises(
            lambda: module.CrossModalTopologyLoss(_args())(distiller, mixed_inputs),
            ValueError,
        ),
    )
    print("--- /criterion ---\n")


def _raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def main():
    torch.manual_seed(0)

    # ---------------------------------------------------------------- bipartite
    n_q, n_c, dim = 9, 7, 16
    q = torch.randn(n_q, dim, dtype=torch.float64)
    c = torch.randn(n_c, dim, dtype=torch.float64)
    dist = cosine_distance_matrix(q, c)

    rows, cols = bipartite_mst_edges(dist)
    check(
        "bipartite MST has V-1 edges",
        rows.numel() == n_q + n_c - 1,
        f"{rows.numel()} vs {n_q + n_c - 1}",
    )
    check(
        "bipartite MST indices stay in range",
        bool(
            (rows >= 0).all()
            and (rows < n_q).all()
            and (cols >= 0).all()
            and (cols < n_c).all()
        ),
    )
    check(
        "bipartite MST touches every node",
        len(set(rows.tolist())) == n_q and len(set(cols.tolist())) == n_c,
    )

    merge = bipartite_merge_matrix(dist)
    merge_reference = minimax_connectivity_reference(dist.numpy())
    check(
        "identity-preserving merge matrix equals definition-level minimax paths",
        np.allclose(merge.numpy(), merge_reference),
        f"max diff {np.abs(merge.numpy() - merge_reference).max():.2e}",
    )
    witness_q, witness_c = bipartite_merge_witnesses(dist, (rows, cols))
    off_diagonal = ~torch.eye(n_q + n_c, dtype=torch.bool)
    check(
        "every off-diagonal indexed pair has a valid MST witness",
        bool((witness_q[off_diagonal] >= 0).all())
        and bool((witness_c[off_diagonal] >= 0).all()),
    )
    check(
        "merge matrix is symmetric with a zero diagonal",
        torch.allclose(merge, merge.t())
        and torch.count_nonzero(torch.diag(merge)) == 0,
    )

    perturbation = torch.empty_like(dist).uniform_(-1e-3, 1e-3)
    perturbed_merge = bipartite_merge_matrix(dist + perturbation)
    check(
        "merge hierarchy obeys the L-infinity stability bound",
        float((merge - perturbed_merge).abs().max())
        <= float(perturbation.abs().max()) + 1e-7,
    )

    ref_deaths, ref_births = kruskal_sweep(n_q + n_c, bipartite_edge_list(dist))
    fast_deaths = h0_deaths(dist, (rows, cols))
    check(
        "H0 deaths match a filtration sweep (bipartite)",
        np.allclose(fast_deaths.numpy(), np.sort(ref_deaths)),
        f"max diff {np.abs(fast_deaths.numpy() - np.sort(ref_deaths)).max():.2e}",
    )

    fast_births = h1_births(dist, (rows, cols), topk=None)
    check(
        "H1 births are the non-tree edges, all of them",
        fast_births.numel() == n_q * n_c - (n_q + n_c - 1) == len(ref_births),
        f"{fast_births.numel()} vs {len(ref_births)}",
    )
    check(
        "H1 births match a filtration sweep",
        np.allclose(fast_births.numpy(), np.sort(ref_births)),
    )
    check(
        "H1 topk keeps the earliest births",
        np.allclose(
            h1_births(dist, (rows, cols), topk=5).numpy(), np.sort(ref_births)[:5]
        ),
    )

    # -------------------------------------------------------------- point cloud
    x = torch.randn(11, dim, dtype=torch.float64)
    pc_dist = cosine_distance_matrix(x, x)
    pc_edges = point_cloud_mst_edges(pc_dist)
    pc_ref_deaths, _ = kruskal_sweep(11, full_edge_list(pc_dist))
    check(
        "point-cloud MST has N-1 edges",
        pc_edges[0].numel() == 10,
        f"{pc_edges[0].numel()}",
    )
    check(
        "point-cloud H0 deaths match a filtration sweep",
        np.allclose(h0_deaths(pc_dist, pc_edges).numpy(), np.sort(pc_ref_deaths)),
    )

    # ------------------------------------------------- degenerate / zero weights
    # scipy drops explicit zeros, so duplicated embeddings (distance exactly 0)
    # are the case the constant shift in src/topology.py exists for.
    dup = torch.cat([q[:3], q[:3]], dim=0)
    dup_dist = cosine_distance_matrix(dup, c)
    dup_rows, dup_cols = bipartite_mst_edges(dup_dist)
    dup_ref, _ = kruskal_sweep(dup.size(0) + n_c, bipartite_edge_list(dup_dist))
    check(
        "zero-distance duplicates keep every MST edge",
        dup_rows.numel() == dup.size(0) + n_c - 1,
        f"{dup_rows.numel()} vs {dup.size(0) + n_c - 1}",
    )
    check(
        "H0 deaths still match with zero-weight edges present",
        np.allclose(
            h0_deaths(dup_dist, (dup_rows, dup_cols)).numpy(), np.sort(dup_ref)
        ),
    )
    check(
        "identity-preserving merge times stay exact with duplicate zero-distance rows",
        np.allclose(
            bipartite_merge_matrix(dup_dist).numpy(),
            minimax_connectivity_reference(dup_dist.numpy()),
        ),
    )

    # ------------------------------------------------------- degenerate shapes
    # A short final batch must not crash the MST helpers.
    single = cosine_distance_matrix(q[:1], c[:1])
    check(
        "1x1 relation gives a single H0 bar and no cycle",
        h0_deaths(single, bipartite_mst_edges(single)).numel() == 1
        and h1_births(single, bipartite_mst_edges(single)).numel() == 0,
    )
    check(
        "1x1 relation has the expected identity-preserving merge matrix",
        torch.allclose(
            bipartite_merge_matrix(single),
            torch.tensor(
                [[0.0, float(single[0, 0])], [float(single[0, 0]), 0.0]],
                dtype=single.dtype,
            ),
        ),
    )
    lone = cosine_distance_matrix(q[:1], q[:1])
    check(
        "a one-point cloud has an empty MST instead of raising",
        point_cloud_mst_edges(lone)[0].numel() == 0
        and float(
            wasserstein2_sorted(
                h0_deaths(lone, point_cloud_mst_edges(lone)),
                h0_deaths(lone, point_cloud_mst_edges(lone)),
            )
        )
        == 0.0,
    )
    check(
        "cosine distance never goes negative on identical vectors",
        float(cosine_distance_matrix(q, q).min()) >= 0.0,
    )

    # ------------------------------------------------------------- Wasserstein
    a = torch.tensor([0.1, 0.4, 0.9])
    check("W2 of a diagram with itself is 0", float(wasserstein2_sorted(a, a)) == 0.0)
    check(
        "W2 sorted == mean of squared sorted differences",
        np.isclose(
            float(wasserstein2_sorted(a, torch.tensor([0.2, 0.4, 0.7]))),
            np.mean([0.01, 0.0, 0.04]),
        ),
    )
    for trial in range(20):
        rng = np.random.default_rng(trial)
        u = rng.random(3) * 2
        v = rng.random(rng.integers(1, 4)) * 2
        exact = wasserstein2_diagram(u, v)
        brute = brute_force_w2(u, v)
        if not np.isclose(exact, brute):
            check(
                "exact diagram W2 matches brute-force matching",
                False,
                f"{exact} vs {brute}",
            )
            break
    else:
        check("exact diagram W2 matches brute-force matching", True, "20 random pairs")

    check(
        "exact diagram W2 uses the diagonal when it is cheaper",
        np.isclose(wasserstein2_diagram([0.0, 2.0], [0.0, 0.0]), 2.0),
        f"{wasserstein2_diagram([0.0, 2.0], [0.0, 0.0])}",
    )

    # -------------------------------------------------------------- gradients
    student = torch.randn(n_q, dim, requires_grad=True)
    teacher_diagram = h0_deaths(dist.float(), (rows, cols)).detach()

    def topo_loss(emb):
        d = cosine_distance_matrix(emb, c.float())
        return wasserstein2_sorted(
            h0_deaths(d, bipartite_mst_edges(d)), teacher_diagram
        )

    start = topo_loss(student)
    start.backward()
    start = float(start.detach())
    check(
        "topology loss has a non-zero gradient w.r.t. the student",
        student.grad is not None
        and torch.isfinite(student.grad).all()
        and student.grad.abs().sum() > 0,
    )

    opt = torch.optim.Adam([student], lr=5e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = topo_loss(student)
        loss.backward()
        opt.step()
    check(
        "gradient descent drives the topology loss down",
        float(loss.detach()) < 0.05 * start,
        f"{start:.3e} -> {float(loss.detach()):.3e}",
    )

    merge_student = torch.randn(n_q, dim, requires_grad=True)
    merge_target = bipartite_merge_matrix(dist.float()).detach()
    merge_loss = (
        bipartite_merge_matrix(cosine_distance_matrix(merge_student, c.float()))
        - merge_target
    ).abs().mean()
    merge_loss.backward()
    check(
        "CM-Merge routes a finite non-zero gradient through witness edges",
        merge_student.grad is not None
        and torch.isfinite(merge_student.grad).all()
        and merge_student.grad.abs().sum() > 0,
    )

    criterion_checks()

    # -------------------------------------------------------------- invariance
    perm_q = torch.randperm(n_q)
    perm_c = torch.randperm(n_c)
    permuted = cosine_distance_matrix(q[perm_q], c[perm_c])
    check(
        "H0 barcode is invariant to permuting batch identities",
        np.allclose(
            h0_deaths(permuted, bipartite_mst_edges(permuted)).numpy(),
            fast_deaths.numpy(),
        ),
    )
    vertex_perm = torch.cat([perm_q, n_q + perm_c])
    check(
        "merge hierarchy is equivariant to simultaneous identity permutation",
        torch.allclose(
            bipartite_merge_matrix(permuted),
            merge[vertex_perm][:, vertex_perm],
        ),
    )
    candidate_permuted = dist[:, perm_c]
    candidate_permuted_h0 = h0_deaths(
        candidate_permuted, bipartite_mst_edges(candidate_permuted)
    )
    check(
        "candidate permutation exposes the barcode's identity blindness",
        torch.allclose(candidate_permuted_h0, fast_deaths)
        and not torch.allclose(bipartite_merge_matrix(candidate_permuted), merge),
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
