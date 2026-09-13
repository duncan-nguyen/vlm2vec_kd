"""Self-checks for Ours.

Run from the repo root: `python tools/misc/test_ours.py`. Needs only torch,
numpy and scipy -- no model download, no GPU.

Each check re-derives a quantity from its definition (all-pairs minimax paths by
Floyd-Warshall) and compares it against the fast MST path the training loss
actually takes.
"""

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
)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


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


def load_criterion_module():
    """Import the Ours criterion without going through `src.criterions`.

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
    spec = importlib.util.spec_from_file_location("ours_criterion", path)
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
        ours_weight=1.0,
        ours_retrieval_loss=True,
        ours_reduction="mean",
        ours_merge_block="all",
        ours_task_homogeneous=True,
        ours_deduplicate_candidates=True,
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
    distiller = _FakeDistiller(student, teacher)
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

    for block in ("all", "cross"):
        out = module.CrossModalTopologyLoss(_args(ours_merge_block=block))(
            distiller, inputs
        )
        check(
            f"criterion runs and stays finite (merge_block={block})",
            all(torch.isfinite(v).all() for v in out.values())
            and float(out["topo_loss"].detach()) > 0,
        )

    check(
        "unknown --ours_reduction is rejected",
        _raises(lambda: module.CrossModalTopologyLoss(_args(ours_reduction="max"))),
    )
    check(
        "unknown --ours_merge_block is rejected",
        _raises(lambda: module.CrossModalTopologyLoss(_args(ours_merge_block="tri"))),
    )

    # Ours needs no teacher/student projector: it only ever compares
    # distances, which are dimension-free. A missing projector must not matter.
    check(
        "Ours runs without any projector",
        float(
            module.CrossModalTopologyLoss(_args())(
                _FakeDistiller(student, teacher, projectors=None), inputs
            )["topo_loss"].detach()
        )
        > 0,
    )

    out = module.CrossModalTopologyLoss(_args())(distiller, inputs)
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

    # The no-teacher control has to be reachable from the flags alone, so that
    # it runs under exactly this sampler and batch construction.
    student_only = module.CrossModalTopologyLoss(_args(ours_weight=0.0))(
        distiller, inputs
    )
    check(
        "--ours_weight 0 leaves the contrastive loss alone",
        torch.allclose(student_only["loss"], student_only["contrastive_loss"])
        and float(student_only["kd_loss"].detach()) == 0.0,
    )

    topo_only = module.CrossModalTopologyLoss(_args(ours_retrieval_loss=False))(
        distiller, inputs
    )
    check(
        "--ours_retrieval_loss False trains on the L_topo term alone",
        torch.allclose(topo_only["loss"], topo_only["kd_loss"])
        and float(topo_only["contrastive_loss"].detach()) > 0,
    )
    check(
        "dropping both loss terms is rejected",
        _raises(
            lambda: module.CrossModalTopologyLoss(
                _args(ours_retrieval_loss=False, ours_weight=0.0)
            )
        ),
    )

    identical = {
        "task_ids": inputs["task_ids"],
        "candidate_ids": inputs["candidate_ids"],
        "student_inputs": inputs["teacher_inputs"],
        "teacher_inputs": inputs["teacher_inputs"],
    }
    out = module.CrossModalTopologyLoss(_args())(
        _FakeDistiller(teacher, teacher), identical
    )
    check(
        "distilling a model from itself gives zero Ours loss",
        float(out["topo_loss"].detach()) == 0.0,
        f"{float(out['topo_loss'].detach()):.2e}",
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

    # A batch with no task ids cannot be validated, so it must fail loudly
    # rather than silently skipping the homogeneity guarantee.
    no_metadata = {k: v for k, v in inputs.items() if k != "task_ids"}
    check(
        "a batch without task ids is rejected under --ours_task_homogeneous",
        _raises(
            lambda: module.CrossModalTopologyLoss(_args())(distiller, no_metadata),
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

    # ------------------------------------------------- degenerate / zero weights
    # scipy drops explicit zeros, so duplicated embeddings (distance exactly 0)
    # are the case the constant shift in src/topology.py exists for.
    dup = torch.cat([q[:3], q[:3]], dim=0)
    dup_dist = cosine_distance_matrix(dup, c)
    dup_rows, dup_cols = bipartite_mst_edges(dup_dist)
    check(
        "zero-distance duplicates keep every MST edge",
        dup_rows.numel() == dup.size(0) + n_c - 1,
        f"{dup_rows.numel()} vs {dup.size(0) + n_c - 1}",
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
        "1x1 relation has a single MST edge",
        bipartite_mst_edges(single)[0].numel() == 1,
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
    check(
        "cosine distance never goes negative on identical vectors",
        float(cosine_distance_matrix(q, q).min()) >= 0.0,
    )

    # -------------------------------------------------------------- gradients
    merge_student = torch.randn(n_q, dim, requires_grad=True)
    merge_target = bipartite_merge_matrix(dist.float()).detach()

    def merge_loss(emb):
        student_merge = bipartite_merge_matrix(
            cosine_distance_matrix(emb, c.float())
        )
        return (student_merge - merge_target).abs().mean()

    start = merge_loss(merge_student)
    start.backward()
    start = float(start.detach())
    check(
        "Ours routes a finite non-zero gradient through witness edges",
        merge_student.grad is not None
        and torch.isfinite(merge_student.grad).all()
        and merge_student.grad.abs().sum() > 0,
    )

    opt = torch.optim.Adam([merge_student], lr=5e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = merge_loss(merge_student)
        loss.backward()
        opt.step()
    # It does not reach zero, and the bound here is deliberately loose. The
    # gradient only ever moves the weights of the V-1 edges currently in the
    # student's MST; nothing pushes a *different* edge into the tree. Descent
    # therefore converges to whatever tree structure it started from -- here to
    # ~35% of the initial loss, with 5 of 15 MST edges shared with the teacher.
    check(
        "gradient descent drives the Ours loss down",
        float(loss.detach()) < 0.5 * start,
        f"{start:.3e} -> {float(loss.detach()):.3e}",
    )

    criterion_checks()

    # -------------------------------------------------------------- invariance
    perm_q = torch.randperm(n_q)
    perm_c = torch.randperm(n_c)
    permuted = cosine_distance_matrix(q[perm_q], c[perm_c])
    vertex_perm = torch.cat([perm_q, n_q + perm_c])
    check(
        "merge hierarchy is equivariant to simultaneous identity permutation",
        torch.allclose(
            bipartite_merge_matrix(permuted),
            merge[vertex_perm][:, vertex_perm],
        ),
    )
    # Permuting candidate identities alone leaves the MST edge weights -- and
    # hence any sorted persistence diagram built from them -- untouched, while
    # the merge matrix moves. That gap is the whole claim of the method.
    candidate_permuted = dist[:, perm_c]
    permuted_edges = bipartite_mst_edges(candidate_permuted)
    check(
        "candidate permutation exposes the barcode's identity blindness",
        torch.allclose(
            torch.sort(candidate_permuted[permuted_edges]).values,
            torch.sort(dist[rows, cols]).values,
        )
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
