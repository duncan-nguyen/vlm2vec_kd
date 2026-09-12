"""Correspondence-aware cross-modal merge distillation (CM-Merge).

Implements ``docs/cross_modal_topological_distillation.md``:

    L = L_retrieval + cmtop_weight * mean_{a<b} |U_t[a,b] - U_s[a,b]|

``U[a,b]`` is the threshold at which indexed vertices ``a`` and ``b`` first
become connected in the bipartite query-candidate filtration. It preserves the
teacher's hierarchy *and* the identity correspondence of concrete examples; a candidate
permutation can retain the old H0 barcode exactly but changes ``U``.  The
criterion also contains the controls needed to isolate that contribution:

| brief's row                      | flags                                                     |
| -------------------------------- | --------------------------------------------------------- |
| student only                     | ``--kd_weight 0 --cmtop_weight 0 --cmtop_endpoint_kd none`` |
| endpoint KD                      | ``--cmtop_weight 0``                                        |
| relation-matrix / VSP KD         | ``--cmtop_weight 0 --cmtop_geometry_weight 1``              |
| point-cloud H0 KD                | ``--cmtop_mode point_cloud``                                |
| barcode H0 KD                    | ``--cmtop_mode cross_modal``                                |
| critical-edge KD                 | ``--cmtop_mode critical_edges``                            |
| correspondence-aware CM-Merge    | ``--cmtop_mode merge``                                     |

Only the teacher's final embeddings are read -- no hidden states, no attention
maps -- which is the black-box property the brief asks to preserve.
"""

import torch
import torch.nn.functional as F

from src.criterions.base import CriterionContext, DistillCriterion
from src.topology import (
    bipartite_merge_matrix,
    bipartite_mst_edges,
    cosine_distance_matrix,
    h0_deaths,
    h1_births,
    point_cloud_mst_edges,
    wasserstein2_sorted,
)

VALID_MODES = {"merge", "critical_edges", "cross_modal", "point_cloud", "union"}


class CrossModalTopologyLoss(DistillCriterion):
    """``--kd_loss_type cmtop``. See the module docstring for the ablation grid."""

    def __init__(self, args):
        super().__init__(args)
        # The base class computes `contrastive + kd_loss_weight * kd_loss`, but
        # The criterion exposes structural, endpoint and dense-geometry
        # ablations with separate weights (the endpoint term is the one
        # --kd_weight scales). Neutralise the base class's outer weight so the
        # launchers can select exactly one control without double scaling it.
        self.endpoint_weight = args.kd_weight
        self.kd_loss_weight = 1.0
        self.cmtop_weight = args.cmtop_weight
        self.h0_weight = args.cmtop_h0_weight
        self.h1_weight = args.cmtop_h1_weight
        self.h1_topk = args.cmtop_h1_topk
        self.geometry_weight = args.cmtop_geometry_weight
        self.endpoint_kd = args.cmtop_endpoint_kd
        self.normalize_scale = args.cmtop_normalize_scale
        self.reduction = args.cmtop_reduction
        self.merge_block = getattr(args, "cmtop_merge_block", "all")
        self.require_task_homogeneous = getattr(
            args, "cmtop_task_homogeneous", True
        )
        self.deduplicate_candidates = getattr(
            args, "cmtop_deduplicate_candidates", True
        )

        self.modes = [m for m in str(args.cmtop_mode).split("+") if m]
        unknown = set(self.modes) - VALID_MODES
        if unknown:
            raise ValueError(
                f"unknown --cmtop_mode component(s) {sorted(unknown)}; "
                f"expected a '+'-joined subset of {sorted(VALID_MODES)}"
            )
        if self.endpoint_kd not in {"cosine", "mse", "none"}:
            raise ValueError(
                f"--cmtop_endpoint_kd must be one of cosine/mse/none, "
                f"got {self.endpoint_kd!r}"
            )
        if self.reduction not in {"mean", "sum"}:
            raise ValueError(
                f"--cmtop_reduction must be mean or sum, got {self.reduction!r}"
            )
        if self.merge_block not in {"all", "cross"}:
            raise ValueError(
                f"--cmtop_merge_block must be all or cross, got {self.merge_block!r}"
            )
        # H1 only exists for the bipartite relation: a point cloud's flag complex
        # is full of triangles, so its cycles die and births alone say nothing.
        # Asking for it under another mode would silently contribute zero.
        if self.h1_weight > 0 and "cross_modal" not in self.modes:
            raise ValueError(
                f"--cmtop_h1_weight {self.h1_weight} has no effect under "
                f"--cmtop_mode {args.cmtop_mode!r}; the H1-birth term is defined "
                f"on the cross-modal relation only"
            )

    # ------------------------------------------------------------------ utils
    #
    # The batch is widened across ranks (so the relation graph sees more of it)
    # and the teacher embeddings are fetched by DistillCriterion; both live in
    # src/criterions/base.py now, shared with every other method.

    def _endpoint_loss(self, student_reps, teacher_reps_projected):
        if self.endpoint_kd == "cosine":
            return (
                1.0
                - F.cosine_similarity(
                    student_reps.float(), teacher_reps_projected.float(), dim=-1
                )
            ).mean()
        return F.mse_loss(student_reps.float(), teacher_reps_projected.float())

    def _rescale(self, diagram):
        """Optionally divide a diagram by its mean bar length.

        A student whose embeddings are globally more (or less) spread than the
        teacher's pays a constant offset on every bar; rescaling removes that and
        leaves only the *shape* of the barcode. Off by default -- both sides are
        cosine distances on the same scale, so the raw comparison is meaningful.
        """
        if not self.normalize_scale or diagram.numel() == 0:
            return diagram
        return diagram / (diagram.mean().detach() + 1e-8)

    def _w2(self, student_diagram, teacher_diagram):
        return wasserstein2_sorted(
            self._rescale(student_diagram),
            self._rescale(teacher_diagram),
            reduction=self.reduction,
        )

    # ------------------------------------------------------------- topology

    def _reduce(self, values):
        if values.numel() == 0:
            return values.new_zeros(())
        return values.mean() if self.reduction == "mean" else values.sum()

    def _merge_loss(self, student_dists, teacher_dists):
        """L1 discrepancy between identity-aligned minimax connectivity matrices."""
        student_merge = bipartite_merge_matrix(student_dists)
        teacher_merge = bipartite_merge_matrix(teacher_dists)
        n_q, n_c = student_dists.shape
        if self.merge_block == "cross":
            values = (student_merge[:n_q, n_q:] - teacher_merge[:n_q, n_q:]).abs()
        else:
            upper = torch.triu(
                torch.ones(
                    n_q + n_c,
                    n_q + n_c,
                    dtype=torch.bool,
                    device=student_dists.device,
                ),
                diagonal=1,
            )
            values = (student_merge - teacher_merge).abs()[upper]
        return self._reduce(values)

    def _critical_edge_loss(self, student_dists, teacher_dists):
        """Correspondence-aware MST-edge matching (TopoAE-style control)."""
        n_c = student_dists.size(1)
        selected = []
        for dists in (student_dists, teacher_dists):
            rows, cols = bipartite_mst_edges(dists)
            selected.append(rows * n_c + cols)
        flat_indices = torch.unique(torch.cat(selected))
        differences = (
            student_dists.reshape(-1)[flat_indices]
            - teacher_dists.reshape(-1)[flat_indices]
        ).abs()
        return self._reduce(differences)

    def _cross_modal_terms(self, student_dists, teacher_dists):
        """H0 (and optionally H1-birth) discrepancy of the bipartite relation."""
        student_edges = bipartite_mst_edges(student_dists)
        teacher_edges = bipartite_mst_edges(teacher_dists)

        h0 = self._w2(
            h0_deaths(student_dists, student_edges),
            h0_deaths(teacher_dists, teacher_edges),
        )

        h1 = student_dists.new_zeros(())
        if self.h1_weight > 0:
            # Same k on both sides, otherwise the diagrams are not comparable.
            topk = self.h1_topk if self.h1_topk > 0 else sum(student_dists.shape)
            h1 = self._w2(
                h1_births(student_dists, student_edges, topk),
                h1_births(teacher_dists, teacher_edges, topk),
            )
        return h0, h1

    def _point_cloud_h0(self, student_points, teacher_points):
        """H0 discrepancy of one point cloud (the ordinary, non-relational KD)."""
        student_dists = cosine_distance_matrix(student_points, student_points)
        teacher_dists = cosine_distance_matrix(teacher_points, teacher_points)
        return self._w2(
            h0_deaths(student_dists, point_cloud_mst_edges(student_dists)),
            h0_deaths(teacher_dists, point_cloud_mst_edges(teacher_dists)),
        )

    def _topology_loss(
        self,
        student_qry,
        student_pos,
        teacher_qry,
        teacher_pos,
        student_cross,
        teacher_cross,
    ):
        merge = student_qry.new_zeros((), dtype=torch.float32)
        critical_edges = student_qry.new_zeros((), dtype=torch.float32)
        h0 = student_qry.new_zeros((), dtype=torch.float32)
        h1 = student_qry.new_zeros((), dtype=torch.float32)

        if "merge" in self.modes:
            merge = self._merge_loss(student_cross, teacher_cross)

        if "critical_edges" in self.modes:
            critical_edges = self._critical_edge_loss(student_cross, teacher_cross)

        if "cross_modal" in self.modes:
            h0_cm, h1 = self._cross_modal_terms(student_cross, teacher_cross)
            h0 = h0 + h0_cm

        if "point_cloud" in self.modes:
            # The control from the brief: the two modalities as separate clouds,
            # which says nothing about how they connect to each other.
            h0 = h0 + 0.5 * (
                self._point_cloud_h0(student_qry, teacher_qry)
                + self._point_cloud_h0(student_pos, teacher_pos)
            )

        if "union" in self.modes:
            h0 = h0 + self._point_cloud_h0(
                torch.cat([student_qry, student_pos], dim=0),
                torch.cat([teacher_qry, teacher_pos], dim=0),
            )

        return merge, critical_edges, h0, h1

    def _geometry_loss(self, student_cross, teacher_cross):
        """VSP-style baseline: match the cross-modal similarity matrices.

        Same relation, same inputs as the topological term, but comparing raw
        pairwise geometry instead of persistence -- the control that isolates
        what the topology actually adds. Takes the distance matrices the
        topological term already built rather than recomputing them.
        """
        return F.smooth_l1_loss(1.0 - student_cross, 1.0 - teacher_cross)

    def _validate_task(self, ctx):
        if not self.require_task_homogeneous or ctx.task_ids is None:
            return
        if torch.any(ctx.task_ids < 0):
            raise ValueError("CM-Merge received an invalid placeholder task id")
        tasks = torch.unique(ctx.task_ids)
        if tasks.numel() != 1:
            raise ValueError(
                "CM-Merge requires one task per gathered batch, got task ids "
                f"{tasks.detach().cpu().tolist()}"
            )

    def _relation_candidates(self, ctx):
        """Return canonical candidate rows while preserving every query row."""
        student_pos, teacher_pos = ctx.student_pos, ctx.teacher_pos
        ids = ctx.candidate_ids
        if not self.deduplicate_candidates or ids is None:
            return student_pos, teacher_pos, student_pos.size(0)
        # Stable hashes legitimately span signed int64, so only -1 is reserved;
        # negative values other than -1 are valid identities.
        if torch.any(ids == -1):
            raise ValueError("CM-Merge received an invalid placeholder candidate id")

        seen = set()
        keep = []
        for index, candidate_id in enumerate(ids.detach().cpu().tolist()):
            if candidate_id not in seen:
                seen.add(candidate_id)
                keep.append(index)
        indices = torch.tensor(keep, dtype=torch.long, device=student_pos.device)
        return (
            student_pos.index_select(0, indices),
            teacher_pos.index_select(0, indices),
            len(keep),
        )

    # ----------------------------------------------------------------- kd

    def kd_loss(self, ctx: CriterionContext):
        """Compute the selected structural target and optional controls.

        Encoding, gathering and the contrastive loss are handled by
        :class:`~src.criterions.base.DistillCriterion`.
        """
        student_qry, student_pos = ctx.student_qry, ctx.student_pos
        teacher_qry, teacher_pos = ctx.teacher_qry, ctx.teacher_pos
        zero = ctx.zeros()

        endpoint_loss = zero
        if self.endpoint_kd != "none" and self.endpoint_weight > 0:
            endpoint_loss = 0.5 * (
                self._endpoint_loss(student_qry, ctx.project_teacher(teacher_qry))
                + self._endpoint_loss(student_pos, ctx.project_teacher(teacher_pos))
            )

        # The topological and the VSP terms read the same two matrices; build
        # them once and only when something actually needs them.
        relation_modes = {"merge", "critical_edges", "cross_modal"}
        wants_topology = self.cmtop_weight > 0 and bool(self.modes)
        wants_structure = self.geometry_weight > 0 or wants_topology
        wants_relation = self.geometry_weight > 0 or (
            self.cmtop_weight > 0 and bool(relation_modes.intersection(self.modes))
        )
        student_cross = teacher_cross = None
        unique_candidates = student_pos.size(0)
        relation_student_pos, relation_teacher_pos = student_pos, teacher_pos
        if wants_structure:
            self._validate_task(ctx)
            relation_student_pos, relation_teacher_pos, unique_candidates = (
                self._relation_candidates(ctx)
            )
        if wants_relation:
            student_cross = cosine_distance_matrix(student_qry, relation_student_pos)
            teacher_cross = cosine_distance_matrix(teacher_qry, relation_teacher_pos)

        merge_loss, critical_edge_loss, h0_loss, h1_loss = (zero, zero, zero, zero)
        if wants_topology:
            merge_loss, critical_edge_loss, h0_loss, h1_loss = self._topology_loss(
                student_qry,
                relation_student_pos,
                teacher_qry,
                relation_teacher_pos,
                student_cross,
                teacher_cross,
            )

        geometry_loss = zero
        if self.geometry_weight > 0:
            geometry_loss = self._geometry_loss(student_cross, teacher_cross)

        cmtop_loss = (
            merge_loss
            + critical_edge_loss
            + self.h0_weight * h0_loss
            + self.h1_weight * h1_loss
        )
        unique_fraction = torch.as_tensor(
            unique_candidates / max(student_pos.size(0), 1),
            dtype=torch.float32,
            device=student_qry.device,
        )
        return {
            "kd_loss": (
                self.endpoint_weight * endpoint_loss
                + self.cmtop_weight * cmtop_loss
                + self.geometry_weight * geometry_loss
            ),
            "endpoint_loss": endpoint_loss,
            "cmtop_loss": cmtop_loss,
            "cmmerge_loss": merge_loss,
            "critical_edge_loss": critical_edge_loss,
            "cmtop_h0_loss": h0_loss,
            "cmtop_h1_loss": h1_loss,
            "geometry_loss": geometry_loss,
            "num_unique_candidates": torch.as_tensor(
                unique_candidates, dtype=torch.float32, device=student_qry.device
            ),
            "candidate_unique_fraction": unique_fraction,
        }
