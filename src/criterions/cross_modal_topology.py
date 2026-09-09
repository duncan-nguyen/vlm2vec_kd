"""Cross-modal topological distillation (CMTop).

Implements ``docs/cross_modal_topological_distillation.md``:

    L = L_contrastive + kd_weight * L_endpoint + cmtop_weight * L_CMTop
    L_CMTop = cmtop_h0_weight * W2^2(H0_t, H0_s) + cmtop_h1_weight * W2^2(H1b_t, H1b_s)

The topology is that of the *retrieval relation* -- the bipartite graph whose
edges are query-candidate distances -- not that of the query and candidate point
clouds taken separately. ``--cmtop_mode`` switches between the two so the ablation
in section 4 of the brief runs off one criterion:

| brief's row                      | flags                                                     |
| -------------------------------- | --------------------------------------------------------- |
| student only                     | ``--kd_weight 0 --cmtop_weight 0 --cmtop_endpoint_kd none`` |
| endpoint KD                      | ``--cmtop_weight 0``                                        |
| pairwise / VSP-style geometry KD | ``--cmtop_weight 0 --cmtop_geometry_weight 1``              |
| point-cloud H0 KD                | ``--cmtop_mode point_cloud``                                |
| cross-modal relation H0 (main)   | ``--cmtop_mode cross_modal``                                |
| + lightweight H1-birth           | ``--cmtop_mode cross_modal --cmtop_h1_weight 0.1``          |

Only the teacher's final embeddings are read -- no hidden states, no attention
maps -- which is the black-box property the brief asks to preserve.
"""

import torch
import torch.nn.functional as F

from src.criterions.base import CriterionContext, DistillCriterion
from src.topology import (
    bipartite_mst_edges,
    cosine_distance_matrix,
    h0_deaths,
    h1_births,
    point_cloud_mst_edges,
    wasserstein2_sorted,
)

VALID_MODES = {"cross_modal", "point_cloud", "union"}


class CrossModalTopologyLoss(DistillCriterion):
    """``--kd_loss_type cmtop``. See the module docstring for the ablation grid."""

    def __init__(self, args):
        super().__init__(args)
        # The base class computes `contrastive + kd_loss_weight * kd_loss`, but
        # CMTop applies a different weight to each of its three KD terms (the
        # endpoint term is the one --kd_weight scales). Neutralise the outer
        # weight and keep --kd_weight for the endpoint term alone.
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
        h0 = student_qry.new_zeros((), dtype=torch.float32)
        h1 = student_qry.new_zeros((), dtype=torch.float32)

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

        return h0, h1

    def _geometry_loss(self, student_cross, teacher_cross):
        """VSP-style baseline: match the cross-modal similarity matrices.

        Same relation, same inputs as the topological term, but comparing raw
        pairwise geometry instead of persistence -- the control that isolates
        what the topology actually adds. Takes the distance matrices the
        topological term already built rather than recomputing them.
        """
        return F.smooth_l1_loss(1.0 - student_cross, 1.0 - teacher_cross)

    # ----------------------------------------------------------------- kd

    def kd_loss(self, ctx: CriterionContext):
        """The three KD terms. Encoding, gathering and the contrastive loss are
        done by :class:`~src.criterions.base.DistillCriterion`."""
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
        wants_relation = self.geometry_weight > 0 or (
            self.cmtop_weight > 0 and "cross_modal" in self.modes
        )
        student_cross = teacher_cross = None
        if wants_relation:
            student_cross = cosine_distance_matrix(student_qry, student_pos)
            teacher_cross = cosine_distance_matrix(teacher_qry, teacher_pos)

        h0_loss, h1_loss = (zero, zero)
        if self.cmtop_weight > 0 and self.modes:
            h0_loss, h1_loss = self._topology_loss(
                student_qry,
                student_pos,
                teacher_qry,
                teacher_pos,
                student_cross,
                teacher_cross,
            )

        geometry_loss = zero
        if self.geometry_weight > 0:
            geometry_loss = self._geometry_loss(student_cross, teacher_cross)

        cmtop_loss = self.h0_weight * h0_loss + self.h1_weight * h1_loss
        return {
            "kd_loss": (
                self.endpoint_weight * endpoint_loss
                + self.cmtop_weight * cmtop_loss
                + self.geometry_weight * geometry_loss
            ),
            "endpoint_loss": endpoint_loss,
            "cmtop_loss": cmtop_loss,
            "cmtop_h0_loss": h0_loss,
            "cmtop_h1_loss": h1_loss,
            "geometry_loss": geometry_loss,
        }
