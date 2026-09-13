"""Correspondence-aware cross-modal merge distillation (CM-Merge).

Implements ``docs/cross_modal_topological_distillation.md``:

    L = L_retrieval + cmtop_weight * mean_{a<b} |U_t[a,b] - U_s[a,b]|

``U[a,b]`` is the threshold at which indexed vertices ``a`` and ``b`` first
become connected in the bipartite query-candidate filtration. It preserves the
teacher's hierarchy *and* the identity correspondence of concrete examples; a
candidate permutation can retain the old H0 barcode exactly but changes ``U``.

Only the teacher's final embeddings are read -- no hidden states, no attention
maps -- which is the black-box property the brief asks to preserve.

The no-teacher control is this same criterion with ``--cmtop_weight 0``: the
retrieval objective alone, under exactly the same sampler and batch
construction, which is what makes the two rows comparable.
"""

import torch

from src.criterions.base import CriterionContext, DistillCriterion
from src.topology import bipartite_merge_matrix, cosine_distance_matrix


class CrossModalTopologyLoss(DistillCriterion):
    """``--kd_loss_type cmtop``. See the module docstring."""

    def __init__(self, args):
        super().__init__(args)
        # The base class computes `contrastive + kd_loss_weight * kd_loss` and
        # scales `kd_loss_weight` from --kd_weight. CM-Merge has exactly one
        # term with its own weight, so neutralise the outer scale rather than
        # multiplying the two together.
        self.kd_loss_weight = 1.0
        self.cmtop_weight = args.cmtop_weight
        self.reduction = args.cmtop_reduction
        self.merge_block = getattr(args, "cmtop_merge_block", "all")
        self.require_task_homogeneous = getattr(args, "cmtop_task_homogeneous", True)
        self.deduplicate_candidates = getattr(
            args, "cmtop_deduplicate_candidates", True
        )

        if self.reduction not in {"mean", "sum"}:
            raise ValueError(
                f"--cmtop_reduction must be mean or sum, got {self.reduction!r}"
            )
        if self.merge_block not in {"all", "cross"}:
            raise ValueError(
                f"--cmtop_merge_block must be all or cross, got {self.merge_block!r}"
            )

    # ------------------------------------------------------------------ utils
    #
    # The batch is widened across ranks (so the relation graph sees more of it)
    # and the teacher embeddings are fetched by DistillCriterion; both live in
    # src/criterions/base.py, shared with every other method.

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

    def _validate_task(self, ctx):
        if not self.require_task_homogeneous:
            return
        if ctx.task_ids is None:
            raise ValueError(
                "--cmtop_task_homogeneous needs per-row task ids, but the batch "
                "carries none; pass --cmtop_task_homogeneous False to run "
                "CM-Merge on a mixed-task relation deliberately"
            )
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
        """Compute the CM-Merge term for one batch.

        Encoding, gathering and the contrastive loss are handled by
        :class:`~src.criterions.base.DistillCriterion`.
        """
        student_qry, teacher_qry = ctx.student_qry, ctx.teacher_qry
        merge_loss = ctx.zeros()
        unique_candidates = ctx.student_pos.size(0)

        if self.cmtop_weight > 0:
            self._validate_task(ctx)
            student_pos, teacher_pos, unique_candidates = self._relation_candidates(ctx)
            merge_loss = self._merge_loss(
                cosine_distance_matrix(student_qry, student_pos),
                cosine_distance_matrix(teacher_qry, teacher_pos),
            )

        unique_fraction = torch.as_tensor(
            unique_candidates / max(ctx.student_pos.size(0), 1),
            dtype=torch.float32,
            device=student_qry.device,
        )
        return {
            "kd_loss": self.cmtop_weight * merge_loss,
            "cmmerge_loss": merge_loss,
            "num_unique_candidates": torch.as_tensor(
                unique_candidates, dtype=torch.float32, device=student_qry.device
            ),
            "candidate_unique_fraction": unique_fraction,
        }
