"""Universal logit distillation on the pooled embeddings (`--kd_loss_type universal_logit`).

Zero-pads the narrower of the two embedding spaces up to the wider one, then
matches student and teacher with an MSE symmetrised over the query/positive
pairing. Reads nothing from the teacher but its final embedding, so it runs
against `--teacher_embedding_cache`.
"""

import torch
import torch.nn.functional as F

from src.criterions.base import CriterionContext, DistillCriterion


class UniversalLogitDistillation(DistillCriterion):
    def kd_loss(self, ctx: CriterionContext):
        return self.compute_universal_logit_loss(
            ctx.student_qry, ctx.student_pos, ctx.teacher_qry, ctx.teacher_pos
        )

    def compute_universal_logit_loss(
        self, student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps
    ):
        size_gap = student_qry_reps.shape[-1] - teacher_qry_reps.shape[-1]
        # The padding is taken as a zeros_like slice of the tensor being padded,
        # so a gap wider than the narrower embedding cannot be expressed.
        if abs(size_gap) > min(student_qry_reps.shape[-1], teacher_qry_reps.shape[-1]):
            raise ValueError(
                f"universal_logit pads the narrower side up to the wider one, "
                f"which needs |student_dim - teacher_dim| <= min(dim); got "
                f"student {student_qry_reps.shape[-1]} and teacher "
                f"{teacher_qry_reps.shape[-1]}"
            )
        if size_gap > 0:
            teacher_qry_reps = torch.cat(
                [teacher_qry_reps, torch.zeros_like(teacher_qry_reps[:, :size_gap])],
                dim=-1,
            )
            teacher_pos_reps = torch.cat(
                [teacher_pos_reps, torch.zeros_like(teacher_pos_reps[:, :size_gap])],
                dim=-1,
            )
        elif size_gap < 0:
            student_qry_reps = torch.cat(
                [
                    student_qry_reps,
                    torch.zeros_like(student_qry_reps[:, :(-size_gap)]),
                ],
                dim=-1,
            )
            student_pos_reps = torch.cat(
                [
                    student_pos_reps,
                    torch.zeros_like(student_pos_reps[:, :(-size_gap)]),
                ],
                dim=-1,
            )

        return (
            F.mse_loss(student_qry_reps, teacher_qry_reps)
            + F.mse_loss(student_pos_reps, teacher_pos_reps)
            + F.mse_loss(student_qry_reps, teacher_pos_reps)
            + F.mse_loss(student_pos_reps, teacher_qry_reps)
        ) / 4.0
