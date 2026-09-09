"""Relational KD on the final embeddings (`--kd_loss_type contrastive_rkd`).

Park et al.'s distance and angle potentials, computed over the concatenation of
the query and positive embeddings. Reads nothing from the teacher but its final
embedding, so it runs against `--teacher_embedding_cache`.
"""

import torch

from src.criterions.base import CriterionContext, DistillCriterion


def _huber(diff):
    """Smooth L1 with beta = 1, matching the original RKD formulation."""
    abs_diff = torch.abs(diff)
    return torch.where(abs_diff < 1.0, 0.5 * abs_diff**2, abs_diff - 0.5).mean()


class ContrastiveLossWithRKD(DistillCriterion):
    def __init__(self, args):
        super().__init__(args)
        # NOTE: --rkd_distance_weight / --rkd_angle_weight are parsed and stored
        # but the combination below is hard-coded to 0.5/0.5, so setting them on
        # the command line has no effect. Left as-is deliberately: every run
        # recorded so far used the hard-coded split, and honouring the flags now
        # would silently change what an old command line means.
        self.distance_weight = args.rkd_distance_weight
        self.angle_weight = args.rkd_angle_weight

    def kd_loss(self, ctx: CriterionContext):
        student_repr = torch.cat([ctx.student_qry, ctx.student_pos], dim=0)
        teacher_repr = torch.cat([ctx.teacher_qry, ctx.teacher_pos], dim=0)

        distance_loss = self.compute_distance_loss(student_repr, teacher_repr)
        angle_loss = self.compute_angle_loss(student_repr, teacher_repr)
        return {
            "kd_loss": 0.5 * distance_loss + 0.5 * angle_loss,
            "rkd_distance_loss": distance_loss,
            "rkd_angle_loss": angle_loss,
        }

    def pairwise_distance(self, x):
        norm = (x**2).sum(dim=1, keepdim=True)
        return norm + norm.t() - 2.0 * torch.mm(x, x.t())

    def compute_distance_loss(self, student_repr, teacher_repr):
        dist_student = self.pairwise_distance(student_repr)
        dist_teacher = self.pairwise_distance(teacher_repr)

        mask = torch.triu(torch.ones_like(dist_student), diagonal=1).bool()
        dist_student = dist_student[mask]
        dist_teacher = dist_teacher[mask]

        # Each side is normalised by its own mean so the loss compares the shape
        # of the distance distribution rather than its scale.
        dist_student = dist_student / (dist_student.mean().detach() + 1e-8)
        dist_teacher = dist_teacher / (dist_teacher.mean().detach() + 1e-8)

        return _huber(dist_student - dist_teacher)

    def angle_potentials(self, x):
        diffs = x.unsqueeze(0) - x.unsqueeze(1)
        norms = torch.norm(diffs, dim=-1, keepdim=True) + 1e-8
        e = diffs / norms
        return torch.einsum("ijd,kjd->ijk", e, e)

    def compute_angle_loss(self, student_repr, teacher_repr):
        psi_student = self.angle_potentials(student_repr)
        psi_teacher = self.angle_potentials(teacher_repr)

        # Drop every triple with a repeated index: those angles are degenerate.
        n = psi_student.size(0)
        mask = torch.ones((n, n, n), dtype=torch.bool, device=psi_student.device)
        idx = torch.arange(n, device=psi_student.device)
        mask[idx, idx, :] = 0
        mask[idx, :, idx] = 0
        mask[:, idx, idx] = 0

        return _huber(psi_student[mask] - psi_teacher[mask])
