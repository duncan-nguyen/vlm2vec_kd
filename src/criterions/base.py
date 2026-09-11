"""Shared machinery for distillation criteria.

Every criterion in this package repeats the same four things: encode both sides
with the student, fetch the teacher's embeddings, widen the batch across ranks,
and compute the in-batch contrastive loss. Eleven copies of that had drifted --
some gather before the contrastive loss and some do not, some build a fresh
`nn.CrossEntropyLoss()` per step, `contrastive_rkd` and `universal_logit` never
gather at all -- which makes the multi-GPU numbers of different methods not
directly comparable.

:class:`DistillCriterion` holds one copy. A new method subclasses it and writes
only :meth:`kd_loss`; see docs/adding_a_method.md.

Nothing here is mandatory: the older criteria are still plain `nn.Module`s and
keep working. The helpers are also available as module-level functions so they
can be reused without subclassing.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from src import dist_utils

# ---------------------------------------------------------------- primitives


def pooled(encoder_output):
    """Pull the pooled embedding out of ``MMEBModel.encode_input``.

    Most backbones return ``(pooled, image_features, attentions, hidden_states)``
    but a few return the pooled tensor on its own.
    """
    if isinstance(encoder_output, (tuple, list)):
        return encoder_output[0]
    return encoder_output


def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def gather_with_grad(x):
    """All-gather that keeps this rank's slice differentiable.

    The gradient of the gathered tensor flows back only into the local slice,
    which is what makes the widened in-batch contrastive loss correct: every
    rank contributes its own rows' gradient exactly once.
    """
    if not is_distributed():
        return x
    return dist_utils.dist_gather(x.contiguous())


def gather_no_grad(x):
    """All-gather for a constant (the teacher side), without the autograd hook."""
    if not is_distributed():
        return x
    return dist_utils.dist_gather_nograd(x.contiguous())


def in_batch_contrastive_loss(student_model, qry_reps, pos_reps, temperature):
    """In-batch softmax over query/positive similarities.

    ``qry_reps`` may hold several positives per query (``len(qry) % len(pos) == 0``
    after the reshape), which is what the stride on ``target`` accounts for.
    """
    scores = student_model.compute_similarity(qry_reps, pos_reps)
    scores = scores.view(qry_reps.size(0), -1)
    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
    target = target * (qry_reps.size(0) // pos_reps.size(0))
    return F.cross_entropy(scores / temperature, target)


def project_teacher(distiller, teacher_reps, student_dim, projector="t2s"):
    """Map teacher embeddings into the student space, if a projector exists.

    Returns ``teacher_reps`` unchanged when the dimensions already agree, so a
    same-width teacher/student pair needs no projector config at all.
    """
    projectors = getattr(distiller, "projectors", None)
    if isinstance(projectors, nn.ModuleDict) and projector in projectors:
        return projectors[projector](teacher_reps)
    if teacher_reps.size(-1) == student_dim:
        return teacher_reps
    raise ValueError(
        f"no {projector!r} projector: teacher dim {teacher_reps.size(-1)} != "
        f"student dim {student_dim}. Pass --projector_config_path with a "
        f"{projector!r} entry enabled."
    )


# ------------------------------------------------------------------ the base


class DistillCriterion(nn.Module):
    """Base class for a distillation method.

    A subclass implements :meth:`kd_loss` and gets the contrastive term, the
    cross-rank gather and the teacher lookup for free::

        class MyLoss(DistillCriterion):
            def kd_loss(self, ctx):
                return {"kd_loss": F.mse_loss(ctx.student_qry, ctx.teacher_qry)}

    :meth:`forward` combines them as ``contrastive + kd_weight * kd_loss`` and
    returns the dict the training loop logs. Everything a subclass might want is
    on the :class:`CriterionContext` handed to :meth:`kd_loss`.
    """

    #: Encode with the student and gather across ranks before `kd_loss` runs.
    #: Turn off only for a criterion that must see the raw per-rank batch.
    gather_reps = True
    #: Fetch the teacher's pooled embeddings before `kd_loss` runs. A criterion
    #: that needs the teacher's hidden states or attentions instead should set
    #: this False and encode the teacher itself.
    fetch_teacher_reps = True

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kd_loss_weight = getattr(args, "kd_weight", 1.0)

    # -- to implement -----------------------------------------------------

    def kd_loss(self, ctx):
        """Return the KD terms for one batch.

        Args:
            ctx: a :class:`CriterionContext`.

        Returns:
            Either a tensor (used as ``kd_loss``) or a dict of named scalars.
            The dict must contain ``kd_loss``; any other key is logged as-is,
            which is how a method surfaces its own components.
        """
        raise NotImplementedError

    # -- optional ---------------------------------------------------------

    def build_parameters(self, distiller):
        """Create whatever weights this method trains alongside the student.

        A no-op for almost every criterion: the shared KD projectors live on the
        `Distiller` and are configured by ``--projector_config_path``. A method
        whose projections are part of the *method* rather than of the
        teacher/student pair -- TALAS has one per anchored layer, sized from
        ``--talas_num_tamd_layers`` -- builds them here instead.

        `src.training.entrypoint` calls this once, after the `Distiller` exists
        (so `student_hidden_dim` / `teacher_hidden_dim` are known) and before the
        optimizer is built. Anything registered on ``self`` then gets its own
        parameter group, is moved to the device and is covered by DDP; anything
        created later is none of those things.
        """

    # -- the shared half --------------------------------------------------

    def forward(self, distiller, input_data, **kwargs):
        ctx = CriterionContext.build(
            self,
            distiller,
            input_data,
            gather=self.gather_reps,
            fetch_teacher=self.fetch_teacher_reps,
            **kwargs,
        )

        contrastive_loss = in_batch_contrastive_loss(
            distiller.student, ctx.student_qry, ctx.student_pos, distiller.temperature
        )

        terms = self.kd_loss(ctx)
        if isinstance(terms, torch.Tensor):
            terms = {"kd_loss": terms}
        if "kd_loss" not in terms:
            raise KeyError(
                f"{type(self).__name__}.kd_loss() must return a 'kd_loss' entry, "
                f"got keys {sorted(terms)}"
            )

        out = dict(terms)
        out["contrastive_loss"] = contrastive_loss
        out["loss"] = contrastive_loss + self.kd_loss_weight * terms["kd_loss"]
        return out


class CriterionContext:
    """The per-batch state every criterion asks for, computed once.

    Attributes:
        distiller: the :class:`~src.distiller.Distiller`.
        batch: the collated batch, for a criterion that needs the raw inputs.
        student_qry / student_pos: pooled student embeddings, gathered across
            ranks when ``gather`` is on.
        teacher_qry / teacher_pos: pooled teacher embeddings, from the model or
            from ``--teacher_embedding_cache``, gathered the same way.
        task_ids / candidate_ids: optional row metadata, gathered in exactly the
            same rank order as the embeddings.  Relation-level criteria use it
            to reject mixed-task graphs and collapse repeated candidates.
        student_qry_out / student_pos_out: the full tuples from
            ``encode_input`` -- hidden states and attentions, for a criterion
            that reads them. Always the *local* batch, never gathered: those
            tensors are far too large to all-gather.
        kwargs: whatever ``Distiller.forward`` passed through, e.g. ``tokenizer``.
    """

    __slots__ = (
        "batch",
        "distiller",
        "kwargs",
        "student_pos",
        "student_pos_out",
        "student_qry",
        "student_qry_out",
        "candidate_ids",
        "task_ids",
        "teacher_pos",
        "teacher_qry",
    )

    def __init__(self, **fields):
        for slot in self.__slots__:
            setattr(self, slot, fields.get(slot))

    @classmethod
    def build(
        cls, criterion, distiller, batch, gather=True, fetch_teacher=True, **kwargs
    ):
        student = distiller.student
        qry_out = student.encode_input(batch["student_inputs"]["qry"])
        pos_out = student.encode_input(batch["student_inputs"]["pos"])
        student_qry, student_pos = pooled(qry_out), pooled(pos_out)

        teacher_qry = teacher_pos = None
        if fetch_teacher:
            # Through the distiller, never `distiller.teacher`: with
            # --teacher_embedding_cache there is no teacher model to call.
            dtype = student_qry.dtype
            teacher_qry = distiller.encode_teacher(batch, "qry", dtype=dtype)
            teacher_pos = distiller.encode_teacher(batch, "pos", dtype=dtype)

        if gather:
            student_qry, student_pos = (
                gather_with_grad(student_qry),
                gather_with_grad(student_pos),
            )
            if teacher_qry is not None:
                teacher_qry, teacher_pos = (
                    gather_no_grad(teacher_qry),
                    gather_no_grad(teacher_pos),
                )

        # Metadata is collated at the batch root and moved to the device by the
        # training loop.  Keep it optional so older datasets and unit-test
        # fixtures continue to work, but when present require one id per row.
        task_ids = batch.get("task_ids")
        candidate_ids = batch.get("candidate_ids")
        metadata = {"task_ids": task_ids, "candidate_ids": candidate_ids}
        for name, ids in metadata.items():
            if ids is None:
                continue
            ids = ids.reshape(-1).to(device=student_qry.device, dtype=torch.long)
            local_rows = pooled(qry_out).size(0)
            if ids.numel() != local_rows:
                raise ValueError(
                    f"{name} has {ids.numel()} entries for {local_rows} local rows"
                )
            if gather:
                ids = gather_no_grad(ids)
            metadata[name] = ids

        if (
            candidate_ids is not None
            and metadata["candidate_ids"].numel() != student_pos.size(0)
        ):
            raise ValueError(
                "candidate_ids and positive embeddings must have the same gathered length"
            )

        return cls(
            distiller=distiller,
            batch=batch,
            student_qry=student_qry,
            student_pos=student_pos,
            teacher_qry=teacher_qry,
            teacher_pos=teacher_pos,
            student_qry_out=qry_out,
            student_pos_out=pos_out,
            task_ids=metadata["task_ids"],
            candidate_ids=metadata["candidate_ids"],
            kwargs=kwargs,
        )

    # -- conveniences -----------------------------------------------------

    @property
    def student_dim(self):
        return self.student_qry.size(-1)

    def project_teacher(self, teacher_reps, projector="t2s"):
        """Teacher embeddings mapped into the student space."""
        return project_teacher(
            self.distiller, teacher_reps, self.student_dim, projector=projector
        )

    def zeros(self):
        """A float32 scalar zero on the right device, for an unused term."""
        return self.student_qry.new_zeros((), dtype=torch.float32)
