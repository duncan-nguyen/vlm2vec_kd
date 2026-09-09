"""TALAS: Teacher-Anchored Layer Alignment with Adaptive Sharpness-aware
minimization (`--kd_loss_type talas`).

Implements `docs/baseline methods/TALAS.pdf` (see `docs/talas_implementation.md`
for the paper -> repo mapping). Three terms:

    L = lambda_1 * L_contrastive + lambda_2 * L_TAMD + lambda_3 * L_LASD

`L_TAMD` (eq. 3) anchors the student's **top K layers** to the teacher's single
final embedding through one learnable projection per layer::

    L_TAMD = 1/K * sum_{l=L-K+1..L} ( 1 - cos( e_l^S W_l , e^T ) )

`L_LASD` (eq. 5) is the student's own top-down self-distillation: the batch
relation matrix R_l = norm(E_l) norm(E_l)^T of layer `l` is pulled towards that
of the layer above it, which acts as a (detached) guide::

    L_LASD = 1/(L-1) * sum_l || R_{l+1} - R_l ||_F^2

`L_contrastive` stands in for the paper's unsupervised SimCSE term: TALAS is a
text-only method whose corpus has no pairs, so it manufactures two views with
dropout. This repo trains on (query, positive) pairs and the base class already
computes the in-batch softmax over them, which plays the same "keep the space
uniform" role. `--talas_contrastive_weight` is that term's lambda_1; the paper's
0.001 is calibrated for the dropout objective, not this one, so the default here
is 1.0 -- the weight every other method in the repo gives the contrastive term.

The optimizer half of TALAS (ASAM) is not part of the criterion: it is an
optimizer, usable with any method, and lives in `src/training/sam.py` behind
`--sharpness_aware asam`.

Reads nothing from the teacher but its final pooled embedding, so it runs
against `--teacher_embedding_cache` -- which is what the paper's Appendix C
prescribes anyway.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.criterions.base import CriterionContext, DistillCriterion, gather_with_grad

VALID_REDUCTIONS = {"sum", "mean"}


class TALASLoss(DistillCriterion):
    """``--kd_loss_type talas``. See the module docstring."""

    # The per-layer pooled student embeddings are gathered inside `kd_loss`
    # (one collective for the whole stack), so the KD terms see the same widened
    # batch as the contrastive term and the teacher embeddings line up row for
    # row without any rank arithmetic.
    gather_reps = True
    fetch_teacher_reps = True

    def __init__(self, args):
        super().__init__(args)
        # --kd_weight is not read: TALAS weights its two KD terms separately
        # with --talas_tamd_weight (lambda_2) and --talas_lasd_weight (lambda_3),
        # so the outer weight is neutralised the way CMTop neutralises it.
        self.kd_loss_weight = 1.0
        self.contrastive_weight = float(args.talas_contrastive_weight)
        self.tamd_weight = float(args.talas_tamd_weight)
        self.lasd_weight = float(args.talas_lasd_weight)
        self.num_tamd_layers = int(args.talas_num_tamd_layers)
        self.num_lasd_pairs = int(args.talas_num_lasd_layers)
        self.detach_guide = bool(args.talas_lasd_detach_guide)
        self.reduction = str(args.talas_lasd_reduction)

        if self.num_tamd_layers < 1:
            raise ValueError(
                f"--talas_num_tamd_layers must be >= 1 (the paper's K, best at 2), "
                f"got {self.num_tamd_layers}"
            )
        if self.reduction not in VALID_REDUCTIONS:
            raise ValueError(
                f"--talas_lasd_reduction must be one of {sorted(VALID_REDUCTIONS)}, "
                f"got {self.reduction!r}"
            )

        # One W_l per anchored layer, built once the student's width is known.
        # `projections[0]` is the *final* layer, `[1]` the one below it, and so
        # on, so lowering --talas_num_tamd_layers drops the deepest anchor
        # rather than renumbering every one of them.
        self.projections = None

    # -- parameters -------------------------------------------------------

    def build_parameters(self, distiller):
        """Create the K learnable ``W_l`` of eq. 3.

        Called by the entrypoint once the `Distiller` exists and before the
        optimizer is built, so these get a parameter group, reach the device and
        are covered by DDP like `distiller.projectors` are.
        """
        student_dim = int(distiller.student_hidden_dim)
        teacher_dim = int(distiller.teacher_hidden_dim)
        dtype = _trainable_dtype(distiller)

        projections = nn.ModuleList()
        for _ in range(self.num_tamd_layers):
            # No bias: eq. 3 is a plain matrix W_l in R^{d_S x d_T}.
            layer = nn.Linear(student_dim, teacher_dim, bias=False)
            # Semi-orthogonal, matching `Distiller.set_projector`; qr wants
            # float, so initialise in fp32 and cast afterwards.
            nn.init.orthogonal_(layer.weight)
            projections.append(layer.to(dtype=dtype))
        self.projections = projections
        return projections

    # -- the loss ---------------------------------------------------------

    def forward(self, distiller, input_data, **kwargs):
        out = super().forward(distiller, input_data, **kwargs)
        # The base class combines the terms as `contrastive + kd`, with the
        # contrastive term at weight 1. TALAS scales it by lambda_1.
        out["loss"] = self.contrastive_weight * out["contrastive_loss"] + out["kd_loss"]
        return out

    def kd_loss(self, ctx: CriterionContext):
        if self.projections is None:
            raise RuntimeError(
                "TALASLoss.build_parameters(distiller) was never called, so the "
                "per-layer projections W_l do not exist. The training entrypoint "
                "calls it; a custom driver has to as well."
            )

        tamd = ctx.zeros()
        lasd = ctx.zeros()
        for side, teacher in (("qry", ctx.teacher_qry), ("pos", ctx.teacher_pos)):
            layer_embeddings = self._layer_embeddings(ctx, side)
            tamd = tamd + self._tamd(layer_embeddings, teacher)
            lasd = lasd + self._lasd(layer_embeddings)
        # Query and positive are two streams of the paper's single sentence
        # batch; averaging keeps the loss the same size as a one-stream run.
        tamd, lasd = tamd / 2.0, lasd / 2.0

        return {
            "kd_loss": self.tamd_weight * tamd + self.lasd_weight * lasd,
            "talas_tamd_loss": tamd,
            "talas_lasd_loss": lasd,
        }

    # -- pieces -----------------------------------------------------------

    def _layer_embeddings(self, ctx, side):
        """``(N, L, D)``: the pooled embedding of every transformer layer.

        Pooled with the student's own `_pooling` and its own attention mask, so
        this is exactly as layout-aware as the model's final pooling is -- which
        is what keeps TALAS correct for both the right-padded FastVLM and the
        left-padded LLaVA-OneVision without either being special-cased.
        """
        out = ctx.student_qry_out if side == "qry" else ctx.student_pos_out
        hidden_states = out[3] if isinstance(out, (tuple, list)) else None
        if not hidden_states:
            raise RuntimeError(
                f"the student returned no hidden states for the {side!r} side; "
                f"TALAS needs them for every layer. `MMEBModel.encode_input` "
                f"passes output_hidden_states=True for the backbones it "
                f"supports -- this one returns something else."
            )

        attention_mask = ctx.batch["student_inputs"][side]["attention_mask"]
        student = ctx.distiller.student
        # hidden_states[0] is the embedding layer's output, not a transformer
        # layer; e_1^S .. e_L^S of the paper are hidden_states[1:].
        per_layer = [student._pooling(h, attention_mask) for h in hidden_states[1:]]
        stacked = torch.stack(per_layer, dim=1)
        # One collective for the whole stack rather than one per layer.
        return gather_with_grad(stacked)

    def _tamd(self, layer_embeddings, teacher_reps):
        """Eq. 3: cosine distance from the top K layers to the teacher anchor."""
        num_layers = layer_embeddings.size(1)
        k = min(self.num_tamd_layers, num_layers)
        teacher = teacher_reps.float()

        total = layer_embeddings.new_zeros((), dtype=torch.float32)
        for depth in range(k):
            student_layer = layer_embeddings[:, num_layers - 1 - depth, :]
            projected = self.projections[depth](student_layer)
            total = (
                total
                + (1.0 - F.cosine_similarity(projected.float(), teacher, dim=-1)).mean()
            )
        return total / k

    def _lasd(self, layer_embeddings):
        """Eq. 5: adjacent layers' batch relation matrices pulled together."""
        num_layers = layer_embeddings.size(1)
        num_pairs = num_layers - 1
        if self.num_lasd_pairs > 0:
            num_pairs = min(self.num_lasd_pairs, num_pairs)
        if num_pairs < 1:
            # A one-layer student has no adjacent pair to align.
            return layer_embeddings.new_zeros((), dtype=torch.float32)

        # The pairs are counted from the top, so `--talas_num_lasd_layers 2`
        # aligns the last three layers rather than the first three.
        first = num_layers - 1 - num_pairs
        relations = [
            _relation_matrix(layer_embeddings[:, layer, :])
            for layer in range(first, num_layers)
        ]

        total = layer_embeddings.new_zeros((), dtype=torch.float32)
        for i in range(num_pairs):
            lower, upper = relations[i], relations[i + 1]
            if self.detach_guide:
                # "we utilize the student's own upper layer (l+1) as a dynamic
                # guide for the immediate lower layer (l)" (sec. 3.2). Without
                # the detach the term is a symmetric smoothness penalty and the
                # teacher-anchored top gets dragged down towards the untrained
                # bottom, which is the opposite of what the section describes.
                upper = upper.detach()
            diff = (upper - lower).pow(2)
            total = total + (diff.mean() if self.reduction == "mean" else diff.sum())
        return total / num_pairs


# ---------------------------------------------------------------- helpers


def _relation_matrix(embeddings):
    """``R = norm(E) norm(E)^T``: the batch's pairwise cosine similarities."""
    normalized = F.normalize(embeddings.float(), p=2, dim=-1)
    return normalized @ normalized.t()


def _trainable_dtype(distiller):
    """The dtype the student's trainable weights were cast to.

    `prepare_trainable_parameters` casts them to bf16 and the backbones run in
    bf16, so fp32 projections would either fail the matmul or silently upcast
    the whole term. Falls back to the student's first parameter, then to fp32.
    """
    for param in distiller.student.parameters():
        if param.requires_grad:
            return param.dtype
    for param in distiller.student.parameters():
        return param.dtype
    return torch.float32
