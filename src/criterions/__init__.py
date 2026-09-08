from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .cross_modal_topology import CrossModalTopologyLoss
from .em_kd import EMKDLoss
from .em_kd_llava_ov import EMKDLLavaLoss
from .emo_loss import EMOLoss
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .propose_with_proj import ProposalLossWithProj
from .span_propose import SpanProposeCriterion
from .span_propose_attn import SpanProposeCriterionWeighted
from .span_propose_attn_only_phrase import SpanProposeCriterionWeightedOnlyPhrase
from .universal_logit_distillation import UniversalLogitDistillation

criterion_list = {
    "contrastive_rkd": ContrastiveLossWithRKD,
    "proposal_dtw": ProposalLossWithDTW,
    "universal_logit": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "emo_loss": EMOLoss,
    "em_kd": EMKDLoss,
    "em_kd_llava_ov": EMKDLLavaLoss,
    "span_propose": SpanProposeCriterion,
    "span_propose_attn": SpanProposeCriterionWeighted,
    "span_propose_attn_only_phrase": SpanProposeCriterionWeightedOnlyPhrase,
    "cmtop": CrossModalTopologyLoss,
}


def build_criterion(args):
    if args.kd_loss_type not in criterion_list:
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)


# Which side's attention matrices each criterion actually consumes, as
# (student, teacher). Asking a model for attentions forces the eager attention
# kernel and materialises a (B, heads, L, L) tensor per layer -- for the student
# that cost is paid in the backward pass too. Everything marked False here can
# run on SDPA instead. Keep this table in sync when adding a criterion; the
# default for an unknown name is (True, True), i.e. the old behaviour.
CRITERION_ATTENTION_NEEDS = {
    "contrastive_rkd": (False, False),
    # CMTop is deliberately black-box: final embeddings only, both sides.
    "cmtop": (False, False),
    "universal_logit": (False, False),
    "em_kd": (False, False),
    "em_kd_llava_ov": (False, False),
    "span_propose": (False, True),
    "span_propose_attn": (False, True),
    "span_propose_attn_only_phrase": (False, True),
    "proposal_dtw": (True, True),
    "proposal_proj": (True, True),
    "emo_loss": (True, True),
}


def attention_needs(kd_loss_type):
    """Return (student_needs_attentions, teacher_needs_attentions)."""
    return CRITERION_ATTENTION_NEEDS.get(kd_loss_type, (True, True))


# Criteria that read nothing from the teacher but its final pooled embedding.
# Only these can be trained against a precomputed teacher embedding cache
# (`--teacher_embedding_cache`); the rest need hidden states or attentions,
# which are far too large to store. Every criterion listed here must obtain its
# teacher embeddings through `Distiller.encode_teacher`, not by calling
# `distiller.teacher` directly -- with a cache in use there is no teacher model.
TEACHER_EMBEDDING_ONLY_CRITERIONS = {
    "cmtop",
    "contrastive_rkd",
    "universal_logit",
}


def supports_teacher_cache(kd_loss_type):
    """Whether this criterion can run without a live teacher model."""
    return kd_loss_type in TEACHER_EMBEDDING_ONLY_CRITERIONS
