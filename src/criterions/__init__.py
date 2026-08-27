from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .emo_loss import EMOLoss
from .em_kd import EMKDLoss
from .em_kd_llava_ov import EMKDLLavaLoss
from .span_propose import SpanProposeCriterion
from .span_propose_attn import SpanProposeCriterionWeighted
from .span_propose_attn_only_phrase import SpanProposeCriterionWeightedOnlyPhrase

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
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)


# Which side's attention matrices each criterion actually consumes, as
# (student, teacher). Asking a model for attentions forces the eager attention
# kernel and materialises a (B, heads, L, L) tensor per layer -- for the student
# that cost is paid in the backward pass too. Everything marked False here can
# run on SDPA instead. Keep this table in sync when adding a criterion; the
# default for an unknown name is (True, True), i.e. the old behaviour.
CRITERION_ATTENTION_NEEDS = {
    "contrastive_rkd":               (False, False),
    "universal_logit":               (False, False),
    "em_kd":                         (False, False),
    "em_kd_llava_ov":                (False, False),
    "span_propose":                  (False, True),
    "span_propose_attn":             (False, True),
    "span_propose_attn_only_phrase": (False, True),
    "proposal_dtw":                  (True, True),
    "proposal_proj":                 (True, True),
    "emo_loss":                      (True, True),
}


def attention_needs(kd_loss_type):
    """Return (student_needs_attentions, teacher_needs_attentions)."""
    return CRITERION_ATTENTION_NEEDS.get(kd_loss_type, (True, True))