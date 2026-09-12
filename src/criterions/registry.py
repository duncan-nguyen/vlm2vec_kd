"""The one table that describes every distillation method.

Adding a method used to mean touching four places in `src/criterions/__init__.py`
(the import, `criterion_list`, `CRITERION_ATTENTION_NEEDS`,
`TEACHER_EMBEDDING_ONLY_CRITERIONS`) plus a hard-coded name list in
`Distiller.forward`. Each of those was a separate opportunity to forget, and
three of the four fail silently: a criterion missing from the attention table
just runs slower, one missing from the teacher-cache set just refuses a flag,
and one missing from the tokenizer list gets a `TypeError` several minutes in.

Now a method is one :class:`CriterionSpec` in :data:`CRITERIONS`. Everything the
rest of the codebase asks about a criterion is derived from it.

The class is named rather than imported so that `import src.criterions` stays
cheap and dependency-light: `span_propose*` pull in spacy, numba and tslearn,
which a CMTop run has no use for. The module is imported the first time that
particular criterion is actually built.
"""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class CriterionSpec:
    """Everything the training stack needs to know about one method.

    Attributes:
        name: the value of ``--kd_loss_type``.
        module: dotted path of the module holding the class. Imported lazily.
        cls: class name inside that module.
        student_attentions: the criterion reads the *student's* attention
            matrices. Asking for them pins the backbone to the eager attention
            kernel and keeps a ``(B, heads, L, L)`` tensor per layer alive
            through the backward pass, so this is the expensive one; leave it
            False unless the loss really indexes ``attention_matrix``.
        teacher_attentions: same for the teacher (forward only, so cheaper).
        teacher_embedding_only: the criterion reads nothing from the teacher but
            its final pooled embedding, and therefore can train against a
            precomputed cache (``--teacher_embedding_cache``) with no teacher
            model in the process at all. Such a criterion **must** obtain its
            teacher embeddings via ``distiller.encode_teacher``; reaching for
            ``distiller.teacher`` directly crashes when the cache is in use.
        needs_tokenizer: ``Distiller.forward`` passes ``tokenizer=`` to it.
        summary: one line, shown by ``--kd_loss_type list``.
    """

    name: str
    module: str
    cls: str
    student_attentions: bool = True
    teacher_attentions: bool = True
    teacher_embedding_only: bool = False
    needs_tokenizer: bool = False
    summary: str = ""

    def load(self):
        """Import the module and return the criterion class."""
        return getattr(import_module(self.module), self.cls)


# ---------------------------------------------------------------- the table
#
# To add a method: write the criterion (see docs/adding_a_method.md), add one
# entry here, and nothing else. The defaults are the conservative ones -- a
# criterion that asks for attentions on both sides runs correctly but slowly --
# so an entry that is merely incomplete costs speed, never correctness.

_SPECS = (
    CriterionSpec(
        name="contrastive_rkd",
        module="src.criterions.contrastive_loss_with_RKD",
        cls="ContrastiveLossWithRKD",
        student_attentions=False,
        teacher_attentions=False,
        teacher_embedding_only=True,
        summary="Relational KD (distance + angle) on the final embeddings.",
    ),
    CriterionSpec(
        name="cmtop",
        module="src.criterions.cross_modal_topology",
        cls="CrossModalTopologyLoss",
        student_attentions=False,
        teacher_attentions=False,
        teacher_embedding_only=True,
        summary="CM-Merge: correspondence-aware multiscale connectivity of the retrieval relation.",
    ),
    CriterionSpec(
        name="talas",
        module="src.criterions.talas",
        cls="TALASLoss",
        student_attentions=False,
        teacher_attentions=False,
        teacher_embedding_only=True,
        summary="TALAS: teacher-anchored top layers + layer-aligned self-distillation.",
    ),
    CriterionSpec(
        name="universal_logit",
        module="src.criterions.universal_logit_distillation",
        cls="UniversalLogitDistillation",
        student_attentions=False,
        teacher_attentions=False,
        teacher_embedding_only=True,
        summary="Universal logit distillation on the pooled embeddings.",
    ),
    CriterionSpec(
        name="em_kd",
        module="src.criterions.em_kd",
        cls="EMKDLoss",
        student_attentions=False,
        teacher_attentions=False,
        summary="EM-KD: Hungarian-matched vision tokens + vision/language affinity.",
    ),
    CriterionSpec(
        name="em_kd_llava_ov",
        module="src.criterions.em_kd_llava_ov",
        cls="EMKDLLavaLoss",
        student_attentions=False,
        teacher_attentions=False,
        summary="EM-KD variant for the LLaVA-OneVision token layout.",
    ),
    CriterionSpec(
        name="emo_loss",
        module="src.criterions.emo_loss",
        cls="EMOLoss",
        summary="EMO: optimal-transport distillation over hidden states.",
    ),
    CriterionSpec(
        name="proposal_dtw",
        module="src.criterions.proposal_loss_with_DTW",
        cls="ProposalLossWithDTW",
        summary="Span proposals matched with soft-DTW.",
    ),
    CriterionSpec(
        name="proposal_proj",
        module="src.criterions.propose_with_proj",
        cls="ProposalLossWithProj",
        summary="Span proposals compared through the layer projectors.",
    ),
    CriterionSpec(
        name="span_propose",
        module="src.criterions.span_propose",
        cls="SpanProposeCriterion",
        student_attentions=False,
        needs_tokenizer=True,
        summary="HieRD span proposals (teacher attentions only).",
    ),
    CriterionSpec(
        name="span_propose_attn",
        module="src.criterions.span_propose_attn",
        cls="SpanProposeCriterionWeighted",
        student_attentions=False,
        needs_tokenizer=True,
        summary="HieRD span proposals weighted by teacher attention.",
    ),
    CriterionSpec(
        name="span_propose_attn_only_phrase",
        module="src.criterions.span_propose_attn_only_phrase",
        cls="SpanProposeCriterionWeightedOnlyPhrase",
        student_attentions=False,
        needs_tokenizer=True,
        summary="HieRD span proposals, phrase spans only.",
    ),
)

CRITERIONS = {spec.name: spec for spec in _SPECS}

if len(CRITERIONS) != len(_SPECS):  # pragma: no cover - guards a typo in the table
    raise RuntimeError("duplicate --kd_loss_type name in src/criterions/registry.py")


# ------------------------------------------------------------------ lookups


def get_spec(kd_loss_type):
    """The spec for ``--kd_loss_type``, with the available names on failure."""
    try:
        return CRITERIONS[kd_loss_type]
    except KeyError:
        raise ValueError(
            f"unknown --kd_loss_type {kd_loss_type!r}; available methods:\n"
            + "\n".join(
                f"    {name:<30} {spec.summary}"
                for name, spec in sorted(CRITERIONS.items())
            )
        ) from None


def build_criterion(args):
    """Instantiate the criterion named by ``args.kd_loss_type``."""
    return get_spec(args.kd_loss_type).load()(args)


def attention_needs(kd_loss_type):
    """``(student_needs_attentions, teacher_needs_attentions)``.

    Unknown names get the conservative answer rather than raising: this is
    consulted from `Distiller._configure_attention`, which runs before anything
    that would give a better error message.
    """
    spec = CRITERIONS.get(kd_loss_type)
    if spec is None:
        return (True, True)
    return (spec.student_attentions, spec.teacher_attentions)


def supports_teacher_cache(kd_loss_type):
    """Whether this criterion can run with no teacher model in the process."""
    spec = CRITERIONS.get(kd_loss_type)
    return bool(spec and spec.teacher_embedding_only)


def needs_tokenizer(kd_loss_type):
    """Whether ``Distiller.forward`` should pass ``tokenizer=`` to it."""
    spec = CRITERIONS.get(kd_loss_type)
    return bool(spec and spec.needs_tokenizer)
