"""Distillation criteria.

One method = one entry in :data:`src.criterions.registry.CRITERIONS`. See
docs/adding_a_method.md.

This module deliberately imports no criterion at import time. `span_propose*`
pull in spacy, numba and tslearn; a CMTop or EM-KD run has no use for any of
them and used to pay both the import time and the hard dependency. The module
for the selected `--kd_loss_type` is imported when the criterion is built.
"""

from src.criterions.base import (
    CriterionContext,
    DistillCriterion,
    gather_no_grad,
    gather_with_grad,
    in_batch_contrastive_loss,
    pooled,
    project_teacher,
)
from src.criterions.registry import (
    CRITERIONS,
    CriterionSpec,
    attention_needs,
    build_criterion,
    get_spec,
    needs_tokenizer,
    supports_teacher_cache,
)

__all__ = [
    "CRITERIONS",
    "CriterionContext",
    "CriterionSpec",
    "DistillCriterion",
    "attention_needs",
    "build_criterion",
    "gather_no_grad",
    "gather_with_grad",
    "get_spec",
    "in_batch_contrastive_loss",
    "needs_tokenizer",
    "pooled",
    "project_teacher",
    "supports_teacher_cache",
]


def __getattr__(name):
    """Keep `from src.criterions import EMKDLoss` working, still lazily.

    Older code (and notebooks) import criterion classes straight off the
    package. Resolving them through the registry means that keeps working
    without importing all eleven modules.
    """
    for spec in CRITERIONS.values():
        if spec.cls == name:
            return spec.load()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
