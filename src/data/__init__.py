"""Data loading for training and evaluation.

Nothing is imported eagerly. `from src.data.dataset import *` used to run at
package import, which pulls in `src.model.processor` and through it every
vendored backbone in `src/model/vlm_backbone/` -- so importing
`src.data.images`, a module whose only dependency is Pillow, needed a
transformers version compatible with all of ColPali, PaliGemma, InternVL and
Phi-3-V. `load_mmeb_dataset` still resolves off this package; it just costs
those imports only when someone asks for it.
"""

__all__ = ["load_mmeb_dataset"]


def __getattr__(name):
    if name == "load_mmeb_dataset":
        from src.data.dataset import load_mmeb_dataset

        return load_mmeb_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
