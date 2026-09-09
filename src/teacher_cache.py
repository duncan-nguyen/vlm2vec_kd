"""Memmapped store of the frozen teacher's final embeddings.

The teacher is frozen and the distillation dataset applies no augmentation, so
its embedding for a sample is a pure function of the dataset index. For a
criterion that reads nothing but the final embedding -- the black-box family:
`cmtop`, `contrastive_rkd`, `universal_logit` -- the teacher forward is therefore
recomputed identically on every step of every run.

Precomputing it once (`tools/precompute_teacher_embeddings.py`) removes, per
training step:

* two forward passes of a model several times the student's size (roughly 60% of
  the step's FLOPs for a 2B teacher against a 0.5B student),
* the teacher's share of image resizing and processor work in the dataloader,
* the teacher's weights from GPU memory entirely, which is what lets the batch
  grow -- and batch size is what sets how much of the retrieval relation a
  topological loss can see.

This is the only one of the obvious speedups that survives contact with the
vendored backbones in `src/model/vlm_backbone/`; see the "What did *not* work"
section of docs/cmtop_implementation.md for the two that do not.

Layout on disk::

    <path>/meta.json        fingerprint + shape
    <path>/embeddings.f16   raw float16 memmap, shape (num_samples, 2, dim)

Row `i` holds `[query_embedding, positive_embedding]` for dataset index `i`.
float16 rather than bfloat16 on purpose: these embeddings are produced in bf16
and are close to unit norm, so fp16 stores them with *more* mantissa than they
were computed with, and the file is the same size.

The fingerprint is the point of the meta file. A cache built with a different
teacher, subset list or resolution is silently wrong rather than loudly broken,
so :meth:`load` refuses to open one whose fingerprint does not match the run.
"""

import json
import os

import numpy as np
import torch

# Bumped to 2 when the dataloader started giving libjpeg a target size before
# decoding (src/data/images.decode_image). The pixels differ from a full-size
# decode by resampling error, which is enough to move an embedding in the last
# bits -- not enough to matter for training, but a cache is either the teacher's
# output for this pipeline or it is not. Version 1 caches are refused rather
# than silently mixed with version 2 embeddings.
FORMAT_VERSION = 2
_EMBEDDINGS = "embeddings.f16"
_META = "meta.json"
_DTYPE = np.float16

# Every setting that changes what the teacher would produce for a given dataset
# index. Anything not listed here is assumed not to affect the embeddings.
_MODEL_FINGERPRINT_FIELDS = (
    "teacher_model_name",
    "teacher_checkpoint_path",
    "teacher_backbone",
    "teacher_pooling",
    "teacher_normalize",
    "teacher_lora",
    "teacher_lora_r",
)
_DATA_FINGERPRINT_FIELDS = (
    "dataset_name",
    "dataset_split",
    "percent_data",
    "image_dir",
    "image_resolution",
    "image_keep_aspect_ratio",
    "max_len",
)


def build_fingerprint(model_args, data_args=None):
    """Everything that would change the teacher's output for a dataset index.

    With `data_args` omitted only the teacher half is described, which
    :meth:`TeacherEmbeddingCache.load` will then check on its own -- a partial
    fingerprint validates a subset of the keys rather than nothing.
    """
    fingerprint = {"format_version": FORMAT_VERSION}
    for field in _MODEL_FINGERPRINT_FIELDS:
        fingerprint[field] = _jsonable(getattr(model_args, field, None))
    if data_args is not None:
        for field in _DATA_FINGERPRINT_FIELDS:
            fingerprint[field] = _jsonable(getattr(data_args, field, None))
        # Order of the subset list changes the index -> sample mapping, so it is
        # part of the identity rather than something to normalise away.
        fingerprint["subset_name"] = _jsonable(getattr(data_args, "subset_name", None))
    return fingerprint


def _jsonable(value):
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class TeacherEmbeddingCacheMismatch(RuntimeError):
    """The cache on disk was not built for this run."""


class TeacherEmbeddingCache:
    """Read/write access to one cache directory."""

    def __init__(self, path, meta, mode="r"):
        self.path = path
        self.meta = meta
        self.num_samples = int(meta["num_samples"])
        self.dim = int(meta["dim"])
        self._array = np.memmap(
            os.path.join(path, _EMBEDDINGS),
            dtype=_DTYPE,
            mode=mode,
            shape=(self.num_samples, 2, self.dim),
        )

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def create(cls, path, num_samples, dim, fingerprint):
        """Allocate an empty cache. Overwrites whatever was there."""
        os.makedirs(path, exist_ok=True)
        meta = dict(fingerprint)
        meta.update(
            {"num_samples": int(num_samples), "dim": int(dim), "complete": False}
        )
        # The data file is created first and the meta marked complete only at
        # close(), so an interrupted build cannot be mistaken for a usable cache.
        np.memmap(
            os.path.join(path, _EMBEDDINGS),
            dtype=_DTYPE,
            mode="w+",
            shape=(int(num_samples), 2, int(dim)),
        ).flush()
        with open(os.path.join(path, _META), "w") as f:
            json.dump(meta, f, indent=2)
        return cls(path, meta, mode="r+")

    @classmethod
    def load(cls, path, fingerprint=None):
        """Open an existing cache, refusing one that does not match `fingerprint`."""
        meta_path = os.path.join(path, _META)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"no teacher embedding cache at {path!r} (missing {_META}); build "
                f"one with tools/precompute_teacher_embeddings.py"
            )
        with open(meta_path) as f:
            meta = json.load(f)

        if not meta.get("complete", False):
            raise TeacherEmbeddingCacheMismatch(
                f"the cache at {path!r} is incomplete -- a previous build did not "
                f"finish. Delete it and rebuild."
            )
        if fingerprint is not None:
            differences = {
                key: (meta.get(key), value)
                for key, value in fingerprint.items()
                if meta.get(key) != value
            }
            if differences:
                lines = "\n".join(
                    f"    {k}: cache has {have!r}, this run wants {want!r}"
                    for k, (have, want) in sorted(differences.items())
                )
                raise TeacherEmbeddingCacheMismatch(
                    f"the teacher embedding cache at {path!r} was built for a "
                    f"different configuration:\n{lines}\n"
                    f"Rebuild it, or point --teacher_embedding_cache elsewhere."
                )
        return cls(path, meta, mode="r")

    def close(self, mark_complete=False):
        self._array.flush()
        if mark_complete:
            self.meta["complete"] = True
            with open(os.path.join(self.path, _META), "w") as f:
                json.dump(self.meta, f, indent=2)

    # ------------------------------------------------------------------- io

    def write(self, sample_ids, qry, pos):
        """Store one batch. `qry`/`pos` are (B, dim); `sample_ids` is (B,)."""
        ids = np.asarray(_to_numpy(sample_ids), dtype=np.int64)
        source = np.stack([_to_numpy(qry), _to_numpy(pos)], axis=1)
        rows = source.astype(_DTYPE)
        if rows.shape[0] != ids.shape[0]:
            raise ValueError(f"{ids.shape[0]} ids against {rows.shape[0]} embeddings")
        if rows.shape[-1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {rows.shape[-1]}")
        # numpy reads a negative index from the end, so a placeholder id would
        # quietly overwrite the last sample instead of failing.
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_samples):
            raise IndexError(
                f"sample ids must be in [0, {self.num_samples}); got "
                f"[{int(ids.min())}, {int(ids.max())}]. A negative id is a "
                f"placeholder for a row with no dataset index, which cannot be "
                f"cached."
            )
        # float16 tops out at 65504 while the bf16 the teacher computes in goes
        # far higher, so an unnormalised embedding with a large activation would
        # be stored as inf and poison every distance computed from it.
        overflowed = np.isfinite(source) & ~np.isfinite(rows)
        if overflowed.any():
            worst = np.abs(source[overflowed]).max()
            raise ValueError(
                f"{int(overflowed.sum())} value(s) overflow float16 (largest "
                f"{worst:g} against a limit of 65504). Teacher embeddings this "
                f"large usually mean --teacher_normalize was off; the cache "
                f"format cannot store them."
            )
        self._array[ids] = rows

    def get(self, sample_ids, device=None, dtype=None):
        """Fetch a batch as ``(qry, pos)`` tensors.

        `sample_ids` may be a tensor on any device; the lookup itself is a fancy
        index into the memmap, so only the requested rows are touched.
        """
        ids = np.asarray(_to_numpy(sample_ids), dtype=np.int64)
        if ids.size and ids.min() < 0:
            raise IndexError(
                f"{int((ids < 0).sum())} row(s) in this batch carry no dataset "
                f"index (DistillationCollator.PLACEHOLDER_SAMPLE_ID), so there is "
                f"no cached teacher embedding for them. That means the dataset "
                f"emitted a row whose text/image pairs were all rejected."
            )
        if ids.size and ids.max() >= self.num_samples:
            raise IndexError(
                f"sample id {int(ids.max())} out of range for a cache of "
                f"{self.num_samples} rows. The cache was probably built from a "
                f"different dataset than the one being trained on."
            )
        rows = torch.from_numpy(np.ascontiguousarray(self._array[ids]))
        if device is not None:
            rows = rows.to(device, non_blocking=True)
        if dtype is not None:
            rows = rows.to(dtype)
        return rows[:, 0], rows[:, 1]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return (
            x.detach().float().cpu().numpy()
            if x.is_floating_point()
            else x.detach().cpu().numpy()
        )
    return np.asarray(x)
