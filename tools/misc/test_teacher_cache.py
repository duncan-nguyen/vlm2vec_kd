"""Self-checks for the teacher embedding cache (`src/teacher_cache.py`).

Run from the repo root: `python tools/misc/test_teacher_cache.py`. Needs only
torch and numpy -- no model download, no GPU, no dataset.

The cache is the one place where a mistake is silent rather than loud: a stale
or mismatched file produces embeddings that are wrong but perfectly well-shaped.
So most of what is checked here is refusal, not round-tripping.
"""

import json
import os as _os
import shutil
import sys as _sys
import tempfile

_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

from types import SimpleNamespace

import numpy as np
import torch

from src.teacher_cache import (
    TeacherEmbeddingCache,
    TeacherEmbeddingCacheMismatch,
    build_fingerprint,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _model_args(**overrides):
    base = dict(
        teacher_model_name="raghavlite/B3_Qwen2_2B",
        teacher_checkpoint_path=None,
        teacher_backbone="qwen2_vl",
        teacher_pooling="eos",
        teacher_normalize=True,
        teacher_lora=True,
        teacher_lora_r=8,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _data_args(**overrides):
    base = dict(
        dataset_name="TIGER-Lab/MMEB-train",
        dataset_split="original",
        subset_name=["ImageNet_1K", "N24News"],
        percent_data=1.0,
        image_dir="./vlm2vec_train/MMEB-train",
        image_resolution="448",
        max_len=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def main():
    torch.manual_seed(0)
    root = tempfile.mkdtemp(prefix="cmtop_cache_test_")
    try:
        n, dim = 37, 16
        path = _os.path.join(root, "cache")
        fingerprint = build_fingerprint(_model_args(), _data_args())

        qry = torch.randn(n, dim)
        pos = torch.randn(n, dim)

        cache = TeacherEmbeddingCache.create(path, n, dim, fingerprint)
        # Written out of order and in chunks, the way sharded ranks do it.
        order = torch.randperm(n)
        for start in range(0, n, 8):
            ids = order[start : start + 8]
            cache.write(ids, qry[ids], pos[ids])

        check(
            "an unfinished cache is refused",
            _raises(lambda: TeacherEmbeddingCache.load(path), TeacherEmbeddingCacheMismatch),
        )
        cache.close(mark_complete=True)

        loaded = TeacherEmbeddingCache.load(path, fingerprint=fingerprint)
        got_qry, got_pos = loaded.get(torch.arange(n))
        check(
            "round-trips every row to float16 precision",
            torch.allclose(got_qry.float(), qry, atol=1e-3)
            and torch.allclose(got_pos.float(), pos, atol=1e-3),
            f"max err {max((got_qry.float() - qry).abs().max(), (got_pos.float() - pos).abs().max()):.2e}",
        )

        ids = torch.tensor([5, 5, 31, 0])
        sub_qry, sub_pos = loaded.get(ids)
        check(
            "a gather keeps the requested order, repeats included",
            torch.allclose(sub_qry.float(), qry[ids], atol=1e-3)
            and torch.allclose(sub_pos.float(), pos[ids], atol=1e-3),
        )

        out = loaded.get(torch.arange(4), dtype=torch.bfloat16)
        check("serves the dtype the student asks for", out[0].dtype == torch.bfloat16)

        check(
            "an out-of-range sample id is an error, not a wrong row",
            _raises(lambda: loaded.get(torch.tensor([n])), IndexError),
        )
        check(
            "a placeholder id is refused rather than read as row -1",
            _raises(lambda: loaded.get(torch.tensor([0, -1])), IndexError),
        )

        overflow = TeacherEmbeddingCache.create(
            _os.path.join(root, "overflow"), 2, dim, fingerprint
        )
        big = torch.full((1, dim), 1e6)
        check(
            "an embedding too large for float16 is refused, not stored as inf",
            _raises(lambda: overflow.write(torch.tensor([0]), big, big), ValueError),
        )
        check(
            "a normal-magnitude embedding still writes",
            overflow.write(torch.tensor([0]), qry[:1], pos[:1]) is None,
        )
        check(
            "writing a placeholder id is refused, not folded onto the last row",
            _raises(
                lambda: overflow.write(torch.tensor([-1]), qry[:1], pos[:1]), IndexError
            ),
        )
        check(
            "writing past the end is refused",
            _raises(
                lambda: overflow.write(torch.tensor([2]), qry[:1], pos[:1]), IndexError
            ),
        )

        # --- the refusals that matter -------------------------------------
        for label, changed in (
            ("teacher", build_fingerprint(_model_args(teacher_model_name="other/model"), _data_args())),
            ("pooling", build_fingerprint(_model_args(teacher_pooling="mean"), _data_args())),
            ("subset list", build_fingerprint(_model_args(), _data_args(subset_name=["N24News"]))),
            ("subset order", build_fingerprint(_model_args(), _data_args(subset_name=["N24News", "ImageNet_1K"]))),
            ("resolution", build_fingerprint(_model_args(), _data_args(image_resolution="336"))),
            ("percent_data", build_fingerprint(_model_args(), _data_args(percent_data=0.5))),
        ):
            check(
                f"refuses a cache built with a different {label}",
                _raises(
                    lambda c=changed: TeacherEmbeddingCache.load(path, fingerprint=c),
                    TeacherEmbeddingCacheMismatch,
                ),
            )

        check(
            "a partial fingerprint still checks the keys it has",
            TeacherEmbeddingCache.load(
                path, fingerprint=build_fingerprint(_model_args())
            ).num_samples == n
            and _raises(
                lambda: TeacherEmbeddingCache.load(
                    path, fingerprint=build_fingerprint(_model_args(teacher_lora_r=64))
                ),
                TeacherEmbeddingCacheMismatch,
            ),
        )

        check(
            "a missing cache says how to build one",
            _raises(
                lambda: TeacherEmbeddingCache.load(_os.path.join(root, "nope")),
                FileNotFoundError,
            ),
        )

        with open(_os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        check(
            "meta records the identity, not just the shape",
            meta["teacher_model_name"] == "raghavlite/B3_Qwen2_2B"
            and meta["subset_name"] == ["ImageNet_1K", "N24News"]
            and meta["num_samples"] == n,
        )

        size = _os.path.getsize(_os.path.join(path, "embeddings.f16"))
        check(
            "the file is exactly (N, 2, dim) float16",
            size == n * 2 * dim * 2,
            f"{size} bytes",
        )

        # --- the criterion runs against the cache ------------------------
        criterion_against_cache(loaded, qry, pos, dim)
        collator_checks()
        registry_checks()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


def registry_checks():
    """Every cache-compatible criterion must actually go through encode_teacher.

    Declaring a criterion in TEACHER_EMBEDDING_ONLY_CRITERIONS while it still
    calls `distiller.teacher` directly would fail only at runtime, with
    `AttributeError: 'NoneType'` several minutes into a run.
    """
    import re

    root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    init = open(_os.path.join(root, "src", "criterions", "__init__.py")).read()

    declared = set(
        re.findall(r'"([^"]+)"',
                   re.search(r"TEACHER_EMBEDDING_ONLY_CRITERIONS = \{(.*?)\}", init, re.S).group(1))
    )
    registry = dict(re.findall(r'"([\w.]+)": (\w+),', init.split("criterion_list = {")[1].split("}")[0]))
    class_to_file = dict(re.findall(r"from \.(\w+) import (\w+)", init))
    class_to_file = {cls: mod for mod, cls in class_to_file.items()}

    for name in sorted(declared):
        module = class_to_file.get(registry.get(name, ""), None)
        if module is None:
            check(f"{name} is a registered criterion", False)
            continue
        source = open(_os.path.join(root, "src", "criterions", f"{module}.py")).read()
        check(
            f"{name} reads the teacher through encode_teacher",
            "distiller.encode_teacher" in source
            and "distiller.teacher" not in source.replace("distiller.teacher_cache", ""),
        )


def collator_checks():
    """The dataset -> collator contract that keys the cache.

    `DistillationCollator` is imported through a stub `src.distiller` module: the
    real one pulls in peft, qwen_vl_utils and the vendored backbones, none of
    which this check needs.
    """
    import importlib.util

    root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    source = open(_os.path.join(root, "src", "distiller.py")).read()
    start = source.index("class DistillationCollator")
    end = source.index("class DistillationDataset")
    namespace = {
        "torch": torch, "Optional": None, "ProcessorMixin": object,
        "ModelArguments": object, "DataArguments": object, "TrainingArguments": object,
        "process_vlm_inputs_fns": {"stub": lambda inputs, processor, max_length: dict(inputs)},
    }
    exec(compile(source[start:end], "distiller_collator", "exec"), namespace)
    Collator = namespace["DistillationCollator"]

    def row(idx, n=1):
        return {
            "sample_ids": [idx] * n,
            "student_query_text": ["q"] * n, "student_query_image": [None] * n,
            "student_pos_text": ["p"] * n, "student_pos_image": [None] * n,
            "teacher_query_text": ["q"] * n, "teacher_query_image": [None] * n,
            "teacher_pos_text": ["p"] * n, "teacher_pos_image": [None] * n,
        }

    model_args = SimpleNamespace(model_backbone="stub", teacher_backbone="stub",
                                 teacher_embedding_cache=None)
    data_args = SimpleNamespace(max_len=None)
    common = dict(student_processor=None, teacher_processor=None, model_args=model_args,
                  data_args=data_args, training_args=None)

    batch = Collator(**common)([row(3), row(7), row(11)])
    check(
        "collator emits one sample id per row, in order",
        batch["sample_ids"].tolist() == [3, 7, 11],
        str(batch["sample_ids"].tolist()),
    )
    check(
        "both sides are processed by default",
        "student_inputs" in batch and "teacher_inputs" in batch,
    )

    cached_args = SimpleNamespace(model_backbone="stub", teacher_backbone="stub",
                                  teacher_embedding_cache="/some/cache")
    cached = Collator(**{**common, "model_args": cached_args})([row(1), row(2)])
    check(
        "a teacher embedding cache turns teacher processing off automatically",
        "teacher_inputs" not in cached and "student_inputs" in cached
        and cached["sample_ids"].tolist() == [1, 2],
    )

    precompute = Collator(**common, include_student=False)([row(4), row(5)])
    check(
        "the precompute pass skips the student side",
        "student_inputs" not in precompute and "teacher_inputs" in precompute,
    )

    # Counts, not uniqueness: a row expanding to several examples still yields
    # matching counts, so that case is caught at cache-write time instead (the
    # precompute tool requires every dataset index exactly once).
    expanded = Collator(**common)([row(0, n=2)])
    check(
        "an expanded row still produces one id per example",
        expanded["sample_ids"].tolist() == [0, 0]
        and len(expanded["student_inputs"]["qry"]["text"]) == 2,
    )

    # `_get_batch_inputs` substitutes a blank row for an example with nothing in
    # it, so `_get_sample_ids` has to substitute an id or every later row shifts.
    placeholder = Collator.PLACEHOLDER_SAMPLE_ID
    empty_row = {k: ([] if isinstance(v, list) else v) for k, v in row(9).items()}
    batch = Collator(**common)([row(3), empty_row, row(5)])
    check(
        "a row with no usable pairs keeps the ids aligned with the rows",
        batch["sample_ids"].tolist() == [3, placeholder, 5]
        and len(batch["student_inputs"]["qry"]["text"]) == 3,
        str(batch["sample_ids"].tolist()),
    )
    check(
        "an example that arrives empty also gets a placeholder",
        Collator(**common)([row(3), {}])["sample_ids"].tolist() == [3, placeholder],
    )
    desynced = row(0)
    desynced["sample_ids"] = []
    check(
        "ids that do not match the row count fall back to placeholders",
        Collator(**common)([desynced])["sample_ids"].tolist() == [placeholder],
    )


def criterion_against_cache(cache, qry, pos, dim):
    """CMTop with no teacher model at all, reading embeddings from the memmap."""
    import importlib.util

    root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location(
        "cmtop_criterion", _os.path.join(root, "src", "criterions", "cross_modal_topology.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _Student(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.proj = torch.nn.Linear(in_dim, out_dim)

        def encode_input(self, batch):
            return self.proj(batch["x"]), None, None, None

        def compute_similarity(self, q, p):
            return q @ p.t()

    class _CachedDistiller:
        """Mirrors Distiller.encode_teacher's dispatch with teacher=None."""

        def __init__(self, student, cache, projectors):
            self.student = student
            self.teacher = None
            self.teacher_cache = cache
            self.temperature = 0.02
            self.projectors = projectors

        def encode_teacher(self, input_data, side, dtype=None):
            qry, pos = self.teacher_cache.get(
                input_data["sample_ids"],
                device=next(self.student.parameters()).device,
                dtype=dtype,
            )
            return qry if side == "qry" else pos

    batch, in_dim, student_dim = 12, 24, 8
    student = _Student(in_dim, student_dim)
    projectors = torch.nn.ModuleDict({"t2s": torch.nn.Linear(dim, student_dim)})
    distiller = _CachedDistiller(student, cache, projectors)

    ids = torch.arange(batch)
    inputs = {
        "sample_ids": ids,
        "student_inputs": {
            "qry": {"x": torch.randn(batch, in_dim)},
            "pos": {"x": torch.randn(batch, in_dim)},
        },
        # Deliberately absent: with a cache there is no teacher_inputs key, and
        # the criterion must not reach for one.
    }
    args = SimpleNamespace(
        kd_weight=1.0, cmtop_weight=1.0, cmtop_mode="cross_modal", cmtop_h0_weight=1.0,
        cmtop_h1_weight=0.1, cmtop_h1_topk=0, cmtop_endpoint_kd="cosine",
        cmtop_geometry_weight=0.5, cmtop_normalize_scale=False, cmtop_reduction="mean",
    )
    out = module.CrossModalTopologyLoss(args)(distiller, inputs)
    check(
        "CMTop runs with cached embeddings and no teacher model",
        all(torch.isfinite(v).all() for v in out.values())
        and float(out["cmtop_loss"].detach()) > 0,
    )
    out["loss"].backward()
    check(
        "the student still gets a gradient on the cached path",
        all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()),
    )

    # The cached teacher embeddings must be the ones the criterion saw.
    cached_qry, _ = cache.get(ids, dtype=torch.float32)
    check(
        "the criterion reads the rows for its own sample ids",
        torch.allclose(cached_qry, qry[ids], atol=1e-3),
    )


if __name__ == "__main__":
    raise SystemExit(main())
