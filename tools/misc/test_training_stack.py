"""Self-checks for the shared training stack. No model download, no GPU.

    python tools/misc/test_training_stack.py

Covers the pieces that `test_cmtop.py` and `test_teacher_cache.py` do not:

* the criterion registry and its lazy imports,
* `DistillCriterion`'s combination of the contrastive and KD terms,
* the image pipeline (draft decoding produces the same geometry as a full
  decode, and the aspect-ratio flag does what it says),
* the training loop's gradient accumulation, which is the part where a
  discrepancy is invisible at `--gradient_accumulation_steps 1` and silently
  rescales the learning rate above it.
"""

import os as _os
import sys as _sys

_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import shutil
import tempfile

import numpy as np
import torch
from PIL import Image
from torch import nn

FAILURES = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


# --------------------------------------------------------------- fake models


class FakeStudent(nn.Module):
    """Stands in for MMEBModel: one trainable linear layer, same interface."""

    def __init__(self, dim=8):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def encode_input(self, inputs):
        return (self.proj(inputs["x"]), None, None, None)

    def compute_similarity(self, q, p):
        return q @ p.t()


class FakeDistiller(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.student = FakeStudent(dim)
        self.projectors = None
        self.temperature = 0.02

    def encode_teacher(self, batch, side, dtype=None):
        return batch["teacher"][side].to(dtype) if dtype else batch["teacher"][side]

    def forward(self, criterion, batch):
        return criterion(self, batch)


class Args:
    kd_weight = 0.5


def make_batch(bs=4, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "student_inputs": {
            "qry": {"x": torch.randn(bs, dim, generator=g)},
            "pos": {"x": torch.randn(bs, dim, generator=g)},
        },
        "teacher": {
            "qry": torch.randn(bs, dim, generator=g),
            "pos": torch.randn(bs, dim, generator=g),
        },
    }


# ------------------------------------------------------------------- checks


def registry_checks():
    from src.criterions.registry import (
        CRITERIONS,
        attention_needs,
        get_spec,
        needs_tokenizer,
        supports_teacher_cache,
    )

    check(
        "every spec key equals its name",
        all(k == s.name for k, s in CRITERIONS.items()),
    )
    check(
        "every spec has a one-line summary",
        all(s.summary for s in CRITERIONS.values()),
    )
    check(
        "a cache-compatible criterion never asks for attentions",
        all(
            not (s.student_attentions or s.teacher_attentions)
            for s in CRITERIONS.values()
            if s.teacher_embedding_only
        ),
    )
    check(
        "unknown name gets the conservative attention answer",
        attention_needs("zzz") == (True, True),
    )
    check(
        "unknown name does not support the cache",
        supports_teacher_cache("zzz") is False,
    )
    check("unknown name needs no tokenizer", needs_tokenizer("zzz") is False)

    try:
        get_spec("zzz")
        check("unknown --kd_loss_type raises", False)
    except ValueError as exc:
        check(
            "unknown --kd_loss_type lists the available methods",
            "cmtop" in str(exc) and "em_kd" in str(exc),
        )

    # Importing the package must not drag in the heavy optional dependencies.
    before = set(_sys.modules)
    import src.criterions  # noqa: F401

    pulled = {m.split(".")[0] for m in set(_sys.modules) - before}
    check(
        "importing src.criterions pulls in no spacy/numba/tslearn",
        not ({"spacy", "numba", "tslearn"} & pulled),
        f"pulled {sorted(pulled)}" if pulled else "",
    )


def base_criterion_checks():
    from src.criterions.base import DistillCriterion

    class Simple(DistillCriterion):
        def kd_loss(self, ctx):
            return {
                "kd_loss": torch.nn.functional.mse_loss(
                    ctx.student_qry, ctx.teacher_qry
                ),
                "extra": ctx.zeros() + 7.0,
            }

    out = Simple(Args())(FakeDistiller(), make_batch())
    check(
        "loss = contrastive + kd_weight * kd_loss",
        torch.allclose(
            out["loss"], out["contrastive_loss"] + 0.5 * out["kd_loss"], atol=1e-6
        ),
    )
    check("a method's own terms are passed through for logging", "extra" in out)
    out["loss"].backward()

    class Broken(DistillCriterion):
        def kd_loss(self, ctx):
            return {"nope": ctx.zeros()}

    try:
        Broken(Args())(FakeDistiller(), make_batch())
        check("a kd_loss() without a 'kd_loss' key is rejected", False)
    except KeyError:
        check("a kd_loss() without a 'kd_loss' key is rejected", True)

    # A bare tensor is the shorthand for {"kd_loss": t}.
    class Bare(DistillCriterion):
        def kd_loss(self, ctx):
            return ctx.zeros() + 2.0

    check(
        "kd_loss() may return a bare tensor",
        float(Bare(Args())(FakeDistiller(), make_batch())["kd_loss"]) == 2.0,
    )


def image_checks():
    from src.data.images import (
        MIN_SIZE,
        decode_image,
        pad_to_min_size,
        resize_to_budget,
        resolve_target,
    )

    check(
        "named resolutions resolve",
        resolve_target("low") == 448 and resolve_target("high") == 1344,
    )
    check("a bare pixel count resolves", resolve_target("336") == 336)
    check(
        "an unparseable resolution falls back to max_dim", resolve_target("???") == 1344
    )

    tmp = tempfile.mkdtemp()
    try:
        target = 448
        for w, h in [
            (4000, 3000),
            (1600, 1200),
            (900, 900),
            (800, 600),
            (500, 375),
            (224, 224),
            (10, 40),
        ]:
            arr = np.random.default_rng(w).integers(0, 255, (h, w, 3)).astype("uint8")
            img = (
                Image.fromarray(arr)
                .resize((max(1, w // 4), max(1, h // 4)))
                .resize((w, h))
            )
            path = _os.path.join(tmp, f"{w}x{h}.jpg")
            img.save(path, quality=88)

            full = resize_to_budget(
                pad_to_min_size(Image.open(path).convert("RGB")), target
            )
            drafted = resize_to_budget(decode_image(path, target_max=target), target)
            check(
                f"draft decoding preserves the output geometry ({w}x{h})",
                full.size == drafted.size,
                f"{full.size} vs {drafted.size}",
            )
            if full.size == drafted.size:
                worst = int(
                    np.abs(np.asarray(full, int) - np.asarray(drafted, int)).max()
                )
                check(
                    f"draft decoding stays within resampling error ({w}x{h})",
                    worst <= 40,
                    f"max channel diff {worst}",
                )

        # A source already inside the budget must come back untouched, or a
        # cache built before the draft hint would disagree with one built after.
        small = Image.open(_os.path.join(tmp, "224x224.jpg")).convert("RGB")
        check(
            "an image inside the budget is not resized",
            resize_to_budget(small, target).size == (224, 224),
        )

        wide = Image.new("RGB", (1000, 500))
        check(
            "square mode squashes to the budget",
            resize_to_budget(wide, target).size == (448, 448),
        )
        check(
            "aspect mode keeps the ratio",
            resize_to_budget(wide, target, keep_aspect=True).size == (448, 224),
        )

        tiny = Image.new("RGB", (4, 4))
        check(
            "a tiny image is padded up to MIN_SIZE",
            pad_to_min_size(tiny).size == (MIN_SIZE, MIN_SIZE),
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def loss_meter_checks():
    from src.training.loop import _LossMeter

    meter = _LossMeter(torch.device("cpu"))
    meter.update({"loss": torch.tensor(1.0), "kd_loss": torch.tensor(3.0)})
    meter.update({"loss": torch.tensor(3.0), "kd_loss": torch.tensor(5.0)})
    means = meter.means()
    check(
        "running means are correct", means == {"loss": 2.0, "kd_loss": 4.0}, str(means)
    )

    # A key the criterion stopped reporting counts as zero rather than crashing.
    meter.update({"loss": torch.tensor(2.0)})
    check(
        "a missing component counts as zero",
        abs(meter.means()["kd_loss"] - 8 / 3) < 1e-6,
    )

    meter.reset()
    check("reset clears the window", meter.means() == {})
    check("an empty meter is falsy", not bool(_LossMeter(torch.device("cpu"))))


def grad_accumulation_checks():
    """N micro-batches must give the same gradient as one batch of N times the size.

    This is the check that would have caught the double division in the old
    Accelerate path: `accelerator.backward` divides by
    `gradient_accumulation_steps` and the loop divided again, so gradients came
    out at 1/accum^2 and the effective learning rate was `accum` times too
    small. Invisible at accum=1, which is what every launcher used.
    """
    from src.criterions.base import DistillCriterion
    from src.training.loop import DistillTrainer

    class Simple(DistillCriterion):
        def kd_loss(self, ctx):
            return torch.nn.functional.mse_loss(ctx.student_qry, ctx.teacher_qry)

    torch.manual_seed(0)
    micro = [make_batch(bs=4, seed=s) for s in range(4)]

    def grads_from_accumulation(grad_accum):
        torch.manual_seed(0)
        distiller = FakeDistiller()
        criterion = Simple(Args())
        optimizer = torch.optim.SGD(distiller.student.parameters(), lr=0.0)

        class TA:
            gradient_accumulation_steps = grad_accum
            logging_steps = 1000
            max_grad_norm = 0.0
            num_train_epochs = 1
            save_strategy = "no"
            output_dir = ""

        class Sched:
            def step(self):
                pass

            def get_last_lr(self):
                return [0.0]

        trainer = DistillTrainer(
            distiller=distiller,
            criterion=criterion,
            dataloader=micro,
            optimizer=optimizer,
            lr_scheduler=Sched(),
            model_args=None,
            training_args=TA(),
            device=torch.device("cpu"),
        )
        # lr=0 so the parameters never move; the last window's gradients are
        # zeroed by the step, so capture them from a hook instead.
        seen = []
        original = trainer._optimizer_step

        def spy(window_batches=None):
            seen.append(
                torch.cat(
                    [p.grad.reshape(-1).clone() for p in distiller.student.parameters()]
                )
            )
            original(window_batches)

        trainer._optimizer_step = spy
        trainer.run_epoch(0)
        return seen

    one_at_a_time = grads_from_accumulation(1)
    accumulated = grads_from_accumulation(4)

    check(
        "accum=1 steps once per micro-batch",
        len(one_at_a_time) == 4,
        f"{len(one_at_a_time)} steps",
    )
    check(
        "accum=4 steps once per window",
        len(accumulated) == 1,
        f"{len(accumulated)} steps",
    )
    if one_at_a_time and accumulated:
        expected = sum(one_at_a_time) / 4.0
        check(
            "accumulated gradient is the mean of the per-micro-batch gradients",
            torch.allclose(accumulated[0], expected, atol=1e-6),
            f"max diff {float((accumulated[0] - expected).abs().max()):.2e}",
        )

    # A trailing partial window must be flushed, not carried into the next epoch.
    trailing = grads_from_accumulation(3)
    check(
        "a trailing partial accumulation window still steps",
        len(trailing) == 2,
        f"{len(trailing)} steps for 4 batches at accum=3",
    )


def max_steps_checks():
    """The custom loop must honour the standard HF ``--max_steps`` flag."""
    from src.criterions.base import DistillCriterion
    from src.training.entrypoint import training_horizon
    from src.training.loop import DistillTrainer

    class Simple(DistillCriterion):
        def kd_loss(self, ctx):
            return torch.nn.functional.mse_loss(ctx.student_qry, ctx.teacher_qry)

    class TA:
        gradient_accumulation_steps = 1
        logging_steps = 1000
        max_grad_norm = 0.0
        max_steps = 2
        num_train_epochs = 9
        save_strategy = "no"
        output_dir = ""

    class Sched:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

        def get_last_lr(self):
            return [0.0]

    distiller = FakeDistiller()
    scheduler = Sched()
    trainer = DistillTrainer(
        distiller=distiller,
        criterion=Simple(Args()),
        dataloader=[make_batch(seed=s) for s in range(5)],
        optimizer=torch.optim.SGD(distiller.student.parameters(), lr=0.0),
        lr_scheduler=scheduler,
        model_args=None,
        training_args=TA(),
        device=torch.device("cpu"),
    )
    trainer.run_epoch(0, max_optimizer_steps=TA.max_steps)
    check(
        "max_steps stops after exactly that many optimizer updates",
        trainer.global_step == TA.max_steps and scheduler.steps == TA.max_steps,
        f"global_step={trainer.global_step}, scheduler_steps={scheduler.steps}",
    )
    check(
        "max_steps overrides a longer epoch schedule",
        training_horizon(TA(), steps_per_epoch=5) == (1, 2),
    )


def dataloader_checks():
    from src.training.dataloader import (
        DEFAULT_NUM_WORKERS,
        TaskHomogeneousSampler,
        build_train_dataloader,
    )

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, i):
            return i

    class TA:
        per_device_train_batch_size = 4
        dataloader_num_workers = 0
        dataloader_pin_memory = False
        dataloader_prefetch_factor = None
        seed = 0

    loader = build_train_dataloader(DS(), lambda b: b, TA())
    check(
        "workers default to something above zero",
        loader.num_workers == DEFAULT_NUM_WORKERS,
    )
    check("workers are persistent when there are any", loader.persistent_workers)
    check("prefetch is deepened past the default 2", loader.prefetch_factor == 4)
    check("the short final batch is dropped", loader.drop_last and len(loader) == 2)

    TA.dataloader_num_workers = 2
    check(
        "an explicit worker count is honoured",
        build_train_dataloader(DS(), lambda b: b, TA()).num_workers == 2,
    )

    TA.dataloader_num_workers = -1
    check(
        "a negative worker count means run in-process",
        build_train_dataloader(DS(), lambda b: b, TA()).num_workers == 0,
    )

    class MultiTaskDS(DS):
        task_index_ranges = [(0, 12), (12, 24)]

        def __len__(self):
            return 24

    rank0 = TaskHomogeneousSampler(
        MultiTaskDS(), local_batch_size=3, seed=7, rank=0, world_size=2
    )
    rank1 = TaskHomogeneousSampler(
        MultiTaskDS(), local_batch_size=3, seed=7, rank=1, world_size=2
    )
    rows0, rows1 = list(rank0), list(rank1)
    global_batches = [
        rows0[i : i + 3] + rows1[i : i + 3]
        for i in range(0, len(rows0), 3)
    ]
    check(
        "task sampler makes each gathered DDP batch task-homogeneous",
        all(all(x < 12 for x in b) or all(x >= 12 for x in b) for b in global_batches),
    )
    check(
        "task sampler gives DDP ranks disjoint slices of the same global batch",
        all(
            set(rows0[i : i + 3]).isdisjoint(rows1[i : i + 3])
            for i in range(0, len(rows0), 3)
        ),
    )
    rank0.set_epoch(1)
    check(
        "task sampler reshuffles across epochs",
        list(rank0) != rows0,
    )


def span_common_checks():
    """The HieRD span helpers: what may be detached, and where the blocks sit.

    Every check here failed before the fix. The gradient ones matter most: a
    detached cluster mean leaves the span-level, cluster-level and cross-modal
    terms constant, so the loss still prints a plausible, falling number while
    contributing nothing to the student.
    """
    from src.criterions import span_common as sc

    torch.manual_seed(0)
    hidden = torch.randn(7, 8, requires_grad=True)
    # deliberately uneven clusters: 4 tokens and 3
    info = {
        "token_indices": torch.arange(7),
        "cluster_ids": torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        "num_clusters": 2,
    }

    weights, mass = sc.compute_intra_cluster_attention_weights(hidden, info)
    check(
        "per-token weights stay constants",
        not weights.requires_grad,
        "a weight says how much a token counts, not what it should become",
    )

    per_cluster = torch.zeros(2).scatter_add_(0, info["cluster_ids"], weights)
    check(
        "per-token weights are normalised within a cluster",
        torch.allclose(per_cluster, torch.ones(2), atol=1e-5),
    )
    check(
        "cluster mass tracks cluster size, so pair weights are not all ones",
        float(mass[0]) > float(mass[1]) + 0.5,
        f"mass={[round(float(x), 2) for x in mass]} for clusters of 4 and 3 tokens",
    )

    mean = sc.compute_weighted_cluster_mean(hidden, info, weights)
    check("the cluster mean keeps the student's graph", mean.requires_grad)
    grad = torch.autograd.grad(mean.sum(), hidden, retain_graph=True)[0]
    check(
        "gradient reaches every token that is in a cluster",
        bool((grad.abs().sum(dim=-1) > 0).all()),
    )

    # the mean is the weighted average it claims to be
    manual = (hidden[:4] * weights[:4].unsqueeze(-1)).sum(0) / weights[:4].sum()
    check(
        "the cluster mean is the weighted average of its tokens",
        torch.allclose(mean[0], manual, atol=1e-5),
    )

    # ---- sequence layout ------------------------------------------------
    nv, nt, pad = 4, 3, 2
    right = torch.zeros(1, nv + nt + pad, 5)
    right[0, :nv], right[0, nv:nv + nt] = 1, 2          # [vision][text][pad]
    left = torch.zeros(1, pad + nv + nt, 5)
    left[0, pad:pad + nv], left[0, pad + nv:] = 1, 2     # [pad][vision][text]

    for label, seq, left_padded in (("right-padded", right, False), ("left-padded", left, True)):
        text = sc.extract_text_hidden_states([seq], 0, nt, nv, left_padded=left_padded)[0]
        vision = sc.extract_vision_hidden_states([seq], 0, nv, nt, left_padded=left_padded)[0]
        check(
            f"{label} sequence: the text slice is the text",
            text.shape[0] == nt and bool((text == 2).all()),
        )
        check(
            f"{label} sequence: the vision slice is the vision",
            vision.shape[0] == nv and bool((vision == 1).all()),
        )

    check(
        "FastVLM is the right-padded student, LLaVA-OneVision is not",
        not sc.is_left_padded("llava_qwen2")
        and sc.is_left_padded("llava_onevision")
        and sc.is_left_padded("qwen2_vl"),
    )

    # ---- DDP: every projector takes part in every step ------------------
    projectors = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    touch = sc.projector_touch(projectors, torch.zeros(()))
    touch.backward()
    check("projector_touch contributes exactly zero", float(touch) == 0.0)
    check(
        "projector_touch still gives every projector a gradient",
        all(
            p.grad is not None and float(p.grad.abs().sum()) == 0.0
            for p in projectors.parameters()
        ),
        "otherwise a rank whose batch has no spans desyncs DDP",
    )


def main():
    for name, fn in [
        ("registry", registry_checks),
        ("criterion base", base_criterion_checks),
        ("span helpers", span_common_checks),
        ("images", image_checks),
        ("loss meter", loss_meter_checks),
        ("gradient accumulation", grad_accumulation_checks),
        ("max steps", max_steps_checks),
        ("dataloader", dataloader_checks),
    ]:
        print(f"\n--- {name} ---")
        fn()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
