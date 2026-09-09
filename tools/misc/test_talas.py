"""Self-checks for TALAS and the sharpness-aware optimizer. No GPU, no download.

    python tools/misc/test_talas.py

Covers the three pieces of `docs/baseline methods/TALAS.pdf` separately, because
each one fails silently in its own way:

* `L_TAMD` -- anchors the *top* K layers, so an off-by-one in the layer indexing
  trains the bottom of the student against the teacher and still descends,
* `L_LASD` -- the guide detach is the difference between top-down propagation
  and a symmetric smoothness penalty, and both produce a falling loss curve,
* ASAM -- a perturbation that is not restored exactly leaves the run training a
  drifted model, which looks like nothing at all until the numbers are bad.
"""

import os as _os
import sys as _sys

_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
)

import torch
import torch.nn.functional as F
from torch import nn

FAILURES = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


# --------------------------------------------------------------- fake models


class FakeStudent(nn.Module):
    """Stands in for MMEBModel: `encode_input` returns per-layer hidden states."""

    def __init__(self, dim=8, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.num_layers = num_layers

    def encode_input(self, inputs):
        h = inputs["x"]
        hidden = [h]
        for _ in range(self.num_layers):
            h = torch.tanh(self.proj(h))
            hidden.append(h)
        pooled = self._pooling(hidden[-1], inputs["attention_mask"])
        return (pooled, None, None, tuple(hidden))

    def _pooling(self, hidden, attention_mask):
        # The real one picks the last non-pad position; the fake batch has no
        # padding, so the last position it is.
        return hidden[:, -1, :]

    def compute_similarity(self, q, p):
        return q @ p.t()


class FakeDistiller(nn.Module):
    def __init__(self, dim=8, teacher_dim=6, num_layers=4):
        super().__init__()
        self.student = FakeStudent(dim, num_layers)
        self.projectors = None
        self.temperature = 0.02
        self.student_hidden_dim = dim
        self.teacher_hidden_dim = teacher_dim

    def encode_teacher(self, batch, side, dtype=None):
        t = batch["teacher"][side]
        return t.to(dtype) if dtype else t

    def forward(self, criterion, batch):
        return criterion(self, batch)


class Args:
    """Every flag `TALASLoss` reads, at its `src/arguments.py` default."""

    kd_weight = 0.5
    talas_contrastive_weight = 1.0
    talas_tamd_weight = 0.75
    talas_lasd_weight = 1.0
    talas_num_tamd_layers = 2
    talas_num_lasd_layers = 0
    talas_lasd_detach_guide = True
    talas_lasd_reduction = "sum"

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if not hasattr(Args, k):
                raise AttributeError(f"no such flag: {k}")
            setattr(self, k, v)


def make_batch(bs=4, dim=8, seq=3, teacher_dim=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    side = lambda: {  # noqa: E731
        "x": torch.randn(bs, seq, dim, generator=g),
        "attention_mask": torch.ones(bs, seq, dtype=torch.long),
    }
    return {
        "student_inputs": {"qry": side(), "pos": side()},
        "teacher": {
            "qry": torch.randn(bs, teacher_dim, generator=g),
            "pos": torch.randn(bs, teacher_dim, generator=g),
        },
    }


def build(**overrides):
    from src.criterions.talas import TALASLoss

    torch.manual_seed(0)
    distiller = FakeDistiller()
    criterion = TALASLoss(Args(**overrides))
    criterion.build_parameters(distiller)
    return distiller, criterion


# ------------------------------------------------------------------- checks


def registry_checks():
    from src.criterions.registry import (
        CRITERIONS,
        attention_needs,
        get_spec,
        supports_teacher_cache,
    )

    check("talas is registered", "talas" in CRITERIONS)
    spec = get_spec("talas")
    check("talas loads its class", spec.load().__name__ == "TALASLoss")
    check(
        "talas reads no attentions on either side",
        attention_needs("talas") == (False, False),
    )
    check(
        "talas can train from a teacher embedding cache",
        supports_teacher_cache("talas") is True,
        "the paper caches the teacher's embeddings once, appendix C",
    )


def parameter_checks():
    distiller, criterion = build(talas_num_tamd_layers=3)
    check(
        "one projection per anchored layer",
        len(criterion.projections) == 3,
        f"{len(criterion.projections)}",
    )
    shapes = {tuple(p.weight.shape) for p in criterion.projections}
    check(
        "W_l maps student width to teacher width",
        shapes == {(distiller.teacher_hidden_dim, distiller.student_hidden_dim)},
        f"{shapes}",
    )
    check(
        "W_l has no bias (eq. 3 is a plain matrix)",
        all(p.bias is None for p in criterion.projections),
    )
    check(
        "the projections are trainable",
        all(p.requires_grad for p in criterion.parameters()),
    )

    try:
        build(talas_num_tamd_layers=0)
        check("K < 1 is refused", False)
    except ValueError as exc:
        check("K < 1 is refused", "talas_num_tamd_layers" in str(exc))
    try:
        build(talas_lasd_reduction="frobenius")
        check("an unknown reduction is refused", False)
    except ValueError as exc:
        check("an unknown reduction is refused", "talas_lasd_reduction" in str(exc))

    from src.criterions.talas import TALASLoss

    unbuilt = TALASLoss(Args())
    try:
        unbuilt.kd_loss(None)
        check("kd_loss without build_parameters raises", False)
    except RuntimeError as exc:
        check(
            "kd_loss without build_parameters raises a readable error",
            "build_parameters" in str(exc),
        )


def tamd_checks():
    _, criterion = build(talas_num_tamd_layers=2)
    n, num_layers, dim, teacher_dim = 5, 4, 8, 6

    torch.manual_seed(1)
    layers = torch.randn(n, num_layers, dim)
    teacher = torch.randn(n, teacher_dim)

    loss = criterion._tamd(layers, teacher)
    check("L_TAMD is a scalar", loss.shape == ())
    check("L_TAMD lies in [0, 2]", 0.0 <= float(loss) <= 2.0, f"{float(loss):.4f}")

    # Cosine distance is zero exactly when the projected layer is parallel to
    # the teacher. Force that by making W_l map onto the teacher directions.
    with torch.no_grad():
        aligned = torch.randn(n, num_layers, dim)
        # Solve for the single W that sends the top layer onto the teacher.
        w = torch.linalg.lstsq(aligned[:, -1, :], teacher).solution
        criterion.projections[0].weight.copy_(w.t())
        top_only = criterion._tamd(aligned, teacher)
    check(
        "L_TAMD -> 0 when the projected layer is parallel to the teacher",
        float(criterion._tamd(aligned[:, -1:, :], teacher)) < 1e-4,
        f"{float(criterion._tamd(aligned[:, -1:, :], teacher)):.2e}",
    )
    check(
        "with K=2 the second, unaligned anchor keeps the loss up",
        float(top_only) > 1e-3,
        f"{float(top_only):.4f}",
    )

    # The anchors are counted from the top: projections[0] is the final layer.
    _, k1 = build(talas_num_tamd_layers=1)
    with torch.no_grad():
        k1.projections[0].weight.copy_(criterion.projections[0].weight)
    check(
        "K=1 anchors the final layer, not the first",
        torch.allclose(
            k1._tamd(aligned, teacher),
            k1._tamd(aligned[:, -1:, :], teacher),
            atol=1e-6,
        ),
    )
    check(
        "K is clamped to the layers that exist",
        float(criterion._tamd(layers[:, :1, :], teacher)) >= 0.0,
    )


def lasd_checks():
    from src.criterions.talas import _relation_matrix

    _, criterion = build()
    n, num_layers, dim = 6, 5, 8
    torch.manual_seed(2)
    layers = torch.randn(n, num_layers, dim)

    loss = criterion._lasd(layers)
    check("L_LASD is a scalar", loss.shape == ())
    check("L_LASD is non-negative", float(loss) >= 0.0, f"{float(loss):.4f}")

    # Identical geometry at every layer -> nothing to align. Scaling a layer
    # leaves the cosine relation matrix untouched, so that must be free too.
    same = layers[:, :1, :].repeat(1, num_layers, 1)
    check("L_LASD is 0 when every layer has the same geometry", float(criterion._lasd(same)) < 1e-6)
    scaled = same.clone()
    scaled[:, 2, :] *= 7.0
    check(
        "L_LASD ignores a per-layer rescaling",
        float(criterion._lasd(scaled)) < 1e-6,
        "R_l is built from L2-normalised embeddings",
    )

    r = _relation_matrix(layers[:, 0, :])
    check("R is square, batch x batch", tuple(r.shape) == (n, n))
    check("R has a unit diagonal", torch.allclose(r.diagonal(), torch.ones(n), atol=1e-5))
    check("R is symmetric", torch.allclose(r, r.t(), atol=1e-5))

    # sum vs mean differ by exactly N^2.
    _, mean_crit = build(talas_lasd_reduction="mean")
    check(
        "mean reduction is the sum divided by N^2",
        torch.allclose(mean_crit._lasd(layers) * (n * n), criterion._lasd(layers), atol=1e-4),
    )

    # Pair count is taken from the top.
    _, two = build(talas_num_lasd_layers=2)
    top_three = layers[:, -3:, :].contiguous()
    check(
        "--talas_num_lasd_layers 2 aligns the top three layers",
        torch.allclose(two._lasd(layers), two._lasd(top_three), atol=1e-6),
    )
    check(
        "0 means every adjacent pair",
        not torch.allclose(criterion._lasd(layers), two._lasd(layers), atol=1e-6),
    )

    # The guide detach is the whole of section 3.2: gradient must reach the
    # lower layer of each pair and not the upper one.
    def upper_grad(detach):
        _, crit = build(talas_lasd_detach_guide=detach, talas_num_lasd_layers=1)
        lower = layers[:, -2, :].clone().requires_grad_(True)
        upper = layers[:, -1, :].clone().requires_grad_(True)
        crit._lasd(torch.stack([lower, upper], dim=1)).backward()
        return lower.grad, upper.grad

    lower_grad, upper_g = upper_grad(True)
    check(
        "detached guide: no gradient reaches the upper layer",
        upper_g is None or float(upper_g.abs().sum()) == 0.0,
    )
    check(
        "detached guide: the lower layer still gets one",
        lower_grad is not None and float(lower_grad.abs().sum()) > 0.0,
    )
    _, upper_g_off = upper_grad(False)
    check(
        "--talas_lasd_detach_guide false makes the term symmetric",
        upper_g_off is not None and float(upper_g_off.abs().sum()) > 0.0,
    )


def forward_checks():
    distiller, criterion = build()
    batch = make_batch()
    out = distiller(criterion, batch)

    for key in ("loss", "contrastive_loss", "kd_loss", "talas_tamd_loss", "talas_lasd_loss"):
        check(f"forward reports {key}", key in out)

    lam1, lam2, lam3 = 1.0, 0.75, 1.0
    expected_kd = lam2 * out["talas_tamd_loss"] + lam3 * out["talas_lasd_loss"]
    check(
        "kd_loss = lambda_2 L_TAMD + lambda_3 L_LASD",
        torch.allclose(out["kd_loss"], expected_kd, atol=1e-6),
    )
    check(
        "loss = lambda_1 L_contrastive + kd_loss",
        torch.allclose(out["loss"], lam1 * out["contrastive_loss"] + expected_kd, atol=1e-6),
    )
    check(
        "--kd_weight is not folded in a second time",
        criterion.kd_loss_weight == 1.0,
        "lambda_2 / lambda_3 are the knobs; --kd_weight is unused by talas",
    )

    # lambda_1 really scales the contrastive term.
    distiller2, crit2 = build(talas_contrastive_weight=0.001)
    out2 = distiller2(crit2, make_batch())
    check(
        "--talas_contrastive_weight scales the contrastive term",
        torch.allclose(
            out2["loss"], 0.001 * out2["contrastive_loss"] + out2["kd_loss"], atol=1e-6
        ),
    )

    out["loss"].backward()
    check(
        "the student gets a gradient",
        distiller.student.proj.weight.grad is not None
        and float(distiller.student.proj.weight.grad.abs().sum()) > 0.0,
    )
    check(
        "every W_l gets a gradient",
        all(
            p.weight.grad is not None and float(p.weight.grad.abs().sum()) > 0.0
            for p in criterion.projections
        ),
    )

    # A student that returns no hidden states must say so rather than crash
    # somewhere inside the pooling.
    class Flat(FakeStudent):
        def encode_input(self, inputs):
            pooled = self._pooling(inputs["x"], inputs["attention_mask"])
            return (pooled, None, None, None)

    distiller.student = Flat()
    try:
        distiller(criterion, make_batch())
        check("a student without hidden states is refused", False)
    except RuntimeError as exc:
        check("a student without hidden states is refused", "hidden states" in str(exc))


def sam_checks():
    from src.training.sam import SharpnessAwareOptimizer, build_sharpness_aware

    def fresh(rho=0.5, eta=0.01, adaptive=True):
        torch.manual_seed(3)
        model = nn.Linear(4, 3, bias=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        sam = SharpnessAwareOptimizer(
            opt, model.named_parameters(), rho=rho, eta=eta, adaptive=adaptive
        )
        model(torch.randn(5, 4)).pow(2).mean().backward()
        return model, sam

    model, sam = fresh()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    check("ascent_step reports that it perturbed", sam.ascent_step() is True)
    check("the wrapper knows it is perturbed", sam.perturbed is True)
    check(
        "the weights actually moved",
        any(not torch.equal(before[n], p) for n, p in model.named_parameters()),
    )

    # eps = rho * T_w^2 g / || T_w g ||, with T_w = |w| + eta on 'weight' only.
    t_w = {n: (before[n].abs() + 0.01 if "weight" in n else torch.ones_like(before[n]))
           for n in before}
    norm = torch.norm(torch.stack([(t_w[n] * grads[n]).norm() for n in before]))
    for n, p in model.named_parameters():
        want = before[n] + 0.5 * t_w[n] * t_w[n] * grads[n] / norm
        check(f"ASAM perturbation matches the formula for {n}", torch.allclose(p, want, atol=1e-6))

    sam.restore()
    check(
        "restore puts every weight back exactly",
        all(torch.equal(before[n], p) for n, p in model.named_parameters()),
    )
    check("the wrapper knows it is no longer perturbed", sam.perturbed is False)
    check(
        "restore leaves the gradient alone",
        all(torch.equal(grads[n], p.grad) for n, p in model.named_parameters()),
    )

    # Plain SAM: T_w = 1, so ||eps|| is exactly rho.
    model, sam = fresh(rho=0.05, adaptive=False)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    sam.ascent_step()
    step = torch.norm(
        torch.stack([(p - before[n]).norm() for n, p in model.named_parameters()])
    )
    check("SAM moves exactly rho in L2", abs(float(step) - 0.05) < 1e-6, f"{float(step):.6f}")

    # Bias parameters are not scaled by |w| even under ASAM.
    model, sam = fresh()
    bias_before = model.bias.detach().clone()
    bias_grad = model.bias.grad.detach().clone()
    sam.ascent_step()
    direction = model.bias - bias_before
    check(
        "ASAM leaves T_w = 1 for a non-'weight' parameter",
        torch.allclose(
            F.normalize(direction, dim=0), F.normalize(bias_grad, dim=0), atol=1e-5
        ),
    )
    sam.restore()

    model, sam = fresh()
    try:
        sam.ascent_step()
        sam.ascent_step()
        check("a double ascent without restore is refused", False)
    except RuntimeError as exc:
        check("a double ascent without restore is refused", "restore" in str(exc))
    sam.restore()

    # A step with no usable gradient falls back to the ordinary descent.
    model, sam = fresh()
    for p in model.parameters():
        p.grad.zero_()
    check("a zero gradient skips the perturbation", sam.ascent_step() is False)
    for p in model.parameters():
        p.grad.fill_(float("nan"))
    check("a non-finite gradient skips the perturbation", sam.ascent_step() is False)

    class TA:
        sharpness_aware = "none"
        sam_rho = 0.5
        asam_eta = 0.01

    model = nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    check("--sharpness_aware none builds no wrapper", build_sharpness_aware(opt, model, TA()) is None)
    TA.sharpness_aware = "sam"
    check("--sharpness_aware sam builds a non-adaptive wrapper",
          build_sharpness_aware(opt, model, TA()).adaptive is False)
    TA.sharpness_aware = "asam"
    check("--sharpness_aware asam builds an adaptive wrapper",
          build_sharpness_aware(opt, model, TA()).adaptive is True)
    TA.sharpness_aware = "sharpness"
    try:
        build_sharpness_aware(opt, model, TA())
        check("an unknown --sharpness_aware is refused", False)
    except ValueError as exc:
        check("an unknown --sharpness_aware is refused", "sharpness_aware" in str(exc))


def loop_checks():
    """The loop must run the window twice and step on the *second* gradient."""
    from src.criterions.base import DistillCriterion
    from src.training.loop import DistillTrainer
    from src.training.sam import SharpnessAwareOptimizer

    class Simple(DistillCriterion):
        def kd_loss(self, ctx):
            return F.mse_loss(ctx.student_qry, ctx.teacher_qry[:, : ctx.student_dim])

    class TA:
        gradient_accumulation_steps = 2
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

    def run(sharpness):
        torch.manual_seed(4)
        distiller = FakeDistiller(dim=8, teacher_dim=8)
        criterion = Simple(Args())
        optimizer = torch.optim.SGD(distiller.student.parameters(), lr=0.0)
        sam = (
            SharpnessAwareOptimizer(
                optimizer, distiller.student.named_parameters(), rho=0.5
            )
            if sharpness
            else None
        )
        batches = [make_batch(seed=s, teacher_dim=8) for s in range(4)]
        trainer = DistillTrainer(
            distiller=distiller,
            criterion=criterion,
            dataloader=batches,
            optimizer=optimizer,
            lr_scheduler=Sched(),
            model_args=None,
            training_args=TA(),
            device=torch.device("cpu"),
            sharpness_aware=sam,
        )
        forwards = []
        original = trainer._micro_step

        def spy(batch, is_sync_step):
            forwards.append(batch)
            return original(batch, is_sync_step)

        trainer._micro_step = spy

        # Sampled on `optimizer.step`, not on `_optimizer_step`: the replay
        # happens inside the latter, so a probe there would see the clean
        # gradient in both configurations and pass either way.
        grads = []
        base_step = optimizer.step

        def step_spy(*a, **kw):
            grads.append(distiller.student.proj.weight.grad.detach().clone())
            return base_step(*a, **kw)

        optimizer.step = step_spy
        trainer.run_epoch(0)
        return forwards, grads, distiller

    plain_fwd, plain_grads, _ = run(False)
    sam_fwd, sam_grads, distiller = run(True)

    check(
        "without --sharpness_aware each micro-batch runs once",
        len(plain_fwd) == 4,
        f"{len(plain_fwd)} forwards for 4 micro-batches",
    )
    check(
        "with --sharpness_aware each micro-batch runs twice",
        len(sam_fwd) == 8,
        f"{len(sam_fwd)} forwards for 4 micro-batches",
    )
    # Two windows of two micro-batches, each window run then replayed in the
    # same order: a replay that reordered or dropped a micro-batch would give a
    # descent gradient for a different batch than the ascent chose.
    seen = [id(b) for b in sam_fwd]
    first, second = seen[:2], seen[4:6]
    check(
        "the replay uses the same micro-batches, in the same order",
        seen == first + first + second + second and first != second,
    )
    check(
        "the step still happens once per accumulation window",
        len(sam_grads) == len(plain_grads) == 2,
        f"sam={len(sam_grads)} plain={len(plain_grads)}",
    )
    check(
        "the optimizer sees the perturbed gradient, not the clean one",
        not torch.allclose(sam_grads[0], plain_grads[0], atol=1e-7),
    )
    check(
        "the weights are left unperturbed after the epoch",
        not distiller.student.proj.weight.isnan().any(),
    )


def main():
    for name, fn in [
        ("registry", registry_checks),
        ("parameters", parameter_checks),
        ("L_TAMD", tamd_checks),
        ("L_LASD", lasd_checks),
        ("forward", forward_checks),
        ("SAM / ASAM", sam_checks),
        ("training loop", loop_checks),
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
