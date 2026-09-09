"""Sharpness-aware minimization (SAM) and its adaptive variant (ASAM).

TALAS's third component, but nothing about it is TALAS-specific: it wraps a
base optimizer and any `--kd_loss_type` can be trained with it via
`--sharpness_aware {sam,asam}`.

Both solve the min-max problem

    min_w  max_{|| T_w^{-1} eps ||_2 <= rho}  L(w + eps)

with a first-order approximation of the inner maximum, which makes one
optimizer step two forward/backward passes:

1. **ascent** -- from the gradient at `w`, perturb to `w + eps` where

       eps = rho * T_w^2 grad / || T_w grad ||_2

2. **descent** -- recompute the gradient at `w + eps`, restore `w`, and hand
   that gradient to the base optimizer.

`T_w` is what separates the two. SAM (Foret et al., 2020) uses `T_w = 1`, so the
neighbourhood is a plain L2 ball and its size is not invariant to rescaling a
layer's weights. ASAM (Kwon et al., 2021) uses the element-wise `T_w = |w| + eta`,
which makes the neighbourhood scale with each weight's own magnitude. Following
the reference implementation, `T_w` is applied only to parameters whose name
contains "weight"; biases and other 1-d parameters keep `T_w = 1`.

The perturbed weights are restored from a saved copy rather than by subtracting
`eps` again: the trainable weights here are bf16, where `(w + eps) - eps` does
not reliably return `w`, and the error would accumulate over a run. The copy is
the same size `eps` would have been.
"""

import torch

VALID_MODES = {"none", "sam", "asam"}


class SharpnessAwareOptimizer:
    """Two-step ascent/descent around a base optimizer.

    Not a `torch.optim.Optimizer`: the training loop drives the two steps
    explicitly because the second one has to replay the whole gradient
    accumulation window, which only the loop knows about.

    Args:
        base_optimizer: the optimizer that actually applies the update.
        named_parameters: `(name, parameter)` pairs to perturb. Only those with
            `requires_grad` are kept.
        rho: neighbourhood radius. SAM's usual 0.05 is far too small for ASAM,
            whose radius is measured in units of `|w|`; ASAM's paper uses 0.5-2.0.
        eta: the `|w| + eta` floor, so a weight that has decayed to zero still
            gets a neighbourhood. ASAM only.
        adaptive: True for ASAM, False for plain SAM.
    """

    def __init__(self, base_optimizer, named_parameters, rho=0.5, eta=0.01,
                 adaptive=True):
        if rho <= 0:
            raise ValueError(f"--sam_rho must be > 0, got {rho}")
        self.base_optimizer = base_optimizer
        self.parameters = [(n, p) for n, p in named_parameters if p.requires_grad]
        self.rho = float(rho)
        self.eta = float(eta)
        self.adaptive = bool(adaptive)
        self._saved = {}
        if not self.parameters:
            raise ValueError(
                "sharpness-aware training was asked for but no parameter has "
                "requires_grad; there is nothing to perturb"
            )

    # -- the two steps ----------------------------------------------------

    @torch.no_grad()
    def ascent_step(self):
        """Move the weights to `w + eps`. Returns False if there was no gradient.

        The caller must have a gradient in `.grad` -- under DDP, one that has
        already been reduced, which is why this runs on the sync micro-batch and
        not before it. Leaves `.grad` alone; the caller zeroes it before the
        second pass.
        """
        if self._saved:
            raise RuntimeError(
                "ascent_step() called twice without a restore(); the weights "
                "are already perturbed and restoring would leave the model at "
                "w + eps"
            )

        scales = {}
        norms = []
        for name, param in self.parameters:
            if param.grad is None:
                continue
            scale = self._scale(name, param)
            scales[name] = scale
            scaled = param.grad if scale is None else param.grad * scale
            norms.append(scaled.detach().float().norm(p=2))
        if not norms:
            return False

        grad_norm = torch.norm(torch.stack(norms), p=2)
        if not torch.isfinite(grad_norm) or grad_norm == 0:
            # A step whose gradient overflowed or vanished: take the ordinary
            # descent step instead of dividing by it.
            return False
        factor = self.rho / (grad_norm + 1e-12)

        for name, param in self.parameters:
            if param.grad is None:
                continue
            scale = scales[name]
            # eps = rho * T_w^2 grad / || T_w grad ||, i.e. T_w applied twice.
            eps = param.grad.detach().float()
            if scale is not None:
                eps = eps * scale.float() * scale.float()
            eps = eps * factor
            self._saved[name] = param.detach().clone()
            param.add_(eps.to(param.dtype))
        return True

    @torch.no_grad()
    def restore(self):
        """Put the weights back to `w`, exactly."""
        for name, param in self.parameters:
            saved = self._saved.pop(name, None)
            if saved is not None:
                param.copy_(saved)
        self._saved.clear()

    @property
    def perturbed(self):
        return bool(self._saved)

    # -- helpers ----------------------------------------------------------

    def _scale(self, name, param):
        """`T_w` for one parameter, or None when it is the identity."""
        if not self.adaptive or "weight" not in name:
            return None
        return param.detach().abs() + self.eta


def build_sharpness_aware(base_optimizer, module, training_args):
    """The wrapper named by `--sharpness_aware`, or None for the plain step."""
    mode = str(getattr(training_args, "sharpness_aware", "none") or "none").lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"--sharpness_aware must be one of {sorted(VALID_MODES)}, got {mode!r}"
        )
    if mode == "none":
        return None
    return SharpnessAwareOptimizer(
        base_optimizer,
        module.named_parameters(),
        rho=getattr(training_args, "sam_rho", 0.5),
        eta=getattr(training_args, "asam_eta", 0.01),
        adaptive=(mode == "asam"),
    )
