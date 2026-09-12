# Legacy criterion hook compatibility

## Problem

`attach_criterion()` unconditionally calls `criterion.build_parameters()`.
New criteria inherit the no-op hook from `DistillCriterion`, but HieRD and
other legacy criteria are plain `nn.Module` classes and do not define it. A
HieRD run therefore fails during setup before the first optimizer step.

## Design

Keep `build_parameters()` optional at the integration boundary. The entrypoint
will call it only when the criterion exposes a callable hook, then retain the
existing parameter-registration behavior unchanged. This preserves TALAS and
future criteria that build trainable weights while restoring the documented
compatibility contract for legacy criteria.

## Verification

Add a regression check that attaches a plain `nn.Module` criterion without the
hook and verifies that setup succeeds. Existing TALAS tests continue to cover
the parameter-building path. Run the training-stack tests, then repeat the
two-step distributed HieRD smoke test on all eight GPUs.
