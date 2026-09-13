# DDP scaling and HieRD contrastive-only run

## Scope

Train the HieRD FastVLM CLS setup on eight H200 GPUs with seed 42 and an
effective global batch of 16, while retaining only the in-batch contrastive
objective requested as `L_base only` for this ablation.

## Objective

For this run the optimized loss is

\[
L = L_{\mathrm{con}},
\]

with the HieRD span, cross-modal, and RKD terms disabled. The launcher must
enforce

\[
B_{\mathrm{global}} = B_{\mathrm{device}} W A = 2 \times 8 \times 1 = 16.
\]

## DDP gradient correction

The current all-gather keeps only the local slice differentiable. Every rank
computes the same global contrastive loss, but DDP subsequently averages the
local parameter gradients. Without a correction, the student gradient is
smaller than the single-process global-batch gradient by `1 / world_size`.

Apply an identity autograd operation to the local tensor before inserting it
into the gathered tensor. Its forward value is unchanged and its backward
gradient is multiplied by `world_size`. This corrects only the gathered student
path; it does not incorrectly scale gradients of criterion-owned projectors or
other rank-local loss terms. Do not change the shared legacy `GatherLayer`,
because older callers already multiply their loss by `world_size`.

## Contrastive-only execution

Add a `--hierd_contrastive_only` switch to the HieRD attention criterion. When
enabled, it performs the student forward and corrected global contrastive loss,
touches HieRD projectors with a zero-valued term for DDP consistency, and returns
before teacher forward, span extraction, clustering, cross-modal loss, and RKD.
The training setup may therefore omit teacher collation and teacher model
loading for this mode.

## Launcher and validation

Make GPU count, per-device batch, accumulation, seed, output directory, and
HieRD weights overridable through environment variables. Compute and print the
global batch in the launcher and fail before `torchrun` if it differs from an
optional expected value.

Add regression checks for the gradient-only scaling identity and for the
launcher validation. Run the repository self-checks in the project virtual
environment before migration, then repeat the training-stack check on the
server.

## Cleanup

Resolve and measure only the recent aborted/invalid ablation directories and
their corresponding logs. Delete those exact paths, report the freed bytes,
and leave unrelated HieRD, VQA, CMTop, datasets, caches, and uploaded results
untouched.
