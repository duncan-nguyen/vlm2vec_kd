# Distributed contrastive VQA and homogeneous HieRD setup

## Goal

Prepare, but do not launch, three FastVLM experiment cells on H200_Tensara:

1. Proposed Method contrastive-only on VQA with `DistributedSampler`.
2. Full HieRD on CLS with one task-homogeneous global batch per step.
3. Full HieRD on VQA with one task-homogeneous global batch per step.

All cells use `raghavlite/B3_Qwen2_2B` as teacher,
`apple/FastVLM-0.5B` as student, seed 42, eight GPUs, gradient accumulation 1,
and global batch 16:

\[
B_{\mathrm{global}} = 8 \times 2 \times 1 = 16.
\]

Only IND evaluation is requested: `vqa_ind` for the Proposed VQA cell,
`cls_ind` for HieRD CLS, and `vqa_ind` for HieRD VQA.

## Sampler selection

Add a method-independent `--task_homogeneous_sampling` training flag. It is
off by default. The dataloader selects `TaskHomogeneousSampler` when this flag
is on, or when the existing Ours-specific homogeneous mode is on. This keeps
all existing method defaults and preserves `--ours_task_homogeneous False` as
the way to select `DistributedSampler` for Ours.

The HieRD homogeneous launchers explicitly enable the new generic flag. The
Proposed contrastive-only launcher explicitly disables the Ours homogeneous
mode, which selects `DistributedSampler`, and sets `--ours_weight 0` through
the existing `student_only` variant while retaining the retrieval loss.

## Launchers

Create ready-to-run launchers under `scripts/train/experiments/`:

- `fastvlm_vqa_contrastive_distributed.sh`
- `hierd_fastvlm_cls_homogeneous.sh`
- `hierd_fastvlm_vqa_homogeneous.sh`

Each launcher validates the global-batch equation before delegating to the
existing method script. Each supports `DRY_RUN=True`, which prints the exact
configuration and exits without starting `torchrun`.

Output directories and Hub paths are unique and identify method, sampler,
seed, global batch, and hardware shape. Evaluation is fail-hard so an
incomplete benchmark set cannot be mistaken for a finished result.

## Verification and deployment

- Unit-test that the generic flag selects `TaskHomogeneousSampler` for HieRD
  without changing the Ours default or the explicit Ours distributed mode.
- Run the existing training-stack tests in the project virtual environment.
- Run `bash -n` and `DRY_RUN=True` for all new launchers; do not train.
- Synchronize tracked source, scripts, tests, and documentation from local to
  `/home/tensara/projects/vlm2vec`, excluding datasets, caches, logs, training
  artifacts, and `.venv`.
- Verify checksums and repeat the no-training dry runs on H200_Tensara.

## Non-goals

- No training, evaluation, checkpoint creation, upload, or GPU allocation.
- No deletion or replacement of existing artifacts.
- No change to sampler defaults for baseline methods.
