# Ours with DistributedSampler: CLS then VQA

## Goal

After the active HieRD contrastive-only CLS run succeeds, run the full `ours`
criterion with PyTorch `DistributedSampler` on CLS and then VQA, sequentially,
on all eight H200 GPUs.

## Fixed protocol

- Teacher/student: `raghavlite/B3_Qwen2_2B` to `apple/FastVLM-0.5B`.
- Seed: 42.
- World size: 8.
- Per-device batch: 2.
- Gradient accumulation: 1.
- Global batch:

  \[
  B_{\mathrm{global}} = 8 \times 2 \times 1 = 16.
  \]

- Criterion: `ours`, with the normal full objective and `ours_weight=1.0`.
- Sampler selection: pass `--ours_task_homogeneous False`. The existing
  dataloader then selects `DistributedSampler`; no sampler code or default is
  changed.
- DDP gathered-gradient correction: use the corrected shared criterion base
  already deployed for the active HieRD run.

## Sequence

1. Require successful completion of the active HieRD run at
   `training/hierd/fastvlm_cls/contrastive_only_seed42_gb16_8xh200`.
2. Launch CLS into
   `training/ours_distributed_sampler/fastvlm_cls/ours_seed42_gb16_8xh200`.
3. Evaluate the CLS checkpoint on `cls_ind` and `cls_ood`.
4. Only if CLS training and evaluation finish without a fatal error, launch VQA
   into `training/ours_distributed_sampler/fastvlm_vqa/ours_seed42_gb16_8xh200`.
5. Evaluate the VQA checkpoint on `vqa_ind` and `vqa_ood`.

Logs live under `logs/ours_distributed_sampler/`. Existing checkpoints and
logs are never reused or overwritten.

## Failure handling and monitoring

The existing five-minute heartbeat owns the state transition. It verifies the
active stage is advancing, has eight ranks and eight GPU processes, contains no
traceback/NCCL/OOM/NaN/killed signal, and logs the expected sampler and batch
configuration. A failed or stalled stage stops the sequence and notifies the
user. Successful completion advances exactly once to the next stage; after VQA
completes, the heartbeat reports both checkpoint/evaluation locations and
disables itself.
