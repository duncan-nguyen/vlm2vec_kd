# H200 CM-Merge CLS deployment design

## Goal

Migrate the current `vlm2vec_kd` workspace to
`H200_ANNP36:/home/annp36/work/vlm_2_vec` and start the main CM-Merge
classification run for teacher `raghavlite/B3_Qwen2_2B` and student
`apple/FastVLM-0.5B` on all eight H200 GPUs.

## Constraints and preserved state

- Reuse the remote project virtual environment. Never install packages into the
  global Python environment.
- Preserve remote MMEB images, Hugging Face/model caches, teacher embeddings,
  evaluation data, checkpoints, logs, credentials, and Git metadata.
- Include the current local source tree, including tracked working-tree changes.
- Do not delete remote-only artifacts during migration.
- Run from the repository root so relative image and cache paths stay valid.

## Migration

Use an in-place `rsync` without `--delete`. Exclude `.git`, `.venv`, data,
caches, outputs, logs, credentials, bytecode, and editor metadata. Compare key
source hashes after transfer and run the repository's configuration checker.

This is preferred over a clean clone, which would require restoring large
artifacts, and over a plain Git pull, which would omit the local working-tree
change in `tools/eval_mmeb.py`.

## Training configuration

- Method: `VARIANT=cmmerge` (`--kd_loss_type cmtop`, merge mode)
- Task/subsets: CLS over `ImageNet_1K`, `N24News`, `HatefulMemes`, `VOC2007`,
  and `SUN397`
- Teacher: `raghavlite/B3_Qwen2_2B`, Qwen2-VL backbone, EOS pooling, normalized
- Student: `apple/FastVLM-0.5B`, `llava_qwen2` backbone, EOS pooling, normalized
- Precision: BF16
- Distributed launch: one node, eight processes (`NUM_GPUS_PER_NODE=8`)
- Per-device batch: 16; global topology micro-batch:
  \[
  B_{\mathrm{global}} = 16 \times 8 = 128.
  \]
- Epochs: 1; learning rate: `1e-4`; seed: 42
- Teacher cache: `cache/b3_qwen2_2b_fastvlm_cls_v2`
- Main output: `training/CMTop/cmmerge_fastvlm_cls_b3qwen2_2b_8xh200_seed42`
- Main log: `logs/cmtop_fastvlm_cls_b3qwen2_2b_8xh200_seed42.log`

The selected teacher cache is complete and matches the model, subset order,
full-data fraction, EOS pooling, normalization, and 448-pixel image setting.

## Validation and launch sequence

1. Run `tools/check_paper_settings.py` and the three CM-Merge/cache/training-stack
   tests in the remote `.venv`.
2. Run a disposable eight-GPU smoke job at a small data fraction, with hub push
   and post-training evaluation disabled.
3. Require a zero smoke exit code, finite logged loss, and no distributed error.
4. Start the full run detached with `nohup` and record its shell PID.
5. Confirm the process tree contains eight workers, `nvidia-smi` shows activity
   on GPUs 0-7, and the log reaches training steps without NaN/Inf or traceback.

If the smoke run exposes an out-of-memory error, reduce only the per-device
batch size while keeping all eight GPUs. If it exposes a cache mismatch, do not
bypass validation; rebuild the teacher cache in the project virtual environment.
