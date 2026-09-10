"""Run the MMEB evaluation at the end of training, without a second command.

`tools/eval_mmeb.py` is launched as a subprocess, once per subset, rather than
imported. Three reasons:

* the eval flags that matter most -- `--image_resolution`, `--model_backbone`,
  the LoRA rank -- are read off the arguments the run was *trained* with, so the
  mismatch the eval scripts warn about in their header comments cannot happen;
* a fresh process gets a clean CUDA context, so the student is loaded for
  inference next to nothing else rather than next to the training graph;
* on a multi-GPU run each rank evaluates its own shard of the subsets, so 20
  benchmarks on 8 GPUs cost roughly what 3 cost on one.
"""

import gc
import os
import subprocess
import sys

import torch
import torch.distributed as dist

from src.evaluation.benchmarks import resolve_groups
from src.evaluation.summary import write_summary
from src.utils import print_master, print_rank

# torchrun sets these; a child that inherits them has HF's `TrainingArguments`
# build a distributed device and try to join a process group of one.
_DIST_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "GROUP_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)

EVAL_SCRIPT = os.path.join("tools", "eval_mmeb.py")


def _rank_and_world():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _child_env(local_rank):
    """The parent environment, minus torchrun, pinned to this rank's GPU.

    `CUDA_VISIBLE_DEVICES` may already restrict the node (the launcher scripts
    set it), so index into it rather than assuming device N is GPU N.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("TORCHELASTIC_")}
    for key in _DIST_ENV:
        env.pop(key, None)
    for key in list(env):
        if key.startswith("ACCELERATE_"):
            env.pop(key)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.is_available():
        devices = visible.split(",") if visible else None
        if devices and local_rank < len(devices):
            env["CUDA_VISIBLE_DEVICES"] = devices[local_rank].strip()
        elif not devices:
            env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    return env


def release_training_memory():
    """Collect the training graph and hand its blocks back to the CUDA driver.

    The eval subprocess allocates on the same physical GPU, and it can only see
    memory this process has returned to the driver -- PyTorch's caching
    allocator holds freed blocks until `empty_cache()`. Call it only once the
    caller has dropped its own references; an object passed in as an argument is
    still referenced by the argument, so freeing it here would do nothing.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def eval_command(checkpoint, subset, model_args, data_args, training_args, out_dir):
    """The `tools/eval_mmeb.py` command line for one subset.

    Everything that has to match training is taken from the training arguments;
    everything that is eval-only comes off the `--eval_*` flags.
    """
    image_dir = data_args.eval_image_dir or os.environ.get(
        "MMEB_EVAL_DIR", "./eval_images"
    )
    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--model_name", checkpoint,
        "--model_backbone", model_args.model_backbone,
        "--encode_output_path", out_dir,
        "--pooling", model_args.pooling,
        "--normalize", str(bool(model_args.normalize)),
        "--dataset_name", data_args.eval_dataset_name or "TIGER-Lab/MMEB-eval",
        "--subset_name", subset,
        "--dataset_split", training_args.eval_dataset_split,
        "--per_device_eval_batch_size", str(training_args.per_device_eval_batch_size),
        "--image_dir", image_dir,
    ]
    if model_args.lora:
        cmd += [
            "--lora",
            "--lora_r", str(model_args.lora_r),
            "--lora_alpha", str(model_args.lora_alpha),
        ]
    if data_args.image_resolution:
        cmd += ["--image_resolution", str(data_args.image_resolution)]
    if training_args.bf16:
        cmd += ["--bf16"]
    if training_args.eval_tgt_prefix_mod:
        cmd += ["--tgt_prefix_mod"]
    return cmd


def run_post_training_eval(model_args, data_args, training_args):
    """Evaluate `checkpoint-final` on the requested benchmarks. Returns failures.

    Raises nothing on a failed subset: the checkpoint is already on disk and a
    broken image directory should not read as a broken training run. The names
    of the subsets that failed come back so the caller can exit accordingly.
    """
    checkpoint = training_args.eval_checkpoint or os.path.join(
        training_args.output_dir, "checkpoint-final"
    )
    subsets = resolve_groups(
        data_args.eval_subset_name or training_args.eval_benchmarks
    )
    out_dir = training_args.eval_output_dir or os.path.join(checkpoint, "mmeb_eval")

    rank, world = _rank_and_world()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        print_master(
            f"Post-training evaluation: {len(subsets)} subsets from "
            f"{checkpoint} -> {out_dir}"
        )
    if dist.is_initialized():
        dist.barrier()

    # Round-robin rather than contiguous chunks: the subsets differ in size by
    # an order of magnitude (ImageNet-1K against VOC2007) and interleaving them
    # keeps the ranks from finishing minutes apart.
    mine = subsets[rank::world]
    env = _child_env(local_rank)
    failures = []
    for subset in mine:
        cmd = eval_command(
            checkpoint, subset, model_args, data_args, training_args, out_dir
        )
        print_rank(f"eval {subset}: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print_rank(f"eval FAILED for {subset} (exit {result.returncode})")
            failures.append(subset)

    if dist.is_initialized():
        # Gather the failures so rank 0 reports the whole run, not its own share.
        gathered = [None] * world
        dist.all_gather_object(gathered, failures)
        failures = [subset for shard in gathered for subset in shard]
        dist.barrier()

    if rank == 0:
        summary, table = write_summary(
            out_dir, subsets, title=f"MMEB results -- {checkpoint}"
        )
        print(table, flush=True)
        print(f"\nWrote {os.path.join(out_dir, 'summary.json')}", flush=True)
        if failures:
            print(
                f"\n{len(failures)} subset(s) failed to evaluate: "
                f"{', '.join(sorted(failures))}",
                flush=True,
            )
    return failures
