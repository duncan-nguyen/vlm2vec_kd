"""Run the frozen teacher over the distillation dataset once and cache its embeddings.

The teacher never changes and the dataset applies no augmentation, so its final
embedding for a sample is a pure function of the dataset index. A criterion that
reads nothing else -- `cmtop`, `contrastive_rkd`, `universal_logit` -- can then
train with no teacher model in the process at all: no teacher forward, no teacher
image preprocessing, and several GB of device memory freed for a larger batch.

Build the cache once and every variant and seed of an ablation reuses it:

    torchrun --nproc_per_node=1 tools/precompute_teacher_embeddings.py \
        --model_name "apple/FastVLM-0.5B" \
        --teacher_model_name "raghavlite/B3_Qwen2_2B" \
        --teacher_backbone "qwen2_vl" --teacher_pooling "eos" \
        --teacher_lora True --teacher_lora_r 8 --teacher_normalize True \
        --dataset_name "TIGER-Lab/MMEB-train" --dataset_split "original" \
        --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
        --image_dir "$MMEB_TRAIN_DIR" --image_resolution "448" \
        --per_device_train_batch_size 32 \
        --teacher_embedding_cache "cache/b3_qwen2_2b_cls" \
        --output_dir /tmp/precompute

The arguments that decide what the teacher produces are written into the cache's
`meta.json` and re-checked at training time, so a cache built for a different
teacher, subset list or resolution is refused rather than silently used.

Under `torchrun` each rank encodes a contiguous block of the dataset and writes
its own shard; rank 0 merges the shards into the final memmap and marks it
complete. Nothing is written concurrently to the same file.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import shutil

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.distiller import DistillationCollator, DistillationDataset
from src.model.model import MMEBModel
from src.teacher_cache import TeacherEmbeddingCache, build_fingerprint
from src.utils import print_master, print_rank

_SHARD_DIR = "_shards"


def _rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _teacher_model_args(model_args):
    """The same teacher description `Distiller` builds, without the student."""
    return ModelArguments(
        model_name=model_args.teacher_model_name,
        checkpoint_path=getattr(model_args, "teacher_checkpoint_path", None),
        lora=model_args.teacher_lora,
        lora_r=model_args.teacher_lora_r,
        lora_alpha=model_args.teacher_lora_alpha,
        lora_dropout=model_args.teacher_lora_dropout,
        lora_target_modules=model_args.teacher_lora_target_modules,
        pooling=model_args.teacher_pooling,
        normalize=model_args.teacher_normalize,
        model_backbone=model_args.teacher_backbone,
    )


def _pooled(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _block_for_rank(num_samples, rank, world_size):
    """Contiguous, disjoint slice of the dataset for this rank."""
    per_rank = (num_samples + world_size - 1) // world_size
    start = min(rank * per_rank, num_samples)
    return start, min(start + per_rank, num_samples)


def encode_block(teacher, loader, device, desc):
    ids, qry_rows, pos_rows = [], [], []
    # MMEBModel.load() gives bf16 weights while the image processor hands back
    # fp32 pixel values; training reconciles the two with Accelerate's bf16
    # mixed precision, so this pass has to autocast the same way or the vision
    # tower sees the wrong dtype.
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=device.type == "cuda")
    for batch in tqdm(loader, desc=desc, disable=_rank_world()[0] != 0):
        sample_ids = batch["sample_ids"]
        inputs = {
            side: {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch["teacher_inputs"][side].items()}
            for side in ("qry", "pos")
        }
        with torch.no_grad(), autocast:
            qry = _pooled(teacher.encode_input(inputs["qry"]))
            pos = _pooled(teacher.encode_input(inputs["pos"]))
        if sample_ids.numel() != qry.size(0):
            raise RuntimeError(
                f"{sample_ids.numel()} sample ids against {qry.size(0)} embeddings; "
                f"a dataset row expanded into more than one example, which the "
                f"index-keyed cache cannot represent"
            )
        ids.append(sample_ids.cpu().numpy())
        qry_rows.append(qry.float().cpu().numpy().astype(np.float16))
        pos_rows.append(pos.float().cpu().numpy().astype(np.float16))
    if not ids:
        return np.zeros(0, dtype=np.int64), None, None
    return np.concatenate(ids), np.concatenate(qry_rows), np.concatenate(pos_rows)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    cache_path = model_args.teacher_embedding_cache
    if not cache_path:
        raise SystemExit("--teacher_embedding_cache is required: it is what to build")

    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size = _rank_world()
    device = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))

    # The teacher side is the whole point here, and the student side is dead
    # weight: skipping it avoids resizing every image a second time.
    dataset = DistillationDataset(
        data_args, model_args, include_student=False, include_teacher=True
    )
    num_samples = len(dataset)
    fingerprint = build_fingerprint(model_args, data_args)

    teacher_args = _teacher_model_args(model_args)
    teacher = MMEBModel.load(teacher_args, is_trainable=False)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    # Nothing here reads attentions, so the backbone can stay on SDPA.
    teacher.output_attentions = False
    teacher.set_attn_implementation("sdpa")

    collator = DistillationCollator(
        student_processor=None,
        teacher_processor=_load_teacher_processor(teacher_args),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        include_student=False,
        include_teacher=True,
    )

    start, end = _block_for_rank(num_samples, rank, world_size)
    print_rank(f"rank {rank}/{world_size} encoding samples [{start}, {end})")
    loader = DataLoader(
        Subset(dataset, range(start, end)),
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=training_args.dataloader_num_workers,
    )

    ids, qry, pos = encode_block(teacher, loader, device, f"teacher rank {rank}")

    shard_dir = os.path.join(cache_path, _SHARD_DIR)
    os.makedirs(shard_dir, exist_ok=True)
    np.savez(
        os.path.join(shard_dir, f"shard_{rank}.npz"),
        ids=ids,
        qry=qry if qry is not None else np.zeros((0, 1), dtype=np.float16),
        pos=pos if pos is not None else np.zeros((0, 1), dtype=np.float16),
    )
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        dim = int(qry.shape[-1]) if qry is not None and qry.size else None
        if dim is None:
            raise SystemExit("rank 0 encoded nothing; is the dataset empty?")
        cache = TeacherEmbeddingCache.create(cache_path, num_samples, dim, fingerprint)
        all_ids = []
        for shard_rank in range(world_size):
            shard = np.load(os.path.join(shard_dir, f"shard_{shard_rank}.npz"))
            shard_ids = shard["ids"]
            if shard_ids.size == 0:
                continue
            cache.write(shard_ids, shard["qry"], shard["pos"])
            all_ids.append(shard_ids)

        # Every dataset index exactly once. Equal counts are not enough: a row
        # that expanded into two examples would write the same slot twice and
        # leave another index untouched, and the cache would then serve one
        # sample's embedding for two different samples.
        written = np.concatenate(all_ids) if all_ids else np.zeros(0, dtype=np.int64)
        unique = np.unique(written)
        if unique.size and unique.min() < 0:
            raise RuntimeError(
                "some rows carried a placeholder sample id: the dataset emitted "
                "an example whose text/image pairs were all rejected, and there "
                "is no dataset index to cache it under."
            )
        if unique.size != num_samples or written.size != num_samples:
            missing = num_samples - unique.size
            raise RuntimeError(
                f"cache would be wrong: {written.size} rows written covering "
                f"{unique.size} of {num_samples} dataset indices "
                f"({missing} never written). A dataset row probably expanded "
                f"into a number of examples other than one, which an "
                f"index-keyed cache cannot represent. Refusing to mark it usable."
            )
        cache.close(mark_complete=True)
        shutil.rmtree(shard_dir, ignore_errors=True)
        size_gb = num_samples * 2 * dim * 2 / 2**30
        print_master(
            f"Wrote {num_samples} teacher embeddings (dim {dim}, {size_gb:.2f} GiB) "
            f"to {cache_path}. Train with --teacher_embedding_cache {cache_path}."
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def _load_teacher_processor(teacher_args):
    from src.model.processor import load_processor

    return load_processor(teacher_args, None)


if __name__ == "__main__":
    main()
