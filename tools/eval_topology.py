"""Report how much of the teacher's retrieval structure a student preserves.

This is the "topology discrepancy + neighborhood preservation" column of the
CMTop experiment plan (docs/cross_modal_topological_distillation.md, section 4).
It reads the query/target embedding dumps that `tools/eval_mmeb.py` already
writes to `--encode_output_path`, so no re-encoding is needed:

    python tools/eval_topology.py \
        --teacher_embeddings runs/teacher/emb \
        --student_embeddings runs/student/emb \
        --dataset_name TIGER-Lab/MMEB-eval \
        --subsets ImageNet-1K N24News \
        --output runs/student/topology_report.json

The dumps hold pools keyed by (text, image path). Queries and candidates are
aligned by those keys across teacher and student, while repeated candidates are
kept once. Their pool sizes need not match: classification is naturally a
many-query-to-few-label relation.

Aligned `.npy` files can be passed directly instead, for embeddings produced
outside the MMEB pipeline:

    python tools/eval_topology.py \
        --teacher_qry t_q.npy --teacher_tgt t_c.npy \
        --student_qry s_q.npy --student_tgt s_c.npy
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import pickle

import numpy as np

from src.evaluation.topology_metrics import (
    batched_topology_discrepancy,
    neighborhood_preservation,
)


def load_mmeb_dump(embeddings_dir, subset):
    """Load `{subset}_qry` / `{subset}_tgt` into (text, img_path) -> embedding maps."""
    maps = []
    for side in ("qry", "tgt"):
        path = os.path.join(embeddings_dir, f"{subset}_{side}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run tools/eval_mmeb.py with "
                f"--encode_output_path {embeddings_dir} first"
            )
        with open(path, "rb") as f:
            tensors, index = pickle.load(f)
        maps.append(
            {
                (entry["text"], entry["img_path"]): np.asarray(t)
                for t, entry in zip(tensors, index)
            }
        )
    return maps[0], maps[1]


def align_pairs(eval_data, teacher_maps, student_maps):
    """Build teacher/student-aligned query and canonical candidate pools."""
    (teacher_qry_map, teacher_tgt_map) = teacher_maps
    (student_qry_map, student_tgt_map) = student_maps

    rows = {"teacher_q": [], "teacher_c": [], "student_q": [], "student_c": []}
    seen_qry, seen_tgt = set(), set()
    skipped = 0
    for row in eval_data:
        qry_key = (row["qry_text"], row["qry_img_path"])
        if qry_key not in seen_qry:
            if qry_key in teacher_qry_map and qry_key in student_qry_map:
                rows["teacher_q"].append(teacher_qry_map[qry_key])
                rows["student_q"].append(student_qry_map[qry_key])
                seen_qry.add(qry_key)
            else:
                skipped += 1

        target_texts = row["tgt_text"]
        target_images = row["tgt_img_path"]
        if not isinstance(target_texts, list):
            target_texts = [target_texts]
        if not isinstance(target_images, list):
            target_images = [target_images]
        if len(target_texts) != len(target_images):
            raise ValueError(
                "tgt_text and tgt_img_path must contain the same number of targets"
            )
        for target_text, target_image in zip(target_texts, target_images):
            tgt_key = (target_text, target_image)
            if tgt_key in seen_tgt:
                continue
            if tgt_key in teacher_tgt_map and tgt_key in student_tgt_map:
                rows["teacher_c"].append(teacher_tgt_map[tgt_key])
                rows["student_c"].append(student_tgt_map[tgt_key])
                seen_tgt.add(tgt_key)
            else:
                skipped += 1

    if not rows["teacher_q"] or not rows["teacher_c"]:
        raise ValueError("no labelled query/candidate relation exists in all dumps")
    return {k: np.stack(v, axis=0) for k, v in rows.items()}, skipped


def report(pairs, args):
    return {
        "num_queries": int(pairs["teacher_q"].shape[0]),
        "num_candidates": int(pairs["teacher_c"].shape[0]),
        "topology_discrepancy": batched_topology_discrepancy(
            pairs["teacher_q"],
            pairs["teacher_c"],
            pairs["student_q"],
            pairs["student_c"],
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            seed=args.seed,
            h1_topk=args.h1_topk or None,
            partition_quantiles=tuple(args.partition_quantiles),
        ),
        "neighborhood_preservation": neighborhood_preservation(
            pairs["teacher_q"],
            pairs["teacher_c"],
            pairs["student_q"],
            pairs["student_c"],
            ks=tuple(args.recall_ks),
            rank_corr_queries=args.rank_corr_queries,
            seed=args.seed,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--teacher_embeddings", help="encode_output_path of the teacher run"
    )
    parser.add_argument(
        "--student_embeddings", help="encode_output_path of the student run"
    )
    parser.add_argument("--dataset_name", default="TIGER-Lab/MMEB-eval")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--subsets", nargs="+", default=[])
    parser.add_argument(
        "--teacher_qry", help="aligned .npy, alternative to the MMEB dumps"
    )
    parser.add_argument("--teacher_tgt")
    parser.add_argument("--student_qry")
    parser.add_argument("--student_tgt")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="filtration size; match the training batch size",
    )
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--h1_topk", type=int, default=0, help="0 uses every H1 birth")
    parser.add_argument(
        "--partition_quantiles",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2],
        help="teacher distance quantiles used for component-partition ARI",
    )
    parser.add_argument("--recall_ks", type=int, nargs="+", default=[1, 5, 10, 50])
    parser.add_argument("--rank_corr_queries", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the report as JSON here")
    args = parser.parse_args()

    results = {}
    array_paths = (
        args.teacher_qry,
        args.teacher_tgt,
        args.student_qry,
        args.student_tgt,
    )
    if any(array_paths) and not all(array_paths):
        parser.error("pass all four teacher/student query/target .npy paths")

    if all(array_paths):
        pairs = {
            "teacher_q": np.load(args.teacher_qry),
            "teacher_c": np.load(args.teacher_tgt),
            "student_q": np.load(args.student_qry),
            "student_c": np.load(args.student_tgt),
        }
        lengths = {k: v.shape[0] for k, v in pairs.items()}
        if lengths["teacher_q"] != lengths["student_q"] or lengths[
            "teacher_c"
        ] != lengths["student_c"]:
            raise ValueError(
                "teacher/student arrays must be aligned within each side, got "
                f"{lengths}"
            )
        results["_arrays"] = report(pairs, args)
    else:
        if not (args.teacher_embeddings and args.student_embeddings and args.subsets):
            parser.error(
                "pass --teacher_embeddings/--student_embeddings/--subsets, "
                "or the four aligned .npy paths"
            )
        from datasets import load_dataset

        for subset in args.subsets:
            eval_data = load_dataset(
                args.dataset_name, subset, split=args.dataset_split
            )
            pairs, skipped = align_pairs(
                eval_data,
                load_mmeb_dump(args.teacher_embeddings, subset),
                load_mmeb_dump(args.student_embeddings, subset),
            )
            subset_report = report(pairs, args)
            subset_report["skipped_examples"] = skipped
            results[subset] = subset_report
            print(f"\n=== {subset} ===")
            print(json.dumps(subset_report, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.output}")
    elif "_arrays" in results:
        print(json.dumps(results["_arrays"], indent=2))


if __name__ == "__main__":
    main()
