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

The dumps hold a de-duplicated pool keyed by (text, image path); the eval
dataset says which candidate is the positive for each query, which is what pairs
row i of the query side with row i of the candidate side. That pairing is the
whole point -- the relation graph is only comparable across two models if its
nodes carry the same labels on both sides.

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
    """Build four row-aligned matrices: (teacher_q, teacher_c, student_q, student_c).

    One row per eval example whose query and positive candidate are present in
    both dumps. Rows repeating a query or a positive already seen are dropped:
    a classification subset points thousands of queries at the same label
    embedding, and duplicated nodes sit at distance 0 from each other, which
    makes the H0 barcode say more about the duplication than about the model.
    """
    (teacher_qry_map, teacher_tgt_map) = teacher_maps
    (student_qry_map, student_tgt_map) = student_maps

    rows = {"teacher_q": [], "teacher_c": [], "student_q": [], "student_c": []}
    seen_qry, seen_tgt = set(), set()
    skipped = 0
    for row in eval_data:
        qry_key = (row["qry_text"], row["qry_img_path"])
        tgt_key = (row["tgt_text"][0], row["tgt_img_path"][0])
        if qry_key in seen_qry or tgt_key in seen_tgt:
            continue
        try:
            rows["teacher_q"].append(teacher_qry_map[qry_key])
            rows["teacher_c"].append(teacher_tgt_map[tgt_key])
            rows["student_q"].append(student_qry_map[qry_key])
            rows["student_c"].append(student_tgt_map[tgt_key])
        except KeyError:
            skipped += 1
            continue
        seen_qry.add(qry_key)
        seen_tgt.add(tgt_key)

    if not rows["teacher_q"]:
        raise ValueError("no example was present in all four embedding dumps")
    return {k: np.stack(v, axis=0) for k, v in rows.items()}, skipped


def report(pairs, args):
    return {
        "num_pairs": int(pairs["teacher_q"].shape[0]),
        "topology_discrepancy": batched_topology_discrepancy(
            pairs["teacher_q"],
            pairs["teacher_c"],
            pairs["student_q"],
            pairs["student_c"],
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            seed=args.seed,
            h1_topk=args.h1_topk or None,
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
    parser.add_argument("--recall_ks", type=int, nargs="+", default=[1, 5, 10, 50])
    parser.add_argument("--rank_corr_queries", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the report as JSON here")
    args = parser.parse_args()

    results = {}
    if args.teacher_qry:
        pairs = {
            "teacher_q": np.load(args.teacher_qry),
            "teacher_c": np.load(args.teacher_tgt),
            "student_q": np.load(args.student_qry),
            "student_c": np.load(args.student_tgt),
        }
        lengths = {k: v.shape[0] for k, v in pairs.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"the four arrays must be row-aligned, got {lengths}")
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
