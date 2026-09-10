"""Turn the per-subset `*_score.json` files into the Table 1 layout.

`tools/eval_mmeb.py` writes one `{subset}_score.json` per subset into its
`--encode_output_path`. This reads them back and prints the grouped table with
the per-group averages, which is what actually goes into the paper.
"""

import json
import os

from src.evaluation.benchmarks import groups_covering


def read_scores(score_dir, subsets):
    """`{subset: accuracy}` for the subsets that have a score file on disk."""
    scores = {}
    for subset in subsets:
        path = os.path.join(score_dir, f"{subset}_score.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                scores[subset] = float(json.load(handle)["acc"])
        except (ValueError, KeyError, OSError):
            # A truncated file from an interrupted run is a missing score, not
            # a reason to lose the twenty that did finish.
            continue
    return scores


def build_summary(scores, subsets):
    """Nested `{group: {"columns": {...}, "avg": float}}` plus the overall mean.

    The group average is over the subsets that actually have a score, and
    `missing` names the ones that do not, so a partial run reports as partial
    instead of quietly averaging fewer numbers.
    """
    summary = {}
    for name, heading, columns in groups_covering(subsets):
        present = {
            label: scores[subset]
            for label, subset in columns.items()
            if subset in scores
        }
        missing = [subset for subset in columns.values() if subset not in scores]
        summary[name] = {
            "heading": heading,
            "columns": present,
            "subsets": {label: subset for label, subset in columns.items()},
            "avg": sum(present.values()) / len(present) if present else None,
            "missing": missing,
        }
    all_scores = list(scores.values())
    return {
        "groups": summary,
        "overall_avg": sum(all_scores) / len(all_scores) if all_scores else None,
        "num_subsets": len(all_scores),
    }


def format_table(summary, title=None):
    """The grouped table as text, one block per group, percentages."""
    lines = []
    if title:
        lines.append(title)
    for group in summary["groups"].values():
        labels = list(group["subsets"])
        cells = [
            f"{group['columns'][label] * 100:.1f}"
            if label in group["columns"]
            else "--"
            for label in labels
        ]
        labels.append("Avg")
        cells.append(f"{group['avg'] * 100:.1f}" if group["avg"] is not None else "--")
        widths = [max(len(a), len(b)) for a, b in zip(labels, cells)]
        header = "  ".join(l.rjust(w) for l, w in zip(labels, widths))
        row = "  ".join(c.rjust(w) for c, w in zip(cells, widths))
        lines.append("")
        lines.append(group["heading"])
        lines.append(header)
        lines.append(row)
        if group["missing"]:
            lines.append(f"  missing: {', '.join(group['missing'])}")
    if summary["overall_avg"] is not None:
        lines.append("")
        lines.append(
            f"Overall average over {summary['num_subsets']} subsets: "
            f"{summary['overall_avg'] * 100:.1f}"
        )
    return "\n".join(lines)


def write_summary(score_dir, subsets, title=None, filename="summary.json"):
    """Read, summarise, write `summary.json` next to the scores, return the text."""
    summary = build_summary(read_scores(score_dir, subsets), subsets)
    os.makedirs(score_dir, exist_ok=True)
    with open(os.path.join(score_dir, filename), "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary, format_table(summary, title)
