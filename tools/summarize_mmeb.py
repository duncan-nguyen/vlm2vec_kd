#!/usr/bin/env python3
"""Print the MMEB results table from the score files an eval run left behind.

    python tools/summarize_mmeb.py <score-dir> [<score-dir> ...]

Each directory is a `--encode_output_path` of `tools/eval_mmeb.py`; the CLS and
VQA scripts write to two different ones, so pass both to get the whole table:

    python tools/summarize_mmeb.py CKPT/mmeb_cls CKPT/mmeb_vqa

Subsets with no score file print as `--`, so a partial sweep is readable while
the rest is still running.
"""

# Run directly from the repo root: put the repo root on sys.path so `import src.…`
# resolves without installing the project.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json

from src.evaluation.benchmarks import ALL_SUBSETS, resolve_groups
from src.evaluation.summary import build_summary, format_table, read_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("score_dirs", nargs="+",
                    help="one or more --encode_output_path directories")
    ap.add_argument("--benchmarks", nargs="+", default=["all"],
                    help="groups to report (default: all); see src/evaluation/benchmarks.py")
    ap.add_argument("--json", metavar="PATH",
                    help="also write the summary as JSON to PATH")
    args = ap.parse_args()

    subsets = resolve_groups(args.benchmarks)
    # Later directories win, so re-running one group into a fresh directory
    # overrides the stale copy of it.
    scores = {}
    for score_dir in args.score_dirs:
        scores.update(read_scores(score_dir, ALL_SUBSETS))

    summary = build_summary(scores, subsets)
    print(format_table(summary, title=" + ".join(args.score_dirs)))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
