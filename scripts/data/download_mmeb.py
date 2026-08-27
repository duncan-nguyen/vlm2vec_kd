#!/usr/bin/env python3
"""Download and extract MMEB image archives.

Replaces the `wget && unzip && rm` chains, which pulled all 47.2 GB of
MMEB-train serially on one connection with no resume and no skip-if-present.

What this does differently:

* **Downloads only what an experiment needs.** `--for <script.sh>` reads the
  `--subset_name` flag out of a training script, so the download always matches
  the run. The current rebuttal sweep needs 15.9 GB, not 47.2 GB.
* **Parallel across files**, and **extraction overlaps downloading** — a zip is
  unpacked while the next one is still in flight.
* **Resumable and idempotent.** Interrupted transfers resume; already-extracted
  subsets are skipped, so re-running is free.
* **Deletes each zip right after extraction**, so peak disk is the extracted
  tree plus one zip rather than everything at once.

Usage
-----
    python scripts/data/download_mmeb.py --preset cls
    python scripts/data/download_mmeb.py --for scripts/train/rebuttal/rebuttal_hierd_grounding.sh
    python scripts/data/download_mmeb.py --subsets MSCOCO VOC2007 --workers 8
    python scripts/data/download_mmeb.py --eval          # MMEB-eval images (7.1 GB)
    python scripts/data/download_mmeb.py --preset all --dry-run

Speed notes
-----------
* Set `HF_TOKEN`. The Hub answers unauthenticated requests with
  "Please set a HF_TOKEN to enable higher rate limits and faster downloads".
* `pip install "huggingface_hub[hf_xet]"` — MMEB is Xet-backed storage, and the
  Xet client fetches chunks in parallel instead of one stream per file.
* `--workers` only helps when the link is not already saturated. If a single
  stream already maxes out the connection, raising it does nothing.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

TRAIN_REPO = "TIGER-Lab/MMEB-train"
EVAL_REPO = "TIGER-Lab/MMEB-eval"

# Compressed sizes in MB, from the Hub API. Used for planning/reporting only.
SIZES_MB = {
    "DocVQA": 12900.4, "VisDial": 6440.7, "N24News": 4389.9, "ImageNet_1K": 3899.0,
    "MSCOCO": 3690.6, "SUN397": 3414.9, "MSCOCO_i2t": 1939.9, "VisualNews_i2t": 1901.1,
    "VisualNews_t2i": 1899.4, "InfographicsVQA": 1312.7, "MSCOCO_t2i": 1205.3,
    "A-OKVQA": 934.5, "ChartQA": 744.7, "Visual7W": 598.0, "NIGHTS": 480.5,
    "OK-VQA": 472.2, "HatefulMemes": 420.2, "CIRR": 271.5, "WebQA": 245.0,
    "VOC2007": 57.2,
}

PRESETS = {
    "cls": ["ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397"],
    "grounding": ["MSCOCO"],
    "vqa": ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"],
    "ret": ["VisDial", "CIRR", "VisualNews_i2t", "VisualNews_t2i",
            "MSCOCO_i2t", "MSCOCO_t2i", "NIGHTS", "WebQA"],
    "all": sorted(SIZES_MB),
}


def subsets_from_script(path):
    """Read the --subset_name values out of a training launcher."""
    src = open(path).read()
    m = re.search(
        r'--subset_name\s+((?:"[^"]*"|[A-Za-z0-9_\-]+)'
        r'(?:[ \t]+(?:"[^"]*"|[A-Za-z0-9_\-]+))*)', src)
    if not m:
        sys.exit(f"error: no --subset_name found in {path}")
    found = [a or b for a, b in re.findall(r'"([^"]*)"|([A-Za-z0-9_\-]+)', m.group(1))]
    unknown = [s for s in found if s not in SIZES_MB]
    if unknown:
        print(f"  note: ignoring unrecognised subset name(s): {', '.join(unknown)}")
    return [s for s in found if s in SIZES_MB]


def human(mb):
    return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def extract(zip_path, dest):
    """Prefer the unzip binary (noticeably faster than zipfile); fall back to Python."""
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", dest], check=True)
    else:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)


def main():
    ap = argparse.ArgumentParser(
        description="Download + extract MMEB image archives",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--subsets", nargs="+", metavar="NAME")
    src.add_argument("--preset", choices=sorted(PRESETS))
    src.add_argument("--for", dest="for_script", metavar="SCRIPT.sh",
                     help="take the subset list from a training launcher")
    src.add_argument("--eval", action="store_true",
                     help="download the MMEB-eval images archive instead (7.1 GB)")
    ap.add_argument("--out", default="./vlm2vec_train/MMEB-train/images",
                    help="extraction directory (default: %(default)s)")
    ap.add_argument("--eval-out", default="./eval_images",
                    help="extraction directory for --eval (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent downloads (default: %(default)s)")
    ap.add_argument("--keep-zips", action="store_true",
                    help="do not delete each archive after extracting it")
    ap.add_argument("--force", action="store_true",
                    help="re-download and re-extract subsets already marked done")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("error: huggingface_hub is required (pip install huggingface_hub)")

    if args.eval:
        repo, out_dir = EVAL_REPO, args.eval_out
        jobs = [("images", "images.zip", 7128.9)]
    else:
        repo, out_dir = TRAIN_REPO, args.out
        names = (args.subsets if args.subsets else
                 PRESETS[args.preset] if args.preset else
                 subsets_from_script(args.for_script))
        unknown = [n for n in names if n not in SIZES_MB]
        if unknown:
            sys.exit(f"error: unknown subset(s): {', '.join(unknown)}\n"
                     f"known: {', '.join(sorted(SIZES_MB))}")
        jobs = [(n, f"images_zip/{n}.zip", SIZES_MB[n]) for n in names]

    marker_dir = os.path.join(out_dir, ".downloaded")
    if not args.dry_run:
        os.makedirs(marker_dir, exist_ok=True)

    pending, skipped = [], []
    for name, remote, mb in jobs:
        if not args.force and os.path.exists(os.path.join(marker_dir, name)):
            skipped.append((name, mb))
        else:
            pending.append((name, remote, mb))

    total_mb = sum(mb for _, _, mb in pending)
    print(f"repo   : {repo}")
    print(f"target : {out_dir}")
    if skipped:
        print(f"skip   : {len(skipped)} already extracted "
              f"({human(sum(mb for _, mb in skipped))}) — pass --force to redo")
    print(f"fetch  : {len(pending)} archive(s), {human(total_mb)} compressed")
    for name, _, mb in sorted(pending, key=lambda j: -j[2]):
        print(f"           {name:<20}{human(mb):>10}")

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("\n  hint: HF_TOKEN is not set — the Hub rate-limits and throttles "
              "unauthenticated downloads.")
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        print("  hint: `pip install \"huggingface_hub[hf_xet]\"` — MMEB is on Xet "
              "storage, whose client downloads chunks in parallel.")

    probe = os.path.abspath(out_dir)
    while not os.path.isdir(probe):            # out_dir may not exist yet
        probe = os.path.dirname(probe)
    free_gb = shutil.disk_usage(probe).free / 1e9
    # Extracted images run well above the compressed size; the zips are removed
    # as we go, so the floor is roughly the extracted tree.
    print(f"\ndisk   : {free_gb:.1f} GB free at {out_dir}")
    if free_gb < total_mb / 1024 * 1.6:
        print("  warning: this may not be enough for the extracted images.")

    if args.dry_run or not pending:
        print("\nnothing to do." if not pending else "\ndry run — nothing downloaded.")
        return

    zip_dir = os.path.join(out_dir, ".zips")
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(marker_dir, exist_ok=True)
    t0 = time.time()
    done_mb = 0.0
    failures = []

    # Downloads and extractions run in separate pools so that unpacking one
    # archive overlaps the transfer of the next.
    with ThreadPoolExecutor(args.workers) as dl_pool, \
            ThreadPoolExecutor(max(2, args.workers // 2)) as ex_pool:

        def fetch(name, remote, mb):
            path = hf_hub_download(repo_id=repo, filename=remote, repo_type="dataset",
                                   local_dir=zip_dir)
            return name, path, mb

        futures = {dl_pool.submit(fetch, n, r, m): n for n, r, m in pending}
        extracting = {}

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                name, path, mb = fut.result()
            except Exception as e:                      # noqa: BLE001
                failures.append((name, f"download failed: {e}"))
                print(f"  !! {name}: download failed: {e}", flush=True)
                continue
            done_mb += mb
            rate = done_mb / max(time.time() - t0, 1e-9)
            print(f"  ↓ {name:<20} {human(mb):>10}  "
                  f"[{done_mb/total_mb*100:5.1f}%, {rate:.1f} MB/s avg]", flush=True)
            extracting[ex_pool.submit(extract, path, out_dir)] = (name, path)

        for fut in as_completed(extracting):
            name, path = extracting[fut]
            try:
                fut.result()
            except Exception as e:                      # noqa: BLE001
                failures.append((name, f"extract failed: {e}"))
                print(f"  !! {name}: extract failed: {e}", flush=True)
                continue
            if not args.keep_zips:
                os.remove(path)
            open(os.path.join(marker_dir, name), "w").close()
            print(f"  ✓ {name} extracted", flush=True)

    if not args.keep_zips:
        shutil.rmtree(zip_dir, ignore_errors=True)

    mins = (time.time() - t0) / 60
    print(f"\n{len(pending) - len(failures)}/{len(pending)} archive(s) in {mins:.1f} min")
    if failures:
        print("failed (re-run to retry — completed subsets are skipped):")
        for name, why in failures:
            print(f"  {name}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
