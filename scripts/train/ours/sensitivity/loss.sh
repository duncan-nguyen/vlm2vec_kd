#!/bin/bash
#
# Loss-term sensitivity for Ours: which of the two terms carries the result.
#
#   L = L_ret + lambda_topo * L_topo      (docs/latex/sections/method.tex)
#
# Three rows per (student, task) cell and seed, with lambda_topo held at
# OURS_WEIGHT -- the lambda sweep is a separate study:
#
#   student_only  L_ret                           --ours_weight 0
#   topo_only     lambda_topo * L_topo            --ours_retrieval_loss False
#   ours          L_ret + lambda_topo * L_topo
#
# Every run goes through the cell's own launcher (scripts/train/ours/<cell>.sh)
# into training/ours/<cell>/<variant>_seed<seed>, the directory the main grid
# uses. A run whose checkpoint-final already exists is skipped, so main-grid
# `ours` and `student_only` rows are reused rather than retrained.
#
#   bash scripts/train/ours/sensitivity/loss.sh
#   CELLS="fastvlm_cls fastvlm_vqa" SEEDS="42 43 44" NUM_GPUS_PER_NODE=8 \
#     bash scripts/train/ours/sensitivity/loss.sh
#   DRY_RUN=1 CELLS="fastvlm_cls llava_onevision_cls" bash scripts/train/ours/sensitivity/loss.sh
#
# Anything after the script name is forwarded to every run, e.g. a smoke test:
#   SEEDS=42 OUTPUT_ROOT=training/ours_smoke bash scripts/train/ours/sensitivity/loss.sh \
#     --percent_data 0.01 --push_to_hub False --eval_after_train False
#
# Environment:
#   CELLS                 launchers to sweep: fastvlm_cls fastvlm_vqa
#                         llava_onevision_cls llava_onevision_vqa   (fastvlm_cls)
#   SEEDS                 space-separated seeds                      (42 43 44)
#   VARIANTS              rows to run                   (student_only topo_only ours)
#   OURS_WEIGHT           lambda_topo for topo_only and ours         (1.0)
#   OUTPUT_ROOT           run directories go under <root>/<cell>/    (training/ours)
#   TEACHER_CACHE_ROOT    caches are <root>/b3_qwen2_2b_<cell><suffix> (cache)
#   TEACHER_CACHE_SUFFIX  e.g. _v2                                   (empty)
#                         a missing cache falls back to the live teacher
#   LOG_DIR               one log per run                  (logs/ours_sensitivity_loss)
#   DRY_RUN               1 = print the plan, launch nothing         (0)
#   NUM_GPUS_PER_NODE, BATCH_SIZE, MMEB_TRAIN_DIR are passed through to the launcher.
set -euo pipefail

CELLS="${CELLS:-fastvlm_cls}"
SEEDS="${SEEDS:-42}"
VARIANTS="${VARIANTS:-student_only topo_only ours}"
OURS_WEIGHT="${OURS_WEIGHT:-1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-training/ours}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-cache}"
TEACHER_CACHE_SUFFIX="${TEACHER_CACHE_SUFFIX:-}"
LOG_DIR="${LOG_DIR:-logs/ours_sensitivity_loss}"
DRY_RUN="${DRY_RUN:-0}"

LAUNCHER_DIR="scripts/train/ours"
if [[ ! -d "$LAUNCHER_DIR" ]]; then
  echo "run from the repo root: $LAUNCHER_DIR not found" >&2
  exit 1
fi

read -r -a cells <<<"$CELLS"
read -r -a seeds <<<"$SEEDS"
read -r -a variants <<<"$VARIANTS"

# Validate the whole grid before the first multi-hour run starts.
for cell in "${cells[@]}"; do
  if [[ ! -f "$LAUNCHER_DIR/$cell.sh" ]]; then
    echo "unknown cell '$cell': $LAUNCHER_DIR/$cell.sh does not exist" >&2
    exit 1
  fi
done
for variant in "${variants[@]}"; do
  case "$variant" in
    student_only|topo_only|ours) ;;
    *) echo "unknown variant '$variant'; expected: student_only topo_only ours" >&2
       exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
stamp() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"; }

failed=()
for cell in "${cells[@]}"; do
  cache="$TEACHER_CACHE_ROOT/b3_qwen2_2b_${cell}${TEACHER_CACHE_SUFFIX}"
  if [[ -f "$cache/meta.json" ]]; then
    cell_cache="$cache"
  else
    cell_cache=""
    stamp "WARN $cell: no teacher cache at $cache, running the teacher live"
  fi

  for seed in "${seeds[@]}"; do
    for variant in "${variants[@]}"; do
      run="${cell}/${variant}_seed${seed}"
      output_dir="$OUTPUT_ROOT/$run"
      log="$LOG_DIR/${cell}_${variant}_seed${seed}.log"

      if [[ -d "$output_dir/checkpoint-final" ]]; then
        stamp "SKIP $run: $output_dir/checkpoint-final exists"
        continue
      fi
      stamp "RUN  $run -> $output_dir (log: $log)"
      if [[ "$DRY_RUN" == "1" ]]; then
        continue
      fi

      if env VARIANT="$variant" SEED="$seed" OURS_WEIGHT="$OURS_WEIGHT" \
          OUTPUT_DIR="$output_dir" TEACHER_CACHE="$cell_cache" \
          bash "$LAUNCHER_DIR/$cell.sh" "$@" 2>&1 | tee "$log"; then
        stamp "DONE $run"
      else
        stamp "FAIL $run (see $log)"
        failed+=("$run")
      fi
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

# Mean +- std of the MMEB averages over seeds, per cell and variant.
python3 - "$OUTPUT_ROOT" "$CELLS" "$VARIANTS" "$SEEDS" <<'PY'
import json
import os
import statistics
import sys

root, cells, variants, seeds = sys.argv[1], *(a.split() for a in sys.argv[2:])


def fmt(values):
    if not values:
        return "--"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} +- {std:.2f} (n={len(values)})"


print("\nloss-term sensitivity (MMEB accuracy from checkpoint-final/mmeb_eval/summary.json)")
for cell in cells:
    print(f"\n{cell}")
    for variant in variants:
        groups, overall = {}, []
        for seed in seeds:
            path = os.path.join(
                root, cell, f"{variant}_seed{seed}",
                "checkpoint-final", "mmeb_eval", "summary.json",
            )
            try:
                with open(path) as handle:
                    summary = json.load(handle)
            except (OSError, ValueError):
                continue
            if summary.get("overall_avg") is not None:
                overall.append(summary["overall_avg"])
            for name, group in summary.get("groups", {}).items():
                if group.get("avg") is not None:
                    groups.setdefault(name, []).append(group["avg"])
        cols = "  ".join(f"{name}={fmt(vals)}" for name, vals in sorted(groups.items()))
        print(f"  {variant:<13} overall={fmt(overall)}  {cols}")
PY

if (( ${#failed[@]} )); then
  stamp "FAILED runs: ${failed[*]}"
  exit 1
fi
stamp "ALL DONE"
