#!/usr/bin/env python3
"""Assert that every launcher under scripts/train matches the HieRD paper.

The paper's configuration lives in four tables:

  Table 6  training config for every method except EM-KD, per student
  Table 7  training config for EM-KD, per student
  Table 8  loss weights: alpha = 0.25, lambda_struct = 2.5, lambda_hid = 0.25
  Table 9  layer selection: word layer 0, phrase layers 18/21/24
  Table 4  DBSCAN min_samples default = 8

They are transcribed below so drift is caught mechanically rather than by
re-reading the PDF. Run from the repo root:

    python tools/check_paper_settings.py          # exits non-zero on mismatch
    python tools/check_paper_settings.py --list   # just print what each uses
"""

import argparse
import glob
import os
import re
import sys

# --- paper tables ----------------------------------------------------------
FASTVLM = "apple/FastVLM-0.5B"
LLAVA_OV = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

# Table 6: everything except EM-KD
TABLE_6 = {
    FASTVLM:  dict(num_train_epochs="1", learning_rate="1e-4", projector_lr="5e-4",
                   per_device_train_batch_size="16", lr_scheduler_type="cosine",
                   warmup_ratio="0.03", weight_decay="0.01",
                   lora_r="64", lora_alpha="64", image_resolution="448"),
    LLAVA_OV: dict(num_train_epochs="1", learning_rate="1e-4", projector_lr="5e-4",
                   per_device_train_batch_size="8", lr_scheduler_type="cosine",
                   warmup_ratio="0.03", weight_decay="0.01",
                   lora_r="64", lora_alpha="64", image_resolution="336"),
}
# Table 7: EM-KD only
TABLE_7 = {
    FASTVLM:  dict(TABLE_6[FASTVLM],  per_device_train_batch_size="8", image_resolution="448"),
    LLAVA_OV: dict(TABLE_6[LLAVA_OV], per_device_train_batch_size="4", image_resolution="128"),
}
# Tables 8/9/4, HieRD only
HIERD_EXTRA = dict(kd_weight="2.5",                 # lambda_struct
                   w_cross_modal_loss="2.5",        # lambda_struct also scales L_cross
                   student_layer_mapping="0 18 21 24",   # Table 9
                   split_layer_mapping="0 1 4 4 4",
                   min_samples_dbscan_teacher="8")       # Table 4

# The whole training set is used; no subsampling is described in the paper.
COMMON = dict(percent_data="1.0")

CLS = ["ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397"]
VQA = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"]


def flag(src, name):
    m = re.search(
        rf'--{name}\s+((?:"[^"]*"|[^\s\\]+)(?:[ \t]+(?:"[^"]*"|[A-Za-z0-9_\-.]+))*)', src)
    if not m:
        return None
    return " ".join(a or b for a, b in re.findall(r'"([^"]*)"|([^\s\\"]+)', m.group(1)))


def check(path):
    src = open(path).read()
    model = flag(src, "model_name")
    kd = flag(src, "kd_loss_type")
    if model not in TABLE_6:
        return [f"unknown student --model_name {model!r}"], {}

    expected = dict(TABLE_7[model] if kd in ("em_kd", "em_kd_llava_ov") else TABLE_6[model])
    expected.update(COMMON)
    if kd == "span_propose_attn":
        expected.update(HIERD_EXTRA)

    # the task is implied by the directory-independent subset list
    subs = (flag(src, "subset_name") or "").split()
    task = "CLS" if set(subs) == set(CLS) else "VQA" if set(subs) == set(VQA) else None

    problems = []
    if task is None:
        problems.append(f"subset_name is neither the CLS nor the VQA list: {subs}")
    for k, want in sorted(expected.items()):
        got = flag(src, k)
        if got != want:
            problems.append(f"--{k}: {got!r} (paper: {want!r})")
    return problems, dict(model=model, kd=kd, task=task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print settings instead of checking")
    args = ap.parse_args()

    paths = sorted(glob.glob("scripts/train/*/*.sh"))
    if not paths:
        sys.exit("no launchers found under scripts/train/ — run from the repo root")

    failed = 0
    for p in paths:
        problems, info = check(p)
        label = f"{p}  [{info.get('kd')} / {os.path.basename(info.get('model') or '?')} / {info.get('task')}]"
        if args.list:
            print(label)
            continue
        if problems:
            failed += 1
            print(f"FAIL  {label}")
            for x in problems:
                print(f"        {x}")
        else:
            print(f"ok    {label}")

    if not args.list:
        print(f"\n{len(paths) - failed}/{len(paths)} launchers match the paper")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
