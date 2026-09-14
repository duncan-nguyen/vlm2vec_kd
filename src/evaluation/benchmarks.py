"""The MMEB benchmark groups reported in the results tables, by name.

One place that knows which subsets belong to which column of the results table,
so `scripts/eval/*.sh`, the post-training eval and `tools/summarize_mmeb.py`
cannot drift apart on the list.

The names are `TIGER-Lab/MMEB-eval` subset names and are passed straight to
`--subset_name`; the short labels are the column headers of the paper table.
The IND/OOD split and the per-subset modalities are documented in
docs/datasets.md.
"""

from collections import OrderedDict

# Column header -> MMEB-eval subset name, in the order the table prints them.
CLS_IND = OrderedDict(
    [
        ("IN-1K", "ImageNet-1K"),
        ("N24News", "N24News"),
        ("Hateful", "HatefulMemes"),
        ("VOC07", "VOC2007"),
        ("SUN397", "SUN397"),
    ]
)

VQA_IND = OrderedDict(
    [
        ("OK", "OK-VQA"),
        ("A-OK", "A-OKVQA"),
        ("Doc", "DocVQA"),
        ("I-VQA", "InfographicsVQA"),
        ("Chart", "ChartQA"),
        ("Vis7W", "Visual7W"),
    ]
)

CLS_OOD = OrderedDict(
    [
        ("Place365", "Place365"),
        ("IN-A", "ImageNet-A"),
        ("IN-R", "ImageNet-R"),
        ("ObjectNet", "ObjectNet"),
        ("Country211", "Country211"),
    ]
)

VQA_OOD = OrderedDict(
    [
        ("ScienceQA", "ScienceQA"),
        ("VizWiz", "VizWiz"),
        ("GQA", "GQA"),
        ("TextVQA", "TextVQA"),
    ]
)

# Retrieval, in the order of VLM2Vec's Table 1.
RET_IND = OrderedDict(
    [
        ("VisDial", "VisDial"),
        ("CIRR", "CIRR"),
        ("VN-t2i", "VisualNews_t2i"),
        ("VN-i2t", "VisualNews_i2t"),
        ("COCO-t2i", "MSCOCO_t2i"),
        ("COCO-i2t", "MSCOCO_i2t"),
        ("NIGHTS", "NIGHTS"),
        ("WebQA", "WebQA"),
    ]
)

RET_OOD = OrderedDict(
    [
        ("OVEN", "OVEN"),
        ("FashionIQ", "FashionIQ"),
        ("EDIS", "EDIS"),
        ("Wiki-SS", "Wiki-SS-NQ"),
    ]
)

# Visual grounding. `MSCOCO` here is the grounding subset, not the retrieval
# subsets `MSCOCO_i2t` / `MSCOCO_t2i`. OOD order follows HieRD's Table 14.
GD_IND = OrderedDict(
    [
        ("COCO", "MSCOCO"),
    ]
)

GD_OOD = OrderedDict(
    [
        ("RefCOCO", "RefCOCO"),
        ("RefCOCO-M", "RefCOCO-Matching"),
        ("V7W-Point", "Visual7W-Pointing"),
    ]
)

# Group name -> (table heading, columns). Group names are what `--eval_benchmarks`
# and `scripts/eval/run_group.sh` accept. The first four keep the order the
# CLS/VQA tables have always printed in.
BENCHMARK_GROUPS = OrderedDict(
    [
        ("cls_ind", ("CLS (IND)", CLS_IND)),
        ("vqa_ind", ("VQA (IND)", VQA_IND)),
        ("cls_ood", ("CLS (OOD)", CLS_OOD)),
        ("vqa_ood", ("VQA (OOD)", VQA_OOD)),
        ("ret_ind", ("RET (IND)", RET_IND)),
        ("ret_ood", ("RET (OOD)", RET_OOD)),
        ("gd_ind", ("GD (IND)", GD_IND)),
        ("gd_ood", ("GD (OOD)", GD_OOD)),
    ]
)

# Every subset any group mentions, deduplicated, in table order.
ALL_SUBSETS = [
    subset for _, columns in BENCHMARK_GROUPS.values() for subset in columns.values()
]

# The MMEB-*train* subsets each task trains on. Separate from the groups above
# because the train and eval repos spell two of them differently -- `ImageNet_1K`
# against `ImageNet-1K` -- so the eval names cannot be reused to recognise a
# training run. Mirrors the presets in scripts/data/download_mmeb.py.
TRAIN_TASKS = {
    "cls": {"ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397"},
    "vqa": {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"},
    "ret": {
        "VisDial",
        "CIRR",
        "VisualNews_t2i",
        "VisualNews_i2t",
        "MSCOCO_t2i",
        "MSCOCO_i2t",
        "NIGHTS",
        "WebQA",
    },
    "grounding": {"MSCOCO"},
}

# The benchmark groups a run trained on one task is evaluated on.
TASK_GROUPS = {
    "cls": ["cls_ind", "cls_ood"],
    "vqa": ["vqa_ind", "vqa_ood"],
    "ret": ["ret_ind", "ret_ood"],
    "grounding": ["gd_ind", "gd_ood"],
}


def matched_tasks(subsets):
    """The tasks whose training subsets appear in `subsets`, in TRAIN_TASKS order."""
    present = set(subsets or [])
    return [task for task, names in TRAIN_TASKS.items() if present & names]


def infer_task(subsets):
    """`"cls"`, `"vqa"`, `"ret"`, `"grounding"`, `"mixed"` or None, from a run's
    `--subset_name`.

    Used to name a checkpoint's directory on the Hub, so an unrecognised subset
    list is not an error -- it just means the caller has to say what the task is.
    """
    if not subsets:
        return None
    matched = matched_tasks(subsets)
    if len(matched) == 1:
        return matched[0]
    return "mixed" if matched else None


def resolve_groups(names, train_subsets=None):
    """Expand `--eval_benchmarks` values into a flat list of MMEB subset names.

    Accepts group names (`cls_ind`, ...), the aliases below, and bare MMEB subset
    names for a one-off run. Order follows the table, and a subset named twice is
    evaluated once.

    `auto` evaluates what the run trained on: the IND and OOD groups of every
    task that `train_subsets` (the run's `--subset_name`) belongs to. A run whose
    subsets match no task falls back to `all`, so an unrecognised run is
    over-evaluated rather than silently not evaluated.
    """
    if not names:
        names = ["auto"]
    if isinstance(names, str):
        names = [names]

    tasks = matched_tasks(train_subsets)
    aliases = {
        "all": list(BENCHMARK_GROUPS),
        "auto": [g for task in tasks for g in TASK_GROUPS[task]] or list(BENCHMARK_GROUPS),
        "ind": [g for g in BENCHMARK_GROUPS if g.endswith("_ind")],
        "iod": [g for g in BENCHMARK_GROUPS if g.endswith("_ind")],  # the table's heading spells it IOD
        "ood": [g for g in BENCHMARK_GROUPS if g.endswith("_ood")],
        "cls": TASK_GROUPS["cls"],
        "vqa": TASK_GROUPS["vqa"],
        "ret": TASK_GROUPS["ret"],
        "gd": TASK_GROUPS["grounding"],
        "grounding": TASK_GROUPS["grounding"],
    }

    requested = []
    for name in names:
        key = name.strip()
        for group in aliases.get(key.lower(), [key]):
            if group in BENCHMARK_GROUPS:
                requested.extend(BENCHMARK_GROUPS[group][1].values())
            elif group in ALL_SUBSETS:
                requested.append(group)
            else:
                raise ValueError(
                    f"unknown benchmark '{group}'; expected one of "
                    f"{sorted(set(aliases) | set(BENCHMARK_GROUPS))} "
                    f"or an MMEB-eval subset name"
                )

    seen, ordered = set(), []
    for subset in requested:
        if subset not in seen:
            seen.add(subset)
            ordered.append(subset)
    return ordered


def groups_covering(subsets):
    """The (heading, columns) pairs that have at least one of `subsets` in them."""
    wanted = set(subsets)
    return [
        (name, heading, columns)
        for name, (heading, columns) in BENCHMARK_GROUPS.items()
        if wanted & set(columns.values())
    ]
