"""The MMEB benchmark groups reported in Table 1, by name.

One place that knows which subsets belong to which column of the results table,
so `scripts/eval/*.sh`, the post-training eval and `tools/summarize_mmeb.py`
cannot drift apart on the list.

The names are `TIGER-Lab/MMEB-eval` subset names and are passed straight to
`--subset_name`; the short labels are the column headers of the paper table.
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

# Group name -> (table heading, columns). Group names are what `--eval_benchmarks`
# and `scripts/eval/all.sh` accept.
BENCHMARK_GROUPS = OrderedDict(
    [
        ("cls_ind", ("CLS (IND)", CLS_IND)),
        ("vqa_ind", ("VQA (IND)", VQA_IND)),
        ("cls_ood", ("CLS (OOD)", CLS_OOD)),
        ("vqa_ood", ("VQA (OOD)", VQA_OOD)),
    ]
)

# Every subset any group mentions, deduplicated, in table order.
ALL_SUBSETS = [
    subset for _, columns in BENCHMARK_GROUPS.values() for subset in columns.values()
]


def resolve_groups(names):
    """Expand `--eval_benchmarks` values into a flat list of MMEB subset names.

    Accepts group names (`cls_ind`, ...), the alias `all`, the aliases `ind` /
    `ood`, and bare MMEB subset names for a one-off run. Order follows the table,
    and a subset named twice is evaluated once.
    """
    if not names:
        names = ["all"]
    if isinstance(names, str):
        names = [names]

    aliases = {
        "all": list(BENCHMARK_GROUPS),
        "ind": ["cls_ind", "vqa_ind"],
        "iod": ["cls_ind", "vqa_ind"],  # the table's heading spells it IOD
        "ood": ["cls_ood", "vqa_ood"],
        "cls": ["cls_ind", "cls_ood"],
        "vqa": ["vqa_ind", "vqa_ood"],
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
