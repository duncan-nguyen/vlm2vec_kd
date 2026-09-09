# Training launchers

One launcher per cell of `method × student × task`:

```
scripts/train/<method>/<student>_<task>.sh
```

`<student>` is `fastvlm` or `llava_onevision`, `<task>` is `cls` or `vqa`. Every
launcher is run **from the repo root** and forwards anything after the script
name to the trainer, so a smoke test is one flag away:

```bash
bash scripts/train/rkd/fastvlm_cls.sh --percent_data 0.01
```

All of them are pinned to the paper's configuration and checked mechanically:

```bash
python tools/check_paper_settings.py     # non-zero exit on any drift
```

## The matrix

Nine methods × two students × two tasks = 36 launchers. The teacher is
`raghavlite/B3_Qwen2_2B` (`qwen2_vl`, hidden 1536) in every cell; both students
are 0.5B with hidden 896, so `--student_hidden_dim` / `--teacher_hidden_dim`
never need setting.

| directory | `--kd_loss_type` | trainer | projector | teacher cache |
| --- | --- | --- | --- | --- |
| `rkd/` | `contrastive_rkd` | autocast | – | yes |
| `uld/` | `universal_logit` | autocast | – | yes |
| `cmtop/` | `cmtop` | autocast | `projector_config_emo.json` | yes |
| `talas/` | `talas` | autocast | *(built by the criterion)* | yes |
| `emkd/` | `em_kd` \| `em_kd_llava_ov` | ddp | `projector_config_emo.json` | no |
| `emo/` | `emo_loss` | ddp | `projector_config_emo.json` | no |
| `hierd/` | `span_propose_attn` | ddp | per-layer list | no |
| `pdtw/` | `proposal_dtw` | ddp | `projector_config.json` | no |
| `pproj/` | `proposal_proj` | ddp | `projector_config.json` | no |

*trainer*: `ddp` = `tools/train_distill_ddp.py` (no autocast), `autocast` =
`tools/train_distill_no_deepspeed.py` (bf16 autocast). The two differ only in the
precision policy; each method stays on the entrypoint its first launcher used so
its numbers remain comparable.

*projector*: the span criteria index `distiller.projectors` **by layer**, so they
must not pass `--projector_config_path` — the list form is built from
`--teacher_layer_mapping`. `pdtw` / `pproj` need the `t2s_img` and `t2s_txt`
entries, which only the full `projector_config.json` has. `talas` builds its own
projections — one per teacher-anchored layer, sized from
`--talas_num_tamd_layers` — through `DistillCriterion.build_parameters`, and must
not pass `--projector_config_path` either: a `t2s` registered and never used
makes DDP abort with "expected to have finished reduction". Everything else needs
just `t2s`.

`src/criterions/registry.py` holds two more span criteria that no launcher
covers: `span_propose` (spans not weighted by the teacher's attention) and
`span_propose_attn_only_phrase` (phrase spans only). They are ablations of HieRD
rather than methods of their own, and they read exactly the same arguments, so
run one by overriding the flag on a `hierd/` launcher:

```bash
bash scripts/train/hierd/fastvlm_cls.sh --kd_loss_type span_propose
```

## Per-student settings

Not free choices — Table 6 (Table 7 for EM-KD) fixes them per student, and the
checker enforces them:

| | FastVLM-0.5B | LLaVA-OneVision-0.5B |
| --- | --- | --- |
| `--model_name` | `apple/FastVLM-0.5B` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |
| `--model_backbone` | `llava_qwen2` | `llava_onevision` |
| batch size (Table 6) | 16 | 8 |
| batch size (EM-KD, Table 7) | 8 | 4 |
| `--image_resolution` (Table 6) | 448 | 336 |
| `--image_resolution` (EM-KD, Table 7) | 448 | 128 |

Pass an explicit pixel budget, never the `high`/`mid`/`low` presets: they
disagree between the training and evaluation code paths.

## Model-pair compatibility

The two students hand a criterion **different token layouts**, and a criterion
that slices hidden states by position is only correct for the layout it was
written against:

| | FastVLM (`llava_qwen2`) | LLaVA-OneVision |
| --- | --- | --- |
| padding side | right (`pad_sequence`, [processing_fastvlm.py:79](../../src/model/llava/processing_fastvlm.py#L79)) | left ([processor.py:144](../../src/model/processor.py#L144)) |
| image in `input_ids` | one `-200` placeholder, expanded by the model | expanded by the processor |
| `len(input_ids)` vs hidden states | shorter (by `N_image - 1`) | equal |
| hidden-state layout | `[vision][text][pad]` | `[pad][vision][text]` |

That gives three groups of methods:

**1. Layout-independent — correct for both students.** `contrastive_rkd`,
`universal_logit`, `cmtop` read nothing but the pooled query/candidate
embeddings. `talas` reads one pooled embedding *per student layer*, but obtains
each through the student's own pooling and attention mask — the identical call
the model makes for its final embedding — so it inherits that pooling's layout
handling rather than assuming one. These four are also the ones that can train
from a teacher cache.

**2. Layout-specific, with a variant per student.** EM-KD ships as two
criteria: `em_kd` slices the student from the front (`[vision][text][pad]`) and
`em_kd_llava_ov` slices it from the back (`[pad][vision][text]`). The launchers
pick the right one; swapping them silently trains on padding.

**3. Layout-specific, single variant — one student is unverified.** These cells
exist so the matrix is complete, and each such launcher carries a `# !`
compatibility note in its header naming the exact assumption:

| criterion | FastVLM | LLaVA-OneVision |
| --- | --- | --- |
| `emo_loss` | ⚠ maps teacher tokens onto student `input_ids` positions and indexes hidden states with them; off by `N_image - 1` here | ok |
| `proposal_dtw`, `proposal_proj` | ⚠ same `input_ids`-position indexing as `emo_loss` | ⚠ reads the last non-pad attention row as `-(num_pad + 1)`, which is the right-padded convention |

The span criteria (`span_propose*`, HieRD) used to sit in this table with a ⚠ on
LLaVA-OneVision. They no longer do: their extraction helpers moved to
[src/criterions/span_common.py](../../src/criterions/span_common.py) and take the
padding side of the actual student instead of assuming the right-padded one.

⚠ does not mean the run crashes — it means the contrastive term trains normally
while the KD term reads the wrong positions, which is worse. Fixing this
properly means making the extraction helpers layout-aware (deriving the offsets
from `model_backbone` instead of assuming one convention) rather than editing
the launchers.

A ⚠ is a property of the criterion, not of the launcher, so it applies equally to
a cell reached by overriding `--kd_loss_type` on another method's launcher.

## Teacher embedding cache

`contrastive_rkd`, `universal_logit`, `cmtop` and `talas` read nothing from the teacher
but its final embedding, so they can train with **no teacher model in the
process**: no teacher forward, no teacher-side image preprocessing, no teacher
weights on the device.

```bash
TASK=cls STUDENT=fastvlm TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls \
  bash scripts/data/precompute_teacher_embeddings.sh

TEACHER_CACHE=cache/b3_qwen2_2b_fastvlm_cls bash scripts/train/rkd/fastvlm_cls.sh
```

The cache is fingerprinted against the teacher settings, the subset list, the
image settings and `percent_data`, and is *refused* rather than silently misused
under a configuration it was not built for. Since `--image_resolution` differs
per student and the subset list per task, the four cache-capable cells each need
their own cache directory:

| cache | built with |
| --- | --- |
| `cache/b3_qwen2_2b_fastvlm_cls` | `TASK=cls STUDENT=fastvlm` (448) |
| `cache/b3_qwen2_2b_fastvlm_vqa` | `TASK=vqa STUDENT=fastvlm` (448) |
| `cache/b3_qwen2_2b_llava_onevision_cls` | `TASK=cls STUDENT=llava_onevision` (336) |
| `cache/b3_qwen2_2b_llava_onevision_vqa` | `TASK=vqa STUDENT=llava_onevision` (336) |

One cache then serves every cache-capable method, CMTop variant and seed at that
cell. Criteria that need the teacher's hidden states are refused, not served
wrong data.

## Data

Download only what a cell needs — the tool reads the subset list straight out of
the launcher:

```bash
python scripts/data/download_mmeb.py --for scripts/train/pproj/llava_onevision_vqa.sh
```

`MMEB_TRAIN_DIR` (default `./vlm2vec_train/MMEB-train`) overrides where the
images live, per run.

## Environment variables

Every generated launcher accepts:

| variable | default | meaning |
| --- | --- | --- |
| `MMEB_TRAIN_DIR` | `./vlm2vec_train/MMEB-train` | MMEB-train images |
| `OUTPUT_DIR` | `training/<method>_<student>_<task>` | checkpoint directory |
| `SEED` | `42` | random seed |
| `NUM_GPUS_PER_NODE` | `1` | `torchrun --nproc_per_node` |
| `TEACHER_CACHE` | *(unset)* | cache-capable methods only; unset runs the teacher live |

The `cmtop/` launchers add `VARIANT` (`student_only`, `endpoint`, `vsp`,
`pointcloud_h0`, `cmtop_h0`, `cmtop_h0_h1`), `CMTOP_WEIGHT` and `KD_WEIGHT` —
see [cmtop/README.md](cmtop/README.md).

The `talas/` launchers add `VARIANT` (`talas`, `no_asam`, `sam`, `no_lasd`,
`no_tamd`), `TALAS_CONTRASTIVE_WEIGHT`, `TAMD_WEIGHT`, `LASD_WEIGHT`,
`TAMD_LAYERS` and `SAM_RHO` — see [talas/README.md](talas/README.md).

`TORCH_DISTRIBUTED_DEBUG=DETAIL` is not exported by the newer launchers; export
it yourself when you need DDP's bucket diagnostics, since it costs speed.
