# Adding a distillation method

Two files. A criterion module, and one line in the registry.

## 1. Write the criterion

`src/criterions/my_method.py`:

```python
import torch.nn.functional as F
from src.criterions.base import CriterionContext, DistillCriterion


class MyMethodLoss(DistillCriterion):
    """--kd_loss_type my_method. One line on what it distils."""

    def __init__(self, args):
        super().__init__(args)          # sets self.kd_loss_weight from --kd_weight
        self.alpha = args.my_method_alpha

    def kd_loss(self, ctx: CriterionContext):
        return {
            "kd_loss": F.mse_loss(ctx.student_qry, ctx.project_teacher(ctx.teacher_qry)),
            "my_alignment_term": ...,   # anything extra is logged, not optimised
        }
```

The base class has already done, once, for both the query and the positive side:

* run the student and pooled its output,
* fetched the teacher's embeddings — from the model, or from
  `--teacher_embedding_cache` if one is in use, without the criterion having to
  know which,
* widened the batch across ranks with a gather that keeps this rank's slice
  differentiable,
* computed the in-batch contrastive loss.

`forward` then returns
`loss = contrastive_loss + kd_weight * kd_loss`, plus every other key you
returned, and the training loop logs all of them by name. There is nothing to
add to the trainer to see a new term in the progress bar.

### What is on `ctx`

| attribute | what it is |
| --- | --- |
| `ctx.student_qry`, `ctx.student_pos` | pooled student embeddings, gathered across ranks |
| `ctx.teacher_qry`, `ctx.teacher_pos` | pooled teacher embeddings, gathered the same way |
| `ctx.student_qry_out`, `ctx.student_pos_out` | the full `(pooled, image_features, attentions, hidden_states)` tuples, **not** gathered |
| `ctx.batch` | the collated batch, for raw `input_ids` / `pixel_values` |
| `ctx.distiller` | the `Distiller`, for `encode_teacher` or the projectors |
| `ctx.project_teacher(t)` | teacher embeddings mapped into the student space through the `t2s` projector |
| `ctx.student_dim` | student embedding width |
| `ctx.zeros()` | a float32 scalar zero on the right device, for a disabled term |

A method that needs the teacher's *hidden states* or *attentions* rather than
its final embedding sets `fetch_teacher_reps = False` on the class and calls
`ctx.distiller.teacher.encode_input(ctx.batch["teacher_inputs"][side])` itself,
inside `torch.no_grad()`.

Subclassing is optional. A criterion can still be a plain `nn.Module` with a
`forward(self, distiller, input_data)` returning a dict containing `loss` — that
is what the older methods in this directory do. The helpers in
`src/criterions/base.py` (`pooled`, `gather_with_grad`,
`in_batch_contrastive_loss`, `project_teacher`) are importable on their own.

## 2. Add the flags

New hyper-parameters go on `TrainingArguments` in `src/arguments.py`, prefixed
with the method name so `--help` stays readable:

```python
my_method_alpha: float = field(default=1.0, metadata={"help": "..."})
```

## 3. Register it

One entry in `_SPECS` in `src/criterions/registry.py`:

```python
CriterionSpec(
    name="my_method",
    module="src.criterions.my_method",
    cls="MyMethodLoss",
    student_attentions=False,
    teacher_attentions=False,
    teacher_embedding_only=True,
    summary="What it distils, in one line.",
),
```

That single entry is what makes `--kd_loss_type my_method` work, and it is also
what the rest of the stack reads:

| field | what it controls |
| --- | --- |
| `student_attentions` | `Distiller` asks the student backbone for attention matrices. **Expensive**: it pins the model to the eager attention kernel and keeps a `(B, heads, L, L)` tensor per layer alive through the backward pass. Leave `False` unless the loss really indexes them. |
| `teacher_attentions` | the same for the teacher — forward only, so cheaper, but still eager. |
| `teacher_embedding_only` | the method can train against `--teacher_embedding_cache`: no teacher model in the process, no teacher forward, no teacher image preprocessing, and its weights out of GPU memory. Only set this if the criterion truly reads nothing but the pooled teacher embedding. |
| `needs_tokenizer` | `Distiller.forward` passes `tokenizer=` into the criterion. |

The defaults are the conservative ones, so an incomplete entry costs speed, not
correctness. The one exception is `teacher_embedding_only`: setting it on a
method that reaches for `distiller.teacher` gives `AttributeError: 'NoneType'`
at the first step of a cached run. `tools/misc/test_teacher_cache.py` checks for
exactly that.

Modules are imported lazily, so a method that needs spacy or numba costs nothing
to anyone not running it.

## 4. Add the launchers

`scripts/train/<method>/<student>_<task>.sh` — four of them, one per cell of
`{fastvlm, llava_onevision} × {cls, vqa}`. Copy the closest existing method's
four. If yours is `teacher_embedding_only`, copy `scripts/train/rkd/` or
`scripts/train/uld/`: they carry the `TEACHER_CACHE` plumbing, which is the
largest single speedup available and is shared between every cache-compatible
method (one cache per cell, any number of methods and seeds).

The per-student values are not free choices — batch size and image resolution
come from Table 6 (Table 7 for EM-KD) and `tools/check_paper_settings.py` fails
on drift. [scripts/train/README.md](../scripts/train/README.md) has them.

**Before writing the LLaVA-OneVision pair, decide what your criterion assumes
about the student's token layout.** The two students differ: FastVLM pads right,
keeps a single `-200` image placeholder in `input_ids` and expands it inside the
model, so its hidden states are `[vision][text][pad]` and longer than
`input_ids`; LLaVA-OneVision pads left and expands the image token in the
processor, so its hidden states are `[pad][vision][text]` and line up 1:1 with
`input_ids`. A criterion that only reads `ctx.student_qry` / `ctx.teacher_qry` is
immune. One that slices hidden states, indexes them with `input_ids` positions,
or picks an attention row by counting from the end is not — and gets it silently
wrong on the student it was not written for, which is how EM-KD ended up as two
criteria. Derive the offsets from `model_backbone` rather than hard-coding one
convention, and say which layouts you support in the method's launcher headers.
`src/criterions/span_common.py` already does this for the span criteria:
`is_left_padded(model_backbone)` plus extraction helpers that take the answer,
rather than each criterion hard-coding one student's layout.

## 5. Check it

```bash
python tools/misc/test_training_stack.py     # registry, base class, loop, images
python tools/misc/test_teacher_cache.py      # cache format + registry consistency
bash scripts/train/<yours>.sh --percent_data 0.01   # end-to-end smoke test
```

## What you do *not* have to touch

`tools/train_distill_ddp.py`, `tools/train_distill_no_deepspeed.py` and
everything in `src/training/` are method-independent. The loop hands
`(criterion, batch)` to the `Distiller` and reads back a dict of scalars; it
never learns a method's name. If you find yourself adding an `if kd_loss_type ==`
to any of them, the thing being branched on probably belongs in `CriterionSpec`.
