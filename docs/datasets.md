# MMEB dataset splits

The CLS, VQA, RET and grounding subsets of MMEB used in this repository, split into in-distribution (IOD) and out-of-distribution (OOD).

**IOD** subsets have a training split in `TIGER-Lab/MMEB-train`. **OOD** subsets exist only in `TIGER-Lab/MMEB-eval` and are never trained on. MMEB has 36 datasets over 4 meta-tasks: 20 IOD and 16 OOD (Jiang et al., *VLM2Vec*, arXiv 2410.05160, Table 1).

Every eval subset has 1000 queries. The metric is Precision@1.

Sources:

- Modalities, training sizes and candidate counts come from VLM2Vec Table 1.
- Subset names were checked against the MMEB-eval configs on the Hugging Face Hub (2026-09-13).
- N24News and the four grounding subsets were checked against the eval parquet files.

Modalities: I = image, T = text, I+T = image and text.

## Classification (CLS)

Candidates are the class labels, so the candidate count is the number of classes.

| Split | Table column | MMEB-train subset | MMEB-eval subset | Query → Target | #Train | #Candidates |
| --- | --- | --- | --- | --- | --- | --- |
| IOD | IN-1K | `ImageNet_1K` | `ImageNet-1K` | I → T | 100K | 1000 |
| IOD | N24News | `N24News` | `N24News` | I+T → T | 49K | 24 |
| IOD | Hateful | `HatefulMemes` | `HatefulMemes` | I → T | 8K | 2 |
| IOD | VOC07 | `VOC2007` | `VOC2007` | I → T | 8K | 20 |
| IOD | SUN397 | `SUN397` | `SUN397` | I → T | 20K | 397 |
| OOD | Place365 | — | `Place365` | I → T | — | 365 |
| OOD | IN-A | — | `ImageNet-A` | I → T | — | 1000 |
| OOD | IN-R | — | `ImageNet-R` | I → T | — | 200 |
| OOD | ObjectNet | — | `ObjectNet` | I → T | — | 313 |
| OOD | Country211 | — | `Country211` | I → T | — | 211 |

IOD training total: about 185K.

- **ImageNet naming.** The train and eval names differ: `ImageNet_1K` in MMEB-train, `ImageNet-1K` in MMEB-eval.
- **N24News target.** VLM2Vec Table 1 (and HieRD Table 10, which copies it) lists the target as I. The eval data has 24 text domain labels with empty `tgt_img_path`, so the target is text.
- **Country211.** The paper writes Country-211; the subset name is `Country211`.

## Visual question answering (VQA)

Each query has 1 ground-truth answer and 999 distractors.

| Split | Table column | MMEB-train subset | MMEB-eval subset | Query → Target | #Train | #Candidates |
| --- | --- | --- | --- | --- | --- | --- |
| IOD | OK | `OK-VQA` | `OK-VQA` | I+T → T | 9K | 1000 |
| IOD | A-OK | `A-OKVQA` | `A-OKVQA` | I+T → T | 17K | 1000 |
| IOD | Doc | `DocVQA` | `DocVQA` | I+T → T | 40K | 1000 |
| IOD | I-VQA | `InfographicsVQA` | `InfographicsVQA` | I+T → T | 24K | 1000 |
| IOD | Chart | `ChartQA` | `ChartQA` | I+T → T | 28K | 1000 |
| IOD | Vis7W | `Visual7W` | `Visual7W` | I+T → T | 70K | 1000 |
| OOD | ScienceQA | — | `ScienceQA` | I+T → T | — | 1000 |
| OOD | VizWiz | — | `VizWiz` | I+T → T | — | 1000 |
| OOD | GQA | — | `GQA` | I+T → T | — | 1000 |
| OOD | TextVQA | — | `TextVQA` | I+T → T | — | 1000 |

IOD training total: about 188K.

## Retrieval (RET)

| Split | MMEB-train subset | MMEB-eval subset | Query → Target | #Train | #Candidates |
| --- | --- | --- | --- | --- | --- |
| IOD | `VisDial` | `VisDial` | T → I | 123K | 1000 |
| IOD | `CIRR` | `CIRR` | I+T → I | 26K | 1000 |
| IOD | `VisualNews_t2i` | `VisualNews_t2i` | T → I | 100K | 1000 |
| IOD | `VisualNews_i2t` | `VisualNews_i2t` | I → T | 100K | 1000 |
| IOD | `MSCOCO_t2i` | `MSCOCO_t2i` | T → I | 100K | 1000 |
| IOD | `MSCOCO_i2t` | `MSCOCO_i2t` | I → T | 113K | 1000 |
| IOD | `NIGHTS` | `NIGHTS` | I → I | 16K | 1000 |
| IOD | `WebQA` | `WebQA` | T → I+T | 17K | 1000 |
| OOD | — | `OVEN` | I+T → I+T | — | 1000 |
| OOD | — | `FashionIQ` | I+T → I | — | 1000 |
| OOD | — | `EDIS` | T → I+T | — | 1000 |
| OOD | — | `Wiki-SS-NQ` | T → I (document screenshots) | — | 1000 |

IOD training total: about 595K, roughly 3.2× CLS.

- **Names.** Train and eval subset names are identical.
- **Pipeline support.**
  - The IOD list equals the `ret` preset in `scripts/data/download_mmeb.py`.
  - RET is not yet wired into `src/evaluation/benchmarks.py`, which rejects subset names outside its groups.
  - It is also not in `scripts/data/precompute_teacher_embeddings.sh`, which accepts only `TASK=cls|vqa`.
- **OOD coverage.**
  - FashionIQ shares CIRR's composed-retrieval form; EDIS shares WebQA's T → I+T form.
  - OVEN (I+T → I+T) and Wiki-SS-NQ (document images) have no IOD counterpart.
- **Subsampling.** NIGHTS and WebQA are small. If the training set is subsampled, prefer a per-subset cap over a uniform `--percent_data`, and apply the same cap to every method and to the teacher cache.

## Visual grounding (GD)

The query is a full image plus an instruction naming a region. The candidates are image crops.

| Split | MMEB-train subset | MMEB-eval subset | Query → Target | #Train | #Candidates |
| --- | --- | --- | --- | --- | --- |
| IOD | `MSCOCO` | `MSCOCO` | I+T → I | 100K | 1000 |
| OOD | — | `Visual7W-Pointing` | I+T → I | — | 1000 |
| OOD | — | `RefCOCO` | I+T → I | — | 1000 |
| OOD | — | `RefCOCO-Matching` | I+T → I+T | — | 1000 |

IOD training total: about 100K, from a single task.

- **What each eval subset asks** (from the eval data):
  - `MSCOCO`: "Crop the image to isolate the object labeled as …". Candidates are object crops from different images.
  - `Visual7W-Pointing`: "Select the portion of the image that answers the question …". Candidates are crops.
  - `RefCOCO`: select the region matching a referring expression. Candidates are crops from different images.
  - `RefCOCO-Matching`: each candidate is an image paired with a referring expression. Some negatives come from **the same image** with a different expression, so it is the hardest of the four.
- **Name clash.** The grounding subset `MSCOCO` is a different dataset from the retrieval subsets `MSCOCO_i2t` and `MSCOCO_t2i`, although all three share the MSCOCO images.
- **Pipeline support.**
  - The `grounding` preset in `scripts/data/download_mmeb.py` downloads `MSCOCO`.
  - Grounding is not yet wired into `src/evaluation/benchmarks.py` or `scripts/data/precompute_teacher_embeddings.sh`.
- **Sampler.** With a single training task, the task-homogeneous sampler has no effect.
- **Published reference** (HieRD Table 14; FastVLM-0.5B student only, single run, batch 16):
  - The B3-Qwen2-2B teacher scores below SFT on `MSCOCO` (70.5 vs 71.9) and on `Visual7W-Pointing` (74.7 vs 82.1).
  - All distillation methods land within about 1 point of SFT on average.

## Where these names live in code

| What | File |
| --- | --- |
| Training subsets per launcher | `scripts/train/<method>/<student>_<task>.sh`, `--subset_name` |
| Download presets (`cls`, `vqa`, `ret`, `grounding`) | `scripts/data/download_mmeb.py` |
| Eval groups and paper column labels | `src/evaluation/benchmarks.py` (`CLS_IND`, `VQA_IND`, `CLS_OOD`, `VQA_OOD`) |
| Teacher cache subset lists | `scripts/data/precompute_teacher_embeddings.sh` |
