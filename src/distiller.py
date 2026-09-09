import json
import math
import os

import torch
from datasets import concatenate_datasets, load_dataset
from torch import nn
from torch.utils.data import (
    Dataset,
)
from transformers import (
    AutoTokenizer,
    ProcessorMixin,
)
from transformers.training_args import TrainingArguments

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.data.images import decode_image, resize_to_budget, resolve_target
from src.model.model import MMEBModel
from src.model.processor import (
    PHI3V,
    VLM_IMAGE_TOKENS,
    load_processor,
    process_vlm_inputs_fns,
)
from src.utils import print_master, print_rank

POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
    "ImageNet_1K": POS_MOD_CLASS_LABEL,
    "HatefulMemes": POS_MOD_CLASS_LABEL,
    "SUN397": POS_MOD_CLASS_LABEL,
    "N24News": POS_MOD_CLASS_LABEL,
    "VOC2007": POS_MOD_CLASS_LABEL,
    "Place365": POS_MOD_CLASS_LABEL,
    "ImageNet-A": POS_MOD_CLASS_LABEL,
    "ImageNet-R": POS_MOD_CLASS_LABEL,
    "ObjectNet": POS_MOD_CLASS_LABEL,
    "Country211": POS_MOD_CLASS_LABEL,
    "OK-VQA": POS_MOD_ANSWER,
    "A-OKVQA": POS_MOD_ANSWER,
    "DocVQA": POS_MOD_ANSWER,
    "InfographicsVQA": POS_MOD_ANSWER,
    "ChartQA": POS_MOD_ANSWER,
    "Visual7W": POS_MOD_ANSWER,
    "ScienceQA": POS_MOD_ANSWER,
    "GQA": POS_MOD_ANSWER,
    "TextVQA": POS_MOD_ANSWER,
    "VizWiz": POS_MOD_ANSWER,
    "MSCOCO_i2t": POS_MOD_IMAGE_CAPTION,
    "VisualNews_i2t": POS_MOD_IMAGE_CAPTION,
}


def process_image(image, resolution, max_dim=1344, keep_aspect=False):
    """Shrink one image to the ``--image_resolution`` budget.

    Thin wrapper kept for callers outside the dataset; the logic lives in
    `src.data.images` so the decode path can share it.
    """
    if image is None:
        return None
    return resize_to_budget(
        image, resolve_target(resolution, max_dim), keep_aspect=keep_aspect
    )


def create_semi_orthogonal_matrix(tensor):
    rows, cols = tensor.shape
    if rows >= cols:
        # QR trực tiếp
        a = torch.randn(rows, cols, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode="reduced")
        tensor.data[:] = q[:, :cols]
    else:
        # QR trên ma trận transpose để đảm bảo W W^T = I
        a = torch.randn(cols, rows, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode="reduced")
        tensor.data[:] = q.T[:rows, :]
    return tensor


class Distiller(nn.Module):
    def __init__(self, model_args, training_args, data_args=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        # Optional: only used to check that a teacher embedding cache was built
        # from the same dataset this run is about to train on.
        self._data_args = data_args
        self.student = self._load_student()
        self.teacher_cache = self._load_teacher_cache()
        # Nothing reads the teacher's weights when its embeddings are cached, so
        # they are not loaded at all -- that is several GB of device memory the
        # batch can use instead.
        self.teacher = None if self.teacher_cache is not None else self._load_teacher()
        self._configure_attention()
        self.student_hidden_dim = self.model_args.student_hidden_dim
        self.teacher_hidden_dim = self.model_args.teacher_hidden_dim
        self.temperature = model_args.temperature
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.teacher_model_name
        )
        from src.criterions import needs_tokenizer

        self._criterion_needs_tokenizer = needs_tokenizer(
            getattr(training_args, "kd_loss_type", None)
        )
        # if self.model_args.projector_config_path is not None:
        self.set_projector()
        print("Projectors set.")

    def _create_model_args(self, model_type="teacher"):
        if model_type == "teacher":
            model_args = ModelArguments(
                model_name=self.model_args.teacher_model_name,
                checkpoint_path=getattr(
                    self.model_args, "teacher_checkpoint_path", None
                ),
                lora=self.model_args.teacher_lora,
                lora_r=self.model_args.teacher_lora_r,
                lora_alpha=self.model_args.teacher_lora_alpha,
                lora_dropout=self.model_args.teacher_lora_dropout,
                lora_target_modules=self.model_args.teacher_lora_target_modules,
                pooling=self.model_args.teacher_pooling,
                normalize=self.model_args.teacher_normalize,
                model_backbone=self.model_args.teacher_backbone,
            )
        else:
            print_rank("Not implemented student model args creation.")
            raise NotImplementedError
        return model_args

    def _load_student(self):
        print("Load student with lora rank:", self.model_args.lora_r)
        print("Student use lora:", self.model_args.lora)
        student = MMEBModel.build(self.model_args)
        print("Student model built.")
        return student

    def _load_teacher(self):
        model_args = self._create_model_args("teacher")
        print("Load teacher with lora rank:", model_args.lora_r)
        print("Teacher use lora:", model_args.lora)
        teacher = MMEBModel.load(model_args, is_trainable=False)
        for param in teacher.parameters():
            param.requires_grad = False
        print("Teacher model loaded.")
        return teacher

    def _load_teacher_cache(self):
        """Open the precomputed teacher embeddings, if this run asked for them."""
        path = getattr(self.model_args, "teacher_embedding_cache", None)
        if not path:
            return None

        from src.criterions import supports_teacher_cache
        from src.teacher_cache import TeacherEmbeddingCache, build_fingerprint

        kd_loss_type = getattr(self.training_args, "kd_loss_type", None)
        if not supports_teacher_cache(kd_loss_type):
            raise ValueError(
                f"--teacher_embedding_cache cannot be used with "
                f"--kd_loss_type {kd_loss_type!r}: that criterion reads the "
                f"teacher's hidden states or attentions, which the cache does "
                f"not store. Drop the flag, or use a criterion that only reads "
                f"the final embedding."
            )
        # Without data_args this checks the teacher half of the fingerprint;
        # pass data_args to Distiller() to have the dataset identity checked too.
        cache = TeacherEmbeddingCache.load(
            path, fingerprint=build_fingerprint(self.model_args, self._data_args)
        )
        # The t2s projector is built from --teacher_hidden_dim, not from the
        # cache, so a disagreement would surface as a shape error inside a
        # matmul several minutes into training.
        declared_dim = getattr(self.model_args, "teacher_hidden_dim", None)
        if declared_dim is not None and int(declared_dim) != cache.dim:
            raise ValueError(
                f"the teacher embedding cache at {path!r} holds {cache.dim}-d "
                f"embeddings but --teacher_hidden_dim is {declared_dim}. Set "
                f"--teacher_hidden_dim {cache.dim}, or rebuild the cache."
            )
        print_master(
            f"Using cached teacher embeddings from {path} "
            f"({cache.num_samples} samples, dim {cache.dim}); the teacher model "
            f"will not be loaded."
        )
        return cache

    def encode_teacher(self, input_data, side, dtype=None):
        """Pooled teacher embeddings for one side of the batch.

        The single place a criterion should get teacher embeddings from: it
        serves them out of the cache when there is one and runs the frozen model
        otherwise, so a criterion does not have to know which it is.
        """
        if self.teacher_cache is not None:
            qry, pos = self.teacher_cache.get(
                input_data["sample_ids"],
                device=next(self.student.parameters()).device,
                dtype=dtype,
            )
            return qry if side == "qry" else pos

        with torch.no_grad():
            self.teacher.eval()
            output = self.teacher.encode_input(input_data["teacher_inputs"][side])
        return output[0] if isinstance(output, (tuple, list)) else output

    def _configure_attention(self):
        """Only ask for attention matrices on the side the criterion reads.

        `output_attentions=True` pins the backbone to the eager attention kernel
        and keeps a (B, heads, L, L) tensor per layer alive. For the student that
        also inflates the backward pass, so the saving is largest there. Any side
        whose attentions are unused switches to SDPA.
        """
        from src.criterions import attention_needs

        kd_loss_type = getattr(self.training_args, "kd_loss_type", None)
        student_needs, teacher_needs = attention_needs(kd_loss_type)
        print_master(
            f"Criterion '{kd_loss_type}' attention needs: "
            f"student={student_needs}, teacher={teacher_needs}"
        )

        self.student.output_attentions = student_needs
        self.student.set_attn_implementation("eager" if student_needs else "sdpa")
        if self.teacher is not None:
            self.teacher.output_attentions = teacher_needs
            self.teacher.set_attn_implementation("eager" if teacher_needs else "sdpa")

    def get_student_processor(self):
        processor = load_processor(self.model_args, None)
        print("Student processor loaded.")
        return processor

    def get_teacher_processor(self):
        if self.teacher_cache is not None:
            print_master("Teacher embeddings are cached; no teacher processor needed.")
            return None
        model_args = self._create_model_args("teacher")
        processor = load_processor(model_args, None)
        print("Teacher processor loaded.")
        return processor

    def forward(self, criterion, batch):
        # `_criterion_needs_tokenizer` is resolved once in __init__ from the
        # criterion registry rather than from a list of names repeated here --
        # a new method that needs a tokenizer and is missing from a list like
        # that fails with a TypeError several minutes into a run.
        if self._criterion_needs_tokenizer:
            return criterion(self, batch, tokenizer=self.tokenizer)
        return criterion(self, batch)

    def set_projector(self):
        """
        Create a list of linear projectors mapping
        student_hidden_dim -> teacher_hidden_dim
        One projector per teacher layer mapping.
        """
        projector_list = nn.ModuleList()

        if self.model_args.projector_config_path is not None:
            self.projectors = nn.ModuleDict()
            projector_config = json.load(
                open(self.model_args.projector_config_path, "r")
            )

            name_dict = {
                "s": self.student_hidden_dim,
                "t": self.teacher_hidden_dim,
                "relu": nn.ReLU(),
            }

            for name, cfg in projector_config.items():
                if not cfg.get("enabled", False):
                    continue
                seq = nn.Sequential()
                parts = cfg["structure"].split("-")
                parsed = []

                for p in parts:
                    if p == "relu":
                        parsed.append("relu")
                    else:
                        coef = int(p[:-1]) if len(p) > 1 and p[:-1].isdigit() else 1
                        parsed.append(coef * name_dict[p[-1]])
                for i in range(len(parsed) - 1):
                    a, b = parsed[i], parsed[i + 1]
                    if isinstance(a, int) and isinstance(b, int):
                        layer = nn.Linear(a, b)
                        create_semi_orthogonal_matrix(layer.weight)
                        layer = layer.to(dtype=torch.bfloat16)
                        seq.append(layer)
                    elif b == "relu":
                        seq.append(name_dict[b])
                    elif a == "relu" and isinstance(b, int):
                        prev_out = (
                            parsed[i - 1] if isinstance(parsed[i - 1], int) else None
                        )
                        layer = nn.Linear(prev_out, b)
                        create_semi_orthogonal_matrix(layer.weight)
                        layer = layer.to(dtype=torch.bfloat16)
                        seq.append(layer)
                self.projectors[name] = seq
        else:
            for _ in range(len(self.training_args.teacher_layer_mapping)):
                projector = nn.Linear(
                    self.student_hidden_dim,
                    self.teacher_hidden_dim,
                    dtype=torch.bfloat16,
                )
                projector_list.append(projector)

            self.projectors = projector_list
        print(f"Created {len(self.projectors)} linear projectors.")

    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "projectors") and self.projectors is not None:
            lr = (
                getattr(self.training_args, "projector_lr", None)
                or self.training_args.learning_rate
            )
            optimizer.add_param_group(
                {"params": self.projectors.parameters(), "lr": lr}
            )
        print("Projector parameters added to optimizer.")
        return optimizer


class DistillationCollator:
    """Turn dataset rows into processed student and teacher model inputs.

    `include_student` / `include_teacher` skip one side entirely. Training with a
    teacher embedding cache does not need the teacher's tokenizer or image
    processor at all, and the precompute pass does not need the student's, so in
    both cases the skipped side costs nothing in the dataloader workers.
    """

    def __init__(
        self,
        student_processor: ProcessorMixin,
        teacher_processor: ProcessorMixin,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        batch_size: int | None = None,
        include_student: bool = True,
        include_teacher: bool | None = None,
    ):
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.batch_size = batch_size
        self.include_student = include_student
        # Same default as DistillationDataset: a run with a teacher embedding
        # cache produces no teacher rows to process, and has no teacher
        # processor to process them with.
        if include_teacher is None:
            include_teacher = not getattr(model_args, "teacher_embedding_cache", None)
        self.include_teacher = include_teacher

    # Stands in for a row that has no dataset index: an example that arrived
    # empty, or one whose every text/image pair was rejected. `_get_batch_inputs`
    # substitutes a blank row for those, so the id list has to substitute
    # something too or the two go out of step and a cache lookup silently reads
    # the wrong sample. -1 is out of range for any cache, so it fails loudly.
    PLACEHOLDER_SAMPLE_ID = -1

    def _get_sample_ids(self, batch):
        """One dataset index per row `_get_batch_inputs` will emit.

        Kept deliberately in lockstep with that method's placeholder handling:
        these ids are the key a teacher embedding cache is read by, so an
        off-by-one here would serve one sample's embedding for another.
        """
        key = "student_query_text" if self.include_student else "teacher_query_text"
        ids = []
        for example in batch:
            if example is None or not example:
                ids.append(self.PLACEHOLDER_SAMPLE_ID)
                continue
            texts = example[key]
            if not isinstance(texts, list):
                texts = [texts]
            row_ids = example.get("sample_ids", [])
            if not texts:
                ids.append(self.PLACEHOLDER_SAMPLE_ID)
            elif len(row_ids) == len(texts):
                ids.extend(row_ids)
            else:
                ids.extend([self.PLACEHOLDER_SAMPLE_ID] * len(texts))
        return torch.tensor(ids, dtype=torch.long)

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        # print("Processing batch for keys:", text_keyname, image_keyname)
        texts, visual_inputs = [], []
        for example in batch:
            if example is None or not example:
                text, visual_input = " ", None
                texts.append(text)
                visual_inputs.append(visual_input)
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                if not isinstance(text, list):
                    text = [text]
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                if not text and not raw_images:
                    text, visual_input = " ", None
                    texts.append(text)
                    visual_inputs.append(visual_input)
                else:
                    for t, img in zip(text, raw_images):
                        if not t and img is None:
                            t, img = " ", None
                        texts.append(t)
                        visual_inputs.append(img)
        inputs = {"text": texts, "images": visual_inputs}
        return inputs

    def __call__(self, examples):
        batch = {"sample_ids": self._get_sample_ids(examples)}
        bs = batch["sample_ids"].numel()

        if self.include_student:
            student_qry_inputs = self._get_batch_inputs(
                examples, "student_query_text", "student_query_image"
            )
            student_pos_inputs = self._get_batch_inputs(
                examples, "student_pos_text", "student_pos_image"
            )
            bs = len(student_qry_inputs["text"])
            if batch["sample_ids"].numel() != bs:
                # Should be unreachable: _get_sample_ids mirrors
                # _get_batch_inputs row for row. Kept because the consequence of
                # the two drifting apart is silently mismatched teacher
                # embeddings rather than a crash.
                raise RuntimeError(
                    f"{batch['sample_ids'].numel()} sample ids for {bs} student "
                    f"rows; _get_sample_ids and _get_batch_inputs disagree"
                )

        assert bs > 0, "An empty batch is detected!"

        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expected batch size {self.batch_size}, but got {bs}.")

        if self.include_student:
            process_student_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
            batch["student_inputs"] = {
                "qry": process_student_fn(
                    student_qry_inputs,
                    processor=self.student_processor,
                    max_length=self.data_args.max_len,
                ),
                "pos": process_student_fn(
                    student_pos_inputs,
                    processor=self.student_processor,
                    max_length=self.data_args.max_len,
                ),
            }

        if self.include_teacher:
            teacher_qry_inputs = self._get_batch_inputs(
                examples, "teacher_query_text", "teacher_query_image"
            )
            teacher_pos_inputs = self._get_batch_inputs(
                examples, "teacher_pos_text", "teacher_pos_image"
            )
            process_teacher_fn = process_vlm_inputs_fns[
                self.model_args.teacher_backbone
            ]
            batch["teacher_inputs"] = {
                "qry": process_teacher_fn(
                    teacher_qry_inputs,
                    processor=self.teacher_processor,
                    max_length=self.data_args.max_len,
                ),
                "pos": process_teacher_fn(
                    teacher_pos_inputs,
                    processor=self.teacher_processor,
                    max_length=self.data_args.max_len,
                ),
            }

        return batch


class DistillationDataset(Dataset):
    def __init__(
        self, data_args, model_args, include_student=True, include_teacher=None
    ):
        self.data_args = data_args
        self.model_args = model_args
        self.include_student = include_student
        # A run with a teacher embedding cache never touches the teacher, so it
        # should not pay for decoding and resizing images for its backbone
        # either. The precompute pass overrides this the other way round.
        if include_teacher is None:
            include_teacher = not getattr(model_args, "teacher_embedding_cache", None)
        self.include_teacher = include_teacher

        # How small the JPEG decoder is allowed to go. Both backbones resize to
        # the same budget, so one hint covers them -- but PHI3V is exempt from
        # the resize entirely, so if either side is PHI3V the image has to be
        # decoded at full resolution and the hint is dropped.
        self._keep_aspect = bool(getattr(data_args, "image_keep_aspect_ratio", False))
        self._resize_target = (
            resolve_target(data_args.image_resolution)
            if data_args.image_resolution
            else None
        )
        backbones = []
        if include_student:
            backbones.append(model_args.model_backbone)
        if include_teacher:
            backbones.append(model_args.teacher_backbone)
        self._decode_target = self._resize_target if PHI3V not in backbones else None
        print_rank(
            f"Image pipeline: resize_target={self._resize_target}, "
            f"decode_hint={self._decode_target}, keep_aspect={self._keep_aspect}"
        )

        train_data = []

        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=f"{self.data_args.dataset_split}",
            )
            if subset == "WebQA" and "qry" in subset_data.column_names:
                subset_data = subset_data.map(
                    lambda x: {"qry": x["qry"].replace("<|image_1|>", "").strip()}
                )
                print_rank("Preprocessed WebQA to remove <image_1> tokens in queries.")
            total_samples = len(subset_data)
            num_samples_to_keep = math.ceil(total_samples * self.data_args.percent_data)
            subset_data = subset_data.select(range(num_samples_to_keep))
            subset_data = subset_data.add_column(
                "pos_text_instruction",
                [
                    POS_MOD_DICT.get(subset, "") + text
                    for text in subset_data["pos_text"]
                ],
            )
            subset_data = subset_data.remove_columns(
                set(["neg_text", "neg_image_path"]) & set(subset_data.column_names)
            )
            subset_data = subset_data.remove_columns(
                set(subset_data.column_names)
                - set(
                    ["qry", "qry_image_path", "pos_image_path", "pos_text_instruction"]
                )
            )
            subset_data = subset_data.rename_column("pos_text_instruction", "pos_text")
            train_data.append(subset_data)

        self.train_data = concatenate_datasets(train_data)
        print_rank(
            f"Loaded {len(self.train_data)} samples from {self.data_args.dataset_name} with subsets {self.data_args.subset_name}"
        )

    def __len__(self):
        return len(self.train_data)

    def _decode_image(self, img_path):
        """Decode one image file to RGB, padded up to the minimum size.

        Split out from _get_image so the student and the teacher can share a
        single decode of the same file instead of hitting the disk and the JPEG
        decoder twice per sample.

        The decode is told what resolution the image is headed for, so libjpeg
        can emit a DCT-scaled image instead of a full-size one that is then
        thrown away (see `src.data.images.decode_image`). Only when *every*
        backbone in this run resizes, though: PHI3V keeps the original.
        """
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        return decode_image(full_img_path, target_max=self._decode_target)

    def _resize_for_backbone(self, image, backbone):
        if image is None:
            return None
        if backbone != PHI3V and self.data_args.image_resolution:
            return resize_to_budget(
                image, self._resize_target, keep_aspect=self._keep_aspect
            )
        return image

    def _get_image(self, img_path, backbone):
        return self._resize_for_backbone(self._decode_image(img_path), backbone)

    def _prepare_side(self, qry_text, pos_text, raw_qry_image, raw_pos_image, backbone):
        """Text/image pair rewritten for one backbone, or None if it is empty."""
        if backbone != PHI3V:
            qry_text = qry_text.replace(
                VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[backbone]
            )
            pos_text = pos_text.replace(
                VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[backbone]
            )
        qry_image = self._resize_for_backbone(raw_qry_image, backbone)
        pos_image = self._resize_for_backbone(raw_pos_image, backbone)
        if (not qry_text and qry_image is None) or (not pos_text and pos_image is None):
            return None
        return qry_text, pos_text, qry_image, pos_image

    def __getitem__(self, data_idx):
        # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>get image called, {data_idx}", flush=True)

        # One row fetch, not four: indexing an Arrow-backed dataset decodes the
        # whole row each time it is subscripted.
        row = self.train_data[data_idx]
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            row["qry"],
            row["qry_image_path"],
            row["pos_text"],
            row["pos_image_path"],
        )

        if not isinstance(qry_texts, list):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]

        student_qry_texts, student_qry_images, student_pos_texts, student_pos_images = (
            [],
            [],
            [],
            [],
        )
        teacher_qry_texts, teacher_qry_images, teacher_pos_texts, teacher_pos_images = (
            [],
            [],
            [],
            [],
        )

        student_backbone = self.model_args.model_backbone
        teacher_backbone = self.model_args.teacher_backbone

        n_emitted = 0
        for qry_text, qry_image_path, pos_text, pos_image_path in zip(
            qry_texts, qry_image_paths, pos_texts, pos_image_paths
        ):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2, qwenvl

            # Decode each file once and reuse it for both backbones; the student
            # and the teacher used to open and JPEG-decode the same two files
            # independently.
            raw_qry_image = self._decode_image(qry_image_path)
            raw_pos_image = self._decode_image(pos_image_path)

            # Both sides are prepared before either is committed. Appending the
            # student first and only then testing the teacher would leave the two
            # lists at different lengths whenever the teacher rejected a pair,
            # silently pairing student row i with teacher row i+1 downstream.
            stu = (
                self._prepare_side(
                    qry_text, pos_text, raw_qry_image, raw_pos_image, student_backbone
                )
                if self.include_student
                else ()
            )
            tea = (
                self._prepare_side(
                    qry_text, pos_text, raw_qry_image, raw_pos_image, teacher_backbone
                )
                if self.include_teacher
                else ()
            )
            if (self.include_student and stu is None) or (
                self.include_teacher and tea is None
            ):
                print("empty inputs")
                continue

            if self.include_student:
                stu_qry_text, stu_pos_text, stu_qry_image, stu_pos_image = stu
                student_qry_texts.append(stu_qry_text)
                student_qry_images.append(stu_qry_image)
                student_pos_texts.append(stu_pos_text)
                student_pos_images.append(stu_pos_image)

            if self.include_teacher:
                (
                    teacher_qry_text,
                    teacher_pos_text,
                    teacher_qry_image,
                    teacher_pos_image,
                ) = tea
                teacher_qry_texts.append(teacher_qry_text)
                teacher_qry_images.append(teacher_qry_image)
                teacher_pos_texts.append(teacher_pos_text)
                teacher_pos_images.append(teacher_pos_image)

            n_emitted += 1

        return {
            # One dataset index per emitted row, so a teacher embedding cache can
            # be keyed by it. Kept even when no cache is in use: it is 8 bytes a
            # row and it makes the batch self-describing.
            "sample_ids": [data_idx] * n_emitted,
            "student_query_text": student_qry_texts,
            "student_query_image": student_qry_images,
            "student_pos_text": student_pos_texts,
            "student_pos_image": student_pos_images,
            "teacher_query_text": teacher_qry_texts,
            "teacher_query_image": teacher_qry_images,
            "teacher_pos_text": teacher_pos_texts,
            "teacher_pos_image": teacher_pos_images,
        }
