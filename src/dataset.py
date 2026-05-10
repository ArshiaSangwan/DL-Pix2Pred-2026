import os
import random
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoProcessor
from data_utils import (
    build_assistant_text,
    build_chat_messages,
    build_chat_messages_with_assistant,
    parse_choices,
)

class SciQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        processor: AutoProcessor,
        max_length=512,
        is_test=False,
        objective="letter_ce",
        target_format="letter",
        include_metadata=True,
        metadata_fields=None,
        lecture_max_chars=None,
        prompt_variant="default",
        include_caption=False,
        caption_max_chars=None,
        image_augmentation=None,
        split="train",
    ):
        if isinstance(csv_path, (list, tuple)):
            self.df = pd.concat([pd.read_csv(path) for path in csv_path], ignore_index=True)
        else:
            self.df = pd.read_csv(csv_path)
        self.df = self.df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.is_test = is_test
        self.objective = objective
        self.target_format = target_format
        self.include_metadata = include_metadata
        self.metadata_fields = metadata_fields
        self.lecture_max_chars = lecture_max_chars
        self.prompt_variant = prompt_variant
        self.include_caption = include_caption
        self.caption_max_chars = caption_max_chars
        self.image_augmentation = image_augmentation or {}
        self.split = split

    def _select_prompt_variant(self):
        if isinstance(self.prompt_variant, (list, tuple)):
            variants = [str(variant).strip() for variant in self.prompt_variant if str(variant).strip()]
            if not variants:
                return "default"
            return random.choice(variants)
        return str(self.prompt_variant)

    def _augment_image(self, image):
        if self.is_test or self.split != "train" or not self.image_augmentation:
            return image

        augmented = image
        rotate_degrees = float(self.image_augmentation.get("rotate_degrees", 0.0) or 0.0)
        if rotate_degrees > 0:
            angle = random.uniform(-rotate_degrees, rotate_degrees)
            augmented = augmented.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(255, 255, 255),
            )

        crop_min_scale = float(self.image_augmentation.get("crop_scale_min", 1.0) or 1.0)
        crop_max_scale = float(self.image_augmentation.get("crop_scale_max", 1.0) or 1.0)
        crop_min_scale = max(0.5, min(crop_min_scale, 1.0))
        crop_max_scale = max(crop_min_scale, min(crop_max_scale, 1.0))
        if crop_max_scale < 1.0:
            width, height = augmented.size
            scale = random.uniform(crop_min_scale, crop_max_scale)
            crop_width = max(1, int(width * scale))
            crop_height = max(1, int(height * scale))
            left = 0 if crop_width == width else random.randint(0, width - crop_width)
            top = 0 if crop_height == height else random.randint(0, height - crop_height)
            augmented = augmented.crop((left, top, left + crop_width, top + crop_height))
            augmented = augmented.resize((width, height), resample=Image.Resampling.BICUBIC)

        brightness = float(self.image_augmentation.get("brightness", 0.0) or 0.0)
        if brightness > 0:
            factor = random.uniform(max(0.1, 1.0 - brightness), 1.0 + brightness)
            augmented = ImageEnhance.Brightness(augmented).enhance(factor)

        contrast = float(self.image_augmentation.get("contrast", 0.0) or 0.0)
        if contrast > 0:
            factor = random.uniform(max(0.1, 1.0 - contrast), 1.0 + contrast)
            augmented = ImageEnhance.Contrast(augmented).enhance(factor)

        saturation = float(self.image_augmentation.get("saturation", 0.0) or 0.0)
        if saturation > 0:
            factor = random.uniform(max(0.1, 1.0 - saturation), 1.0 + saturation)
            augmented = ImageEnhance.Color(augmented).enhance(factor)

        horizontal_flip_prob = float(self.image_augmentation.get("horizontal_flip_prob", 0.0) or 0.0)
        if horizontal_flip_prob > 0 and random.random() < horizontal_flip_prob:
            augmented = ImageOps.mirror(augmented)

        return augmented

    def __len__(self):
        return len(self.df)

    def resolve_image_path(self, row):
        row_image_path = str(row["image_path"])
        basename = os.path.basename(row_image_path)
        split_hint = basename.split("_", 1)[0] if "_" in basename else None
        candidates = []
        if self.image_dir:
            candidates.append(os.path.join(self.image_dir, basename))
        candidates.append(row_image_path)
        candidates.append(os.path.join(os.getcwd(), row_image_path))
        if row_image_path.startswith("images/"):
            candidates.append(os.path.join("data", row_image_path))
            candidates.append(os.path.join(os.getcwd(), "data", row_image_path))
        if split_hint in {"train", "val", "test"}:
            candidates.append(os.path.join("data", "images", split_hint, basename))
            candidates.append(os.path.join(os.getcwd(), "data", "images", split_hint, basename))

        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not resolve image for {row.get('id', '<unknown>')}: {candidates}")

    def _processor_call(self, text, image):
        kwargs = {
            "text": text,
            "images": image,
            "return_tensors": "pt",
            "truncation": False,
        }
        if self.max_length:
            kwargs["max_length"] = self.max_length
        return self.processor(**kwargs)

    def _unbatch_inputs(self, inputs):
        inputs.pop("token_type_ids", None)
        item = {}
        for key, value in inputs.items():
            if key in {"pixel_values", "pixel_attention_mask"} and value.dim() >= 4:
                item[key] = value.squeeze(0)
            else:
                item[key] = value.squeeze(0)
        return item

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.resolve_image_path(row)
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            image = self._augment_image(image)

        prompt_variant = self._select_prompt_variant()

        prompt_messages = build_chat_messages(
            row,
            include_metadata=self.include_metadata,
            metadata_fields=self.metadata_fields,
            lecture_max_chars=self.lecture_max_chars,
            prompt_variant=prompt_variant,
            include_caption=self.include_caption,
            caption_max_chars=self.caption_max_chars,
        )
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if self.objective == "sft" and not self.is_test:
            assistant_text = build_assistant_text(row, target_format=self.target_format)
            full_prompt = self.processor.apply_chat_template(
                build_chat_messages_with_assistant(
                    row,
                    assistant_text,
                    include_metadata=self.include_metadata,
                    metadata_fields=self.metadata_fields,
                    lecture_max_chars=self.lecture_max_chars,
                    prompt_variant=prompt_variant,
                    include_caption=self.include_caption,
                    caption_max_chars=self.caption_max_chars,
                ),
                add_generation_prompt=False,
                tokenize=False,
            )
            inputs = self._processor_call(full_prompt, image)
            prompt_inputs = self._processor_call(prompt, image)
            item = self._unbatch_inputs(inputs)
            labels = item["input_ids"].clone()
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[:prompt_len] = -100
            if "attention_mask" in item:
                labels[item["attention_mask"] == 0] = -100
            item["labels"] = labels
        else:
            inputs = self._processor_call(prompt, image)
            item = self._unbatch_inputs(inputs)

        item['id'] = row['id']
        choices_list = parse_choices(row["choices"])
        item['num_choices'] = len(choices_list)
        if not self.is_test:
            item['label'] = int(row['answer'])
        return item
