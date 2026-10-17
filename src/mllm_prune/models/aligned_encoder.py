"""Frozen CLIP adapter exposing pooled text and projected dense image features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel


@dataclass(frozen=True)
class DenseImageFeatures:
    """Projected dense features over the full square-resized source image."""

    features: torch.Tensor
    original_size: tuple[int, int]


class CLIPAlignedEncoder:
    """CLIP scorer with deterministic full-image resize and dense patch projection."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        text_batch_size: int = 64,
        image_batch_size: int = 8,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.text_batch_size = text_batch_size
        self.image_batch_size = image_batch_size
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False)
        self.model = CLIPModel.from_pretrained(self.model_path, dtype=dtype).to(device).eval()
        vision_config = self.model.config.vision_config
        self.image_size = int(vision_config.image_size)
        self.patch_size = int(vision_config.patch_size)
        if self.image_size % self.patch_size:
            raise ValueError("CLIP image size must be divisible by patch size")

    @property
    def identifier(self) -> str:
        return str(self.model_path.resolve())

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.model.config.projection_dim), dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        tokenizer = self.processor.tokenizer
        max_length = int(tokenizer.model_max_length)
        for start in range(0, len(texts), self.text_batch_size):
            batch = list(texts[start : start + self.text_batch_size])
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            features = self.model.get_text_features(**inputs)
            outputs.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    def _prepare_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        # Direct square resize sees the full document/image and gives every Qwen
        # token a defined source coordinate. The choice is part of the config.
        resized = [image.convert("RGB").resize((self.image_size, self.image_size)) for image in images]
        return self.processor.image_processor(
            images=resized,
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt",
        )["pixel_values"]

    def encode_image_patches(
        self, images: Sequence[Image.Image]
    ) -> tuple[list[DenseImageFeatures], float]:
        if not images:
            return [], 0.0
        results: list[DenseImageFeatures] = []
        started = time.perf_counter()
        grid = self.image_size // self.patch_size
        for start in range(0, len(images), self.image_batch_size):
            batch_images = images[start : start + self.image_batch_size]
            pixels = self._prepare_images(batch_images).to(self.device, dtype=self.dtype)
            vision = self.model.vision_model(pixel_values=pixels)
            patches = vision.last_hidden_state[:, 1:, :]
            patches = self.model.vision_model.post_layernorm(patches)
            patches = self.model.visual_projection(patches)
            patches = F.normalize(patches.float(), dim=-1).reshape(-1, grid, grid, patches.shape[-1])
            for image, feature in zip(batch_images, patches, strict=True):
                results.append(DenseImageFeatures(feature.cpu(), image.size))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return results, (time.perf_counter() - started) * 1000
