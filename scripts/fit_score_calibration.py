"""Extract aligned raw scores and fit experiment-003 empirical CDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from PIL import Image
import torch
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from mllm_prune.algorithms.aligned_scoring import compute_aligned_raw_scores
from mllm_prune.algorithms.calibration import (
    EmpiricalCDFCalibrator,
    WeightedScores,
    manifest_fingerprint,
)
from mllm_prune.config import path_from_env, value_from_env
from mllm_prune.data.mmcoqa import build_prompt_spec, load_subset, resolve_sample
from mllm_prune.models.aligned_encoder import CLIPAlignedEncoder
from mllm_prune.models.qwen25vl import encode_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subset = path_from_env("EXPERIMENT_003_CALIBRATION_SUBSET_PATH")
    image_dir = path_from_env("MMCOQA_IMAGE_DIR")
    qwen_path = path_from_env("QWEN_MODEL_PATH")
    aligned_path = path_from_env("ALIGNED_ENCODER_PATH")
    output = path_from_env("EXPERIMENT_003_CALIBRATION_PATH")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/003_aligned_encoder_scoring.yaml"))
    parser.add_argument("--subset", type=Path, default=subset, required=subset is None)
    parser.add_argument("--image-dir", type=Path, default=image_dir, required=image_dir is None)
    parser.add_argument("--qwen-path", type=Path, default=qwen_path, required=qwen_path is None)
    parser.add_argument("--aligned-path", type=Path, default=aligned_path, required=aligned_path is None)
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attn-implementation", default=value_from_env("MLLM_ATTN_IMPLEMENTATION", "sdpa"), choices=("flash_attention_2", "sdpa", "eager"))
    return parser.parse_args()


def _load_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing calibration on CPU")
    subset, samples = load_subset(args.subset, args.limit)
    qwen_processor = AutoProcessor.from_pretrained(
        args.qwen_path, min_pixels=256 * 28 * 28, max_pixels=int(config["max_pixels"])
    )
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_path,
        dtype=torch.bfloat16,
        device_map={"": device.index or 0},
        attn_implementation=args.attn_implementation,
    ).eval()
    aligned_cfg = config["aligned_encoder"]
    aligned = CLIPAlignedEncoder(
        args.aligned_path,
        device=device,
        dtype=torch.bfloat16,
        text_batch_size=int(aligned_cfg["text_batch_size"]),
        image_batch_size=int(aligned_cfg["image_batch_size"]),
    )
    text_samples: list[WeightedScores] = []
    vision_samples: list[WeightedScores] = []
    sample_ids: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for position, sample in enumerate(samples, start=1):
            instruction, context, image_paths = resolve_sample(subset, sample, args.image_dir)
            spec = build_prompt_spec(instruction, context, image_paths)
            encoded = encode_sample(
                qwen,
                qwen_processor,
                spec,
                device=device,
                min_text_tokens=int(config["text_chunk"]["min_tokens"]),
                max_text_tokens=int(config["text_chunk"]["max_tokens"]),
                vision_window=int(config["vision_window"]),
            )
            raw = compute_aligned_raw_scores(
                aligned,
                spec.query_text,
                encoded,
                _load_images(image_paths),
                window_tokens=int(config["text_scoring"]["window_tokens"]),
                stride_tokens=int(config["text_scoring"]["stride_tokens"]),
            )
            text_samples.append(WeightedScores(tuple(float(v) for v in raw.text)))
            vision_samples.append(WeightedScores(tuple(float(v) for v in raw.vision)))
            sample_ids.append(int(sample["sample_id"]))
            print(f"calibration {position}/{len(samples)} sample={sample_ids[-1]}")
    metadata = {
        "encoder_identifier": aligned.identifier,
        "encoder_revision": str(aligned_cfg["revision"]),
        "image_resize": str(aligned_cfg["image_resize"]),
        "window_tokens": int(config["text_scoring"]["window_tokens"]),
        "stride_tokens": int(config["text_scoring"]["stride_tokens"]),
        "manifest_fingerprint": manifest_fingerprint(sample_ids),
        "sample_ids": sample_ids,
    }
    calibrator = EmpiricalCDFCalibrator.fit(text_samples, vision_samples, metadata)
    calibrator.save(args.output)
    print(json.dumps({
        "output": str(args.output),
        "samples": len(samples),
        "text_candidates": calibrator.text.candidate_count,
        "vision_candidates": calibrator.vision.candidate_count,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
