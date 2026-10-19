"""Cache calibrated aligned token scores for experiment 005."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from PIL import Image
import torch
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from mllm_prune.algorithms.aligned_scoring import compute_aligned_raw_scores
from mllm_prune.algorithms.calibration import EmpiricalCDFCalibrator
from mllm_prune.config import path_from_env, value_from_env
from mllm_prune.data.mmcoqa import build_prompt_spec, load_subset, resolve_sample
from mllm_prune.models.aligned_encoder import CLIPAlignedEncoder
from mllm_prune.models.qwen25vl import encode_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/005_learned_utility_scorer_v2.yaml"))
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    split_dir = path_from_env("EXPERIMENT_004_SPLIT_DIR")
    parser.add_argument("--split-dir", type=Path, default=split_dir, required=split_dir is None)
    parser.add_argument("--manifest", type=Path)
    for flag, env in (
        ("image-dir", "MMCOQA_IMAGE_DIR"),
        ("qwen-path", "QWEN_MODEL_PATH"),
        ("aligned-path", "ALIGNED_ENCODER_PATH"),
        ("calibration", "EXPERIMENT_003_CALIBRATION_PATH"),
        ("output-root", "EXPERIMENT_005_ALIGNED_CACHE_DIR"),
    ):
        value = path_from_env(env)
        parser.add_argument(f"--{flag}", type=Path, default=value, required=value is None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attn-implementation", default=value_from_env("MLLM_ATTN_IMPLEMENTATION", "sdpa"), choices=("flash_attention_2", "sdpa", "eager"))
    return parser.parse_args()


def _atomic_save(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(record, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _layout_payload(layout) -> dict:
    return {
        "num_text_tokens": layout.num_text_tokens,
        "num_vision_tokens": layout.num_vision_tokens,
        "text_chunks": [list(chunk.token_indices) for chunk in layout.text_chunks],
        "vision_chunks": [list(chunk.token_indices) for chunk in layout.vision_chunks],
    }


def _fingerprint(config: dict, qwen_path: Path, aligned_path: Path, calibration: Path) -> str:
    payload = {
        "experiment": config["experiment"],
        "qwen": str(qwen_path.resolve()),
        "aligned": str(aligned_path.resolve()),
        "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "max_pixels": config["max_pixels"],
        "text_chunk": config["text_chunk"],
        "vision_window": config["vision_window"],
        "aligned_encoder": config["aligned_encoder"],
        "text_scoring": config["text_scoring"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    manifest = args.manifest or args.split_dir / f"{args.split}.json"
    subset, samples = load_subset(manifest, args.limit)
    samples = samples[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing aligned caching on CPU")
    fingerprint = _fingerprint(config, args.qwen_path, args.aligned_path, args.calibration)
    processor = AutoProcessor.from_pretrained(
        args.qwen_path, min_pixels=256 * 28 * 28, max_pixels=int(config["max_pixels"])
    )
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_path, dtype=torch.bfloat16, device_map={"": device.index or 0},
        attn_implementation=args.attn_implementation,
    ).eval()
    aligned_cfg = config["aligned_encoder"]
    aligned = CLIPAlignedEncoder(
        args.aligned_path, device=device, dtype=torch.bfloat16,
        text_batch_size=int(aligned_cfg["text_batch_size"]),
        image_batch_size=int(aligned_cfg["image_batch_size"]),
    )
    calibrator = EmpiricalCDFCalibrator.load(args.calibration)
    calibrator.validate_compatibility({
        "encoder_identifier": aligned.identifier,
        "encoder_revision": str(aligned_cfg["revision"]),
        "image_resize": str(aligned_cfg["image_resize"]),
        "window_tokens": int(config["text_scoring"]["window_tokens"]),
        "stride_tokens": int(config["text_scoring"]["stride_tokens"]),
    })
    output_dir = args.output_root / args.split
    print(f"exp005 aligned split={args.split} shard={args.shard_index}/{args.num_shards} samples={len(samples)}")
    with torch.inference_mode():
        for position, sample in enumerate(samples, start=1):
            sample_id = int(sample["sample_id"])
            output = output_dir / f"{sample_id}.pt"
            if output.exists() and not args.overwrite:
                existing = torch.load(output, map_location="cpu", weights_only=True)
                if existing.get("fingerprint") == fingerprint:
                    continue
                raise ValueError(f"stale aligned cache exists: {output}")
            started = time.perf_counter()
            instruction, context, image_paths = resolve_sample(subset, sample, args.image_dir)
            spec = build_prompt_spec(instruction, context, image_paths)
            encoded = encode_sample(
                qwen, processor, spec, device=device,
                min_text_tokens=int(config["text_chunk"]["min_tokens"]),
                max_text_tokens=int(config["text_chunk"]["max_tokens"]),
                vision_window=int(config["vision_window"]),
            )
            raw = compute_aligned_raw_scores(
                aligned, spec.query_text, encoded, _load_images(image_paths),
                window_tokens=int(config["text_scoring"]["window_tokens"]),
                stride_tokens=int(config["text_scoring"]["stride_tokens"]),
            )
            text, vision = calibrator.transform(raw.text, raw.vision)
            record = {
                "format_version": 1,
                "experiment": config["experiment"],
                "fingerprint": fingerprint,
                "sample_id": sample_id,
                "layout": _layout_payload(encoded.layout),
                "text_scores": text.float().cpu(),
                "vision_scores": vision.float().cpu(),
                "raw_text_scores": raw.text.float().cpu(),
                "raw_vision_scores": raw.vision.float().cpu(),
                "timing": {
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "qwen_vision_encode_ms": encoded.vision_encode_ms,
                    "aligned_text_encode_ms": raw.text_encode_ms,
                    "aligned_vision_encode_ms": raw.vision_encode_ms,
                    "mapping_ms": raw.mapping_ms,
                },
            }
            _atomic_save(record, output)
            print(f"aligned {position}/{len(samples)} sample={sample_id} time={record['timing']['total_ms']/1000:.1f}s")


if __name__ == "__main__":
    main()
