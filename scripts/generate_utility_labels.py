"""Generate resumable leave-one-chunk-out labels and frozen Qwen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from mllm_prune.algorithms.utility import (
    build_utility_features,
    features_to_record,
    utility_from_nll,
)
from mllm_prune.config import path_from_env, value_from_env
from mllm_prune.data.mmcoqa import build_prompt_spec, load_subset, resolve_sample
from mllm_prune.models.qwen25vl import (
    answer_token_ids,
    candidate_embeddings,
    encode_sample,
    score_gold_answer_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/004_learned_utility_scorer.yaml"),
    )
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    split_dir = path_from_env("EXPERIMENT_004_SPLIT_DIR")
    output_root = path_from_env("EXPERIMENT_004_LABEL_DIR")
    image_dir = path_from_env("MMCOQA_IMAGE_DIR")
    model_path = path_from_env("QWEN_MODEL_PATH")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split-dir", type=Path, default=split_dir, required=split_dir is None)
    parser.add_argument("--image-dir", type=Path, default=image_dir, required=image_dir is None)
    parser.add_argument("--model-path", type=Path, default=model_path, required=model_path is None)
    parser.add_argument(
        "--output-root", type=Path, default=output_root, required=output_root is None
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--variant-batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default=value_from_env("MLLM_ATTN_IMPLEMENTATION", "sdpa"),
        choices=("flash_attention_2", "sdpa", "eager"),
    )
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


def _settings_fingerprint(config: dict, model_path: Path) -> str:
    payload = {
        "experiment": config["experiment"],
        "model_path": str(model_path.resolve()),
        "text_chunk": config["text_chunk"],
        "vision_window": config["vision_window"],
        "include_eos": config["teacher"]["include_eos_in_label"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _fingerprint(settings_fingerprint: str, manifest: Path) -> str:
    payload = {
        "settings_fingerprint": settings_fingerprint,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if bool(config["teacher"]["include_eos_in_label"]):
        raise NotImplementedError("initial experiment 004 labels answer content only")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    manifest = args.manifest or args.split_dir / f"{args.split}.json"
    subset, samples = load_subset(manifest, args.limit)
    samples = samples[args.shard_index :: args.num_shards]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing to generate labels on CPU")
    batch_size = args.variant_batch_size or int(config["label_generation"]["variant_batch_size"])
    settings_fingerprint = _settings_fingerprint(config, args.model_path)
    fingerprint = _fingerprint(settings_fingerprint, manifest)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=int(config["max_pixels"]),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": device.index or 0},
        attn_implementation=args.attn_implementation,
    ).eval()
    output_dir = args.output_root / args.split
    print(
        f"exp004 labels split={args.split} shard={args.shard_index}/{args.num_shards} "
        f"samples={len(samples)} batch={batch_size} attention={model.config._attn_implementation}"
    )
    completed = failed = 0
    with torch.inference_mode():
        for sample in samples:
            sample_id = int(sample["sample_id"])
            output = output_dir / f"{sample_id}.pt"
            if output.exists() and not args.overwrite:
                existing = torch.load(output, map_location="cpu", weights_only=True)
                if existing.get("fingerprint") == fingerprint:
                    completed += 1
                    continue
                raise ValueError(f"stale label record exists: {output}")
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                instruction, context, image_paths = resolve_sample(
                    subset, sample, args.image_dir
                )
                spec = build_prompt_spec(instruction, context, image_paths)
                encoded = encode_sample(
                    model,
                    processor,
                    spec,
                    device=device,
                    min_text_tokens=int(config["text_chunk"]["min_tokens"]),
                    max_text_tokens=int(config["text_chunk"]["max_tokens"]),
                    vision_window=int(config["vision_window"]),
                )
                text, vision, question = candidate_embeddings(encoded)
                features = build_utility_features(
                    encoded.layout, text, vision, question
                )
                gold_ids = answer_token_ids(
                    processor, str(sample["response"]), device=device
                )
                deletion_specs = [
                    None,
                    *(("text", chunk) for chunk in encoded.layout.text_chunks),
                    *(("vision", chunk) for chunk in encoded.layout.vision_chunks),
                ]
                nlls = score_gold_answer_variants(
                    model,
                    encoded,
                    gold_ids,
                    deletion_specs,
                    variant_batch_size=batch_size,
                )
                utilities = utility_from_nll(
                    nlls[0].mean_nll, [value.mean_nll for value in nlls[1:]]
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                record = {
                    "format_version": 1,
                    "fingerprint": fingerprint,
                    "settings_fingerprint": settings_fingerprint,
                    "experiment": config["experiment"],
                    "split": args.split,
                    "sample_id": sample_id,
                    "group_hint": sorted(set(sample["task_instance"].get("images_path", []))),
                    "answer": str(sample["response"]),
                    "answer_token_ids": gold_ids.cpu(),
                    "full_mean_nll": nlls[0].mean_nll,
                    "full_sum_nll": nlls[0].sum_nll,
                    "deleted_mean_nlls": torch.tensor(
                        [value.mean_nll for value in nlls[1:]], dtype=torch.float32
                    ),
                    "utility_labels": utilities,
                    "num_text_chunks": len(encoded.layout.text_chunks),
                    "num_vision_chunks": len(encoded.layout.vision_chunks),
                    "num_text_tokens": encoded.layout.num_text_tokens,
                    "num_vision_tokens": encoded.layout.num_vision_tokens,
                    "features": features_to_record(features),
                    "timing": {
                        "total_ms": elapsed_ms,
                        "vision_encode_ms": encoded.vision_encode_ms,
                    },
                    "peak_memory_gb": (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    ),
                    "attention_implementation": model.config._attn_implementation,
                    "variant_batch_size": batch_size,
                }
                _atomic_save(record, output)
                completed += 1
                print(
                    f"sample={sample_id} chunks={len(utilities)} "
                    f"utility=[{utilities.min():.4f},{utilities.max():.4f}] "
                    f"time={elapsed_ms / 1000:.1f}s peak={record['peak_memory_gb']:.2f}GB"
                )
            except Exception as error:
                failed += 1
                print(f"FAILED sample={sample_id}: {type(error).__name__}: {error}")
                raise
    print(f"completed={completed} failed={failed}")


if __name__ == "__main__":
    main()

