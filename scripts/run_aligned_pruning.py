"""Run experiment 003 with frozen aligned scoring and resumable JSONL output."""

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
from mllm_prune.algorithms.calibration import EmpiricalCDFCalibrator
from mllm_prune.algorithms.joint_pruning import (
    JointPruningConfig,
    build_relevance_from_token_scores,
    compute_relevance_scores,
    plan_joint_pruning,
)
from mllm_prune.config import path_from_env, value_from_env
from mllm_prune.data.mmcoqa import build_prompt_spec, load_subset, resolve_sample
from mllm_prune.evaluation.mmcoqa import is_correct, is_correct_robust
from mllm_prune.models.aligned_encoder import CLIPAlignedEncoder
from mllm_prune.models.qwen25vl import candidate_embeddings, encode_sample, generate_from_plan


METHODS = (
    "baseline", "fixed_joint", "random_joint", "raw_qwen_joint",
    "aligned_uncalibrated", "aligned_calibrated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subset = path_from_env("MMCOQA_SUBSET_PATH")
    image_dir = path_from_env("MMCOQA_IMAGE_DIR")
    qwen_path = path_from_env("QWEN_MODEL_PATH")
    aligned_path = path_from_env("ALIGNED_ENCODER_PATH")
    calibration = path_from_env("EXPERIMENT_003_CALIBRATION_PATH")
    output = path_from_env("EXPERIMENT_003_PREDICTIONS_PATH")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/003_aligned_encoder_scoring.yaml"))
    parser.add_argument("--subset", type=Path, default=subset, required=subset is None)
    parser.add_argument("--image-dir", type=Path, default=image_dir, required=image_dir is None)
    parser.add_argument("--qwen-path", type=Path, default=qwen_path, required=qwen_path is None)
    parser.add_argument("--aligned-path", type=Path, default=aligned_path, required=aligned_path is None)
    parser.add_argument("--calibration", type=Path, default=calibration, required=calibration is None)
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--retentions", nargs="+", type=float)
    parser.add_argument("--random-seeds", nargs="+", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--verify-baseline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attn-implementation", default=value_from_env("MLLM_ATTN_IMPLEMENTATION", "sdpa"), choices=("flash_attention_2", "sdpa", "eager"))
    return parser.parse_args()


def _conditions(args: argparse.Namespace, config: dict) -> list[tuple[str, float, int]]:
    methods = args.methods or ["baseline", *config["methods"]]
    retentions = args.retentions or config["retentions"]
    seeds = args.random_seeds or config["random_seeds"]
    result = [("baseline", 1.0, 0)] if "baseline" in methods else []
    for retention in retentions:
        for method in methods:
            if method == "baseline":
                continue
            method_seeds = seeds if method == "random_joint" else [0]
            result.extend((method, float(retention), int(seed)) for seed in method_seeds)
    return result


def _algorithm_config(method: str, retention: float, seed: int, config: dict) -> JointPruningConfig:
    return JointPruningConfig(
        method=method, retention=retention, seed=seed,
        modality_temperature=float(config["modality_temperature"]),
        chunk_temperature=float(config["chunk_temperature"]),
        token_top_fraction=float(config["token_top_fraction"]),
        chunk_top_fraction=float(config["chunk_top_fraction"]),
        min_text_fraction=float(config["min_text_fraction"]),
        min_vision_fraction=float(config["min_vision_fraction"]),
        fixed_text_fraction=float(config["fixed_text_fraction"]),
    )


def _load_completed(path: Path) -> set[tuple[int, str, float, int]]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            completed.add((int(row["sample_id"]), row["method"], float(row["retention"]), int(row["seed"])))
    return completed


def _load_images(paths: list[Path]) -> list[Image.Image]:
    result = []
    for path in paths:
        with Image.open(path) as image:
            result.append(image.convert("RGB").copy())
    return result


def _stats(values: torch.Tensor) -> dict:
    values = values.float()
    return {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
    }


def _reference_generate(model, encoded, max_new_tokens: int) -> tuple[int, ...]:
    generated = model.generate(
        input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
        pixel_values=encoded.pixel_values, image_grid_thw=encoded.image_grid_thw,
        max_new_tokens=max_new_tokens, do_sample=False,
    )
    return tuple(int(value) for value in generated[0, encoded.input_ids.shape[1]:].tolist())


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing to run experiment 003 on CPU")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    subset, samples = load_subset(args.subset, args.limit)
    samples = samples[args.shard_index::args.num_shards]
    conditions = _conditions(args, config)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed = _load_completed(args.output)

    processor = AutoProcessor.from_pretrained(
        args.qwen_path, min_pixels=256 * 28 * 28, max_pixels=int(config["max_pixels"])
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
    configured_eos = model.generation_config.eos_token_id or model.config.eos_token_id
    eos_values = configured_eos if isinstance(configured_eos, (list, tuple)) else [configured_eos]
    eos_ids = {int(value) for value in eos_values if value is not None}
    if processor.tokenizer.eos_token_id is not None:
        eos_ids.add(int(processor.tokenizer.eos_token_id))
    max_new_tokens = args.max_new_tokens or int(config["max_new_tokens"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"experiment 003 shard {args.shard_index}/{args.num_shards}: samples={len(samples)} conditions={len(conditions)} attention={model.config._attn_implementation}")
    with args.output.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for sample in samples:
            sample_id = int(sample["sample_id"])
            instruction, context, image_paths = resolve_sample(subset, sample, args.image_dir)
            spec = build_prompt_spec(instruction, context, image_paths)
            encoded = encode_sample(
                model, processor, spec, device=device,
                min_text_tokens=int(config["text_chunk"]["min_tokens"]),
                max_text_tokens=int(config["text_chunk"]["max_tokens"]),
                vision_window=int(config["vision_window"]),
            )
            text_embeddings, vision_embeddings, query_embeddings = candidate_embeddings(encoded)
            raw_qwen = compute_relevance_scores(
                encoded.layout, text_embeddings, vision_embeddings, query_embeddings,
                token_top_fraction=float(config["token_top_fraction"]),
                chunk_top_fraction=float(config["chunk_top_fraction"]),
            )
            aligned_raw = compute_aligned_raw_scores(
                aligned, spec.query_text, encoded, _load_images(image_paths),
                window_tokens=int(config["text_scoring"]["window_tokens"]),
                stride_tokens=int(config["text_scoring"]["stride_tokens"]),
            )
            calibrated_text, calibrated_vision = calibrator.transform(aligned_raw.text, aligned_raw.vision)
            raw_relevance = build_relevance_from_token_scores(
                encoded.layout, aligned_raw.text, aligned_raw.vision,
                token_top_fraction=float(config["token_top_fraction"]),
                chunk_top_fraction=float(config["chunk_top_fraction"]),
            )
            calibrated_relevance = build_relevance_from_token_scores(
                encoded.layout, calibrated_text, calibrated_vision,
                token_top_fraction=float(config["token_top_fraction"]),
                chunk_top_fraction=float(config["chunk_top_fraction"]),
            )
            for method, retention, seed in conditions:
                key = (sample_id, method, retention, seed)
                if key in completed:
                    continue
                if method == "raw_qwen_joint" or method == "baseline":
                    relevance = raw_qwen
                elif method == "aligned_uncalibrated":
                    relevance = raw_relevance
                else:
                    relevance = calibrated_relevance
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                plan = plan_joint_pruning(
                    encoded.layout, text_embeddings, vision_embeddings, query_embeddings,
                    _algorithm_config(method, retention, seed, config), relevance,
                )
                planning_ms = (time.perf_counter() - started) * 1000
                result = generate_from_plan(
                    model, encoded, plan, max_new_tokens=max_new_tokens, eos_token_ids=eos_ids
                )
                if args.verify_baseline and method == "baseline":
                    reference = _reference_generate(model, encoded, max_new_tokens)
                    if result.token_ids != reference:
                        raise AssertionError(f"baseline mismatch sample={sample_id}: custom={result.token_ids} reference={reference}")
                prediction = processor.decode(
                    result.token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                answer = str(sample["response"])
                row = {
                    "experiment": config["experiment"],
                    "sample_id": sample_id, "method": method,
                    "retention": retention, "seed": seed,
                    "shard_index": args.shard_index, "num_shards": args.num_shards,
                    "attention_implementation": model.config._attn_implementation,
                    "original_input_tokens": int(encoded.input_ids.shape[1]),
                    "protected_tokens": int(encoded.protected_sequence_indices.numel()),
                    "original_text_candidates": encoded.layout.num_text_tokens,
                    "original_vision_candidates": encoded.layout.num_vision_tokens,
                    "compressed_input_tokens": result.compressed_input_tokens,
                    "actual_candidate_retention": plan.total_budget / (encoded.layout.num_text_tokens + encoded.layout.num_vision_tokens),
                    "plan": plan.to_dict(),
                    "score_stats": {
                        "aligned_raw_text": _stats(aligned_raw.text),
                        "aligned_raw_vision": _stats(aligned_raw.vision),
                        "aligned_calibrated_text": _stats(calibrated_text),
                        "aligned_calibrated_vision": _stats(calibrated_vision),
                    },
                    "vision_encode_ms": encoded.vision_encode_ms,
                    "external_text_encode_ms": aligned_raw.text_encode_ms,
                    "external_vision_encode_ms": aligned_raw.vision_encode_ms,
                    "external_mapping_ms": aligned_raw.mapping_ms,
                    "planning_ms": planning_ms,
                    "prefill_ms": result.prefill_ms, "decode_ms": result.decode_ms,
                    "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "output_tokens": len(result.token_ids),
                    "prediction": prediction, "answer": answer,
                    "correct": is_correct(prediction, answer),
                    "correct_robust": is_correct_robust(prediction, answer),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed.add(key)
                print(f"sample={sample_id} method={method} r={retention:.2f} tokens={result.compressed_input_tokens} correct={row['correct']}")


if __name__ == "__main__":
    main()
