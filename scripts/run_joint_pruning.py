"""Run experiment 002 with resumable per-condition JSONL output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from mllm_prune.algorithms.joint_pruning import (
    JointPruningConfig,
    compute_relevance_scores,
    plan_joint_pruning,
)
from mllm_prune.config import path_from_env, value_from_env
from mllm_prune.data.mmcoqa import build_prompt_spec, load_subset, resolve_sample
from mllm_prune.evaluation.mmcoqa import is_correct
from mllm_prune.models.qwen25vl import candidate_embeddings, encode_sample, generate_from_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subset = path_from_env("MMCOQA_SUBSET_PATH")
    image_dir = path_from_env("MMCOQA_IMAGE_DIR")
    model_path = path_from_env("QWEN_MODEL_PATH")
    output = path_from_env("EXPERIMENT_002_PREDICTIONS_PATH")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/002_chunk_based_joint_pruning.yaml"))
    parser.add_argument("--subset", type=Path, default=subset, required=subset is None)
    parser.add_argument("--image-dir", type=Path, default=image_dir, required=image_dir is None)
    parser.add_argument("--model-path", type=Path, default=model_path, required=model_path is None)
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    parser.add_argument("--attn-implementation", default=value_from_env("MLLM_ATTN_IMPLEMENTATION", "sdpa"), choices=("flash_attention_2", "sdpa", "eager"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=("baseline", "visual_only", "fixed_joint", "random_joint", "proposed_joint"))
    parser.add_argument("--retentions", nargs="+", type=float)
    parser.add_argument("--random-seeds", nargs="+", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--verify-baseline", action="store_true", help="Compare custom full-input decoding against Transformers generate.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def conditions(args: argparse.Namespace, config: dict) -> list[tuple[str, float, int]]:
    methods = args.methods or ["baseline", *config["methods"]]
    retentions = args.retentions or config["retentions"]
    random_seeds = args.random_seeds or config["random_seeds"]
    result: list[tuple[str, float, int]] = []
    if "baseline" in methods:
        result.append(("baseline", 1.0, 0))
    for retention in retentions:
        for method in methods:
            if method == "baseline":
                continue
            seeds = random_seeds if method == "random_joint" else [0]
            result.extend((method, float(retention), int(seed)) for seed in seeds)
    return result


def load_completed(path: Path) -> set[tuple[int, str, float, int]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add((int(row["sample_id"]), row["method"], float(row["retention"]), int(row["seed"])))
    return completed


def make_algorithm_config(method: str, retention: float, seed: int, config: dict) -> JointPruningConfig:
    return JointPruningConfig(
        method=method,
        retention=retention,
        seed=seed,
        modality_temperature=float(config["modality_temperature"]),
        chunk_temperature=float(config["chunk_temperature"]),
        token_top_fraction=float(config["token_top_fraction"]),
        chunk_top_fraction=float(config["chunk_top_fraction"]),
        min_text_fraction=float(config["min_text_fraction"]),
        min_vision_fraction=float(config["min_vision_fraction"]),
        fixed_text_fraction=float(config["fixed_text_fraction"]),
    )


def _reference_generate(model, encoded, max_new_tokens: int) -> tuple[int, ...]:
    generated = model.generate(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        pixel_values=encoded.pixel_values,
        image_grid_thw=encoded.image_grid_thw,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tuple(int(value) for value in generated[0, encoded.input_ids.shape[1] :].tolist())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    max_new_tokens = args.max_new_tokens or int(config["max_new_tokens"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing to silently run this GPU experiment on CPU")
    subset, samples = load_subset(args.subset, args.limit)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    samples = samples[args.shard_index :: args.num_shards]
    run_conditions = conditions(args, config)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed = load_completed(args.output)

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
    configured_eos = model.generation_config.eos_token_id
    if configured_eos is None:
        configured_eos = model.config.eos_token_id
    eos_values = configured_eos if isinstance(configured_eos, (list, tuple)) else [configured_eos]
    eos_ids = {int(value) for value in eos_values if value is not None}
    if processor.tokenizer.eos_token_id is not None:
        eos_ids.add(int(processor.tokenizer.eos_token_id))
    if not eos_ids:
        raise ValueError("model and tokenizer provide no EOS token ID")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = len(samples) * len(run_conditions)
    print(
        f"Running experiment 002 shard {args.shard_index}/{args.num_shards}: "
        f"{total} sample-condition pairs; resume has {len(completed)} keys; "
        f"attention={model.config._attn_implementation}"
    )
    with args.output.open("a", encoding="utf-8") as output_handle, torch.inference_mode():
        for sample in samples:
            sample_id = int(sample["sample_id"])
            instruction, context, image_paths = resolve_sample(subset, sample, args.image_dir)
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
            text_embeddings, vision_embeddings, query_embeddings = candidate_embeddings(encoded)
            relevance_start = time.perf_counter()
            relevance = compute_relevance_scores(
                encoded.layout,
                text_embeddings,
                vision_embeddings,
                query_embeddings,
                token_top_fraction=float(config["token_top_fraction"]),
                chunk_top_fraction=float(config["chunk_top_fraction"]),
            )
            relevance_ms = (time.perf_counter() - relevance_start) * 1000
            for method, retention, seed in run_conditions:
                key = (sample_id, method, retention, seed)
                if key in completed:
                    continue
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                prune_start = time.perf_counter()
                plan = plan_joint_pruning(
                    encoded.layout,
                    text_embeddings,
                    vision_embeddings,
                    query_embeddings,
                    make_algorithm_config(method, retention, seed, config),
                    relevance,
                )
                planning_ms = (time.perf_counter() - prune_start) * 1000
                result = generate_from_plan(
                    model,
                    encoded,
                    plan,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_ids,
                )
                if args.verify_baseline and method == "baseline":
                    reference = _reference_generate(model, encoded, max_new_tokens)
                    if result.token_ids != reference:
                        raise AssertionError(
                            f"custom baseline mismatch for sample {sample_id}: "
                            f"custom={result.token_ids}, reference={reference}"
                        )
                prediction = processor.decode(
                    result.token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                answer = str(sample["response"])
                row = {
                    "experiment": config["experiment"],
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "attention_implementation": model.config._attn_implementation,
                    "sample_id": sample_id,
                    "method": method,
                    "retention": retention,
                    "seed": seed,
                    "original_input_tokens": int(encoded.input_ids.shape[1]),
                    "protected_tokens": int(encoded.protected_sequence_indices.numel()),
                    "original_text_candidates": encoded.layout.num_text_tokens,
                    "original_vision_candidates": encoded.layout.num_vision_tokens,
                    "compressed_input_tokens": result.compressed_input_tokens,
                    "target_candidate_retention": retention,
                    "actual_candidate_retention": plan.total_budget / (encoded.layout.num_text_tokens + encoded.layout.num_vision_tokens),
                    "plan": plan.to_dict(),
                    "vision_encode_ms": encoded.vision_encode_ms,
                    "relevance_ms": relevance_ms,
                    "planning_ms": planning_ms,
                    "pruning_ms": relevance_ms + planning_ms,
                    "prefill_ms": result.prefill_ms,
                    "decode_ms": result.decode_ms,
                    "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
                    "output_tokens": len(result.token_ids),
                    "prediction": prediction,
                    "answer": answer,
                    "correct": is_correct(prediction, answer),
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed.add(key)
                print(
                    f"sample={sample_id} method={method} r={retention:.2f} "
                    f"tokens={result.compressed_input_tokens} correct={row['correct']}"
                )


if __name__ == "__main__":
    main()
