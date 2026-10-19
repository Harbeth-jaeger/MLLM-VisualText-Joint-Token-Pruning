"""Evaluate learned utility checkpoints under strict joint token budgets."""

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
from mllm_prune.algorithms.utility import build_utility_features, plan_utility_pruning
from mllm_prune.config import path_from_env, value_from_env
from mllm_prune.data.mmcoqa import build_prompt_spec, load_subset, resolve_sample
from mllm_prune.evaluation.mmcoqa import is_correct, is_correct_robust
from mllm_prune.models.qwen25vl import candidate_embeddings, encode_sample, generate_from_plan
from mllm_prune.models.utility_scorer import load_scorer_artifact


METHODS = ("baseline", "fixed_joint", "random_joint", "raw_qwen_joint", "learned_utility_joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/004_learned_utility_scorer.yaml"),
    )
    split_dir = path_from_env("EXPERIMENT_004_SPLIT_DIR")
    image_dir = path_from_env("MMCOQA_IMAGE_DIR")
    model_path = path_from_env("QWEN_MODEL_PATH")
    checkpoint_dir = path_from_env("EXPERIMENT_004_CHECKPOINT_DIR")
    output = path_from_env("EXPERIMENT_004_PREDICTIONS_PATH")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--split-dir", type=Path, default=split_dir, required=split_dir is None)
    parser.add_argument("--image-dir", type=Path, default=image_dir, required=image_dir is None)
    parser.add_argument("--model-path", type=Path, default=model_path, required=model_path is None)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=checkpoint_dir,
        required=checkpoint_dir is None,
    )
    parser.add_argument("--checkpoints", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--retentions", nargs="+", type=float)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default=value_from_env("MLLM_ATTN_IMPLEMENTATION", "sdpa"),
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    return parser.parse_args()


def _base_config(method: str, retention: float, seed: int, config: dict) -> JointPruningConfig:
    return JointPruningConfig(
        method=method,
        retention=retention,
        seed=seed,
        token_top_fraction=float(config["token_top_fraction"]),
        chunk_top_fraction=float(config["chunk_top_fraction"]),
        fixed_text_fraction=0.5,
    )


def _completed(path: Path) -> set[tuple[int, str, float, int]]:
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result.add((int(row["sample_id"]), row["method"], float(row["retention"]), int(row["seed"])))
    return result


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    subset_path = args.subset or args.split_dir / "test.json"
    subset, samples = load_subset(subset_path, args.limit)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    samples = samples[args.shard_index :: args.num_shards]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing to evaluate on CPU")
    checkpoint_paths = args.checkpoints or [
        args.checkpoint_dir / f"seed_{seed}.pt" for seed in config["training"]["seeds"]
    ]
    scorers = []
    for path in checkpoint_paths:
        scorer, artifact = load_scorer_artifact(path, device)
        scorers.append((int(artifact["metadata"]["seed"]), scorer, artifact))
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
    configured_eos = model.generation_config.eos_token_id or model.config.eos_token_id
    eos_values = configured_eos if isinstance(configured_eos, (list, tuple)) else [configured_eos]
    eos_ids = {int(value) for value in eos_values if value is not None}
    if processor.tokenizer.eos_token_id is not None:
        eos_ids.add(int(processor.tokenizer.eos_token_id))
    methods = args.methods or list(METHODS)
    retentions = args.retentions or config["retentions"]
    max_new_tokens = args.max_new_tokens or int(config["max_new_tokens"])
    if args.overwrite and args.output.exists():
        args.output.unlink()
    done = _completed(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"exp004 eval shard={args.shard_index}/{args.num_shards} samples={len(samples)} "
        f"attention={model.config._attn_implementation}"
    )
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
            text, vision, question = candidate_embeddings(encoded)
            relevance = compute_relevance_scores(
                encoded.layout, text, vision, question,
                token_top_fraction=float(config["token_top_fraction"]),
                chunk_top_fraction=float(config["chunk_top_fraction"]),
            )
            features = build_utility_features(encoded.layout, text, vision, question)
            predictions = {}
            for seed, scorer, artifact in scorers:
                normalized = scorer(
                    features.chunk_features.float(), features.question_feature.float(),
                    features.modality_ids, features.structural_features,
                )
                predictions[seed] = (
                    normalized * float(artifact["label_scale"])
                    + float(artifact["label_center"])
                )
            conditions: list[tuple[str, float, int]] = []
            if "baseline" in methods:
                conditions.append(("baseline", 1.0, 0))
            for retention in retentions:
                for method in methods:
                    if method == "baseline":
                        continue
                    seeds = (
                        [seed for seed, _, _ in scorers]
                        if method == "learned_utility_joint"
                        else args.random_seeds if method == "random_joint" else [0]
                    )
                    conditions.extend((method, float(retention), int(seed)) for seed in seeds)
            for method, retention, seed in conditions:
                key = (sample_id, method, retention, seed)
                if key in done:
                    continue
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                if method == "learned_utility_joint":
                    plan = plan_utility_pruning(
                        encoded.layout, predictions[seed], relevance,
                        retention=retention,
                        density_gamma=float(config["planning"]["density_gamma"]),
                        seed=seed,
                    )
                else:
                    plan = plan_joint_pruning(
                        encoded.layout, text, vision, question,
                        _base_config(method, retention, seed, config), relevance,
                    )
                planning_ms = (time.perf_counter() - started) * 1000
                result = generate_from_plan(
                    model, encoded, plan,
                    max_new_tokens=max_new_tokens, eos_token_ids=eos_ids,
                )
                prediction = processor.decode(
                    result.token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                answer = str(sample["response"])
                row = {
                    "experiment": config["experiment"],
                    "sample_id": sample_id,
                    "method": method,
                    "retention": retention,
                    "seed": seed,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "attention_implementation": model.config._attn_implementation,
                    "plan": plan.to_dict(),
                    "original_input_tokens": int(encoded.input_ids.shape[1]),
                    "compressed_input_tokens": result.compressed_input_tokens,
                    "actual_candidate_retention": plan.total_budget / (
                        encoded.layout.num_text_tokens + encoded.layout.num_vision_tokens
                    ),
                    "vision_encode_ms": encoded.vision_encode_ms,
                    "planning_ms": planning_ms,
                    "prefill_ms": result.prefill_ms,
                    "decode_ms": result.decode_ms,
                    "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "prediction": prediction,
                    "answer": answer,
                    "correct": is_correct(prediction, answer),
                    "correct_robust": is_correct_robust(prediction, answer),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done.add(key)
                print(
                    f"sample={sample_id} method={method} seed={seed} r={retention:.2f} "
                    f"tokens={result.compressed_input_tokens} correct={row['correct']}"
                )


if __name__ == "__main__":
    main()

