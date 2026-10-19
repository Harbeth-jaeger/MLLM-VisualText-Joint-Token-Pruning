"""Generate resumable conditional-utility labels for experiment 005."""

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

from mllm_prune.algorithms.cascaded_utility import build_aligned_prefilter_state
from mllm_prune.algorithms.conditional_utility import (
    build_conditional_utility_features,
    conditional_features_to_record,
    conditional_selections,
    conditional_utility_from_nll,
)
from mllm_prune.algorithms.joint_pruning import build_relevance_from_token_scores
from mllm_prune.config import path_from_env, value_from_env
from mllm_prune.data.mmcoqa import build_prompt_spec, load_subset, resolve_sample
from mllm_prune.models.qwen25vl import (
    answer_token_ids,
    candidate_embeddings,
    encode_sample,
    score_gold_answer_selections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/005_learned_utility_scorer_v2.yaml"))
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    split_dir = path_from_env("EXPERIMENT_004_SPLIT_DIR")
    parser.add_argument("--split-dir", type=Path, default=split_dir, required=split_dir is None)
    parser.add_argument("--manifest", type=Path)
    for flag, env in (
        ("image-dir", "MMCOQA_IMAGE_DIR"),
        ("model-path", "QWEN_MODEL_PATH"),
        ("aligned-cache-root", "EXPERIMENT_005_ALIGNED_CACHE_DIR"),
        ("output-root", "EXPERIMENT_005_LABEL_DIR"),
    ):
        value = path_from_env(env)
        parser.add_argument(f"--{flag}", type=Path, default=value, required=value is None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--retentions", nargs="+", type=float)
    parser.add_argument("--prefilter-fractions", nargs="+", type=float)
    parser.add_argument("--variant-batch-size", type=int)
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


def _state_name(retention: float, alpha: float) -> str:
    return f"r{retention:.4f}_a{alpha:.4f}".replace(".", "p")


def _layout_payload(layout) -> dict:
    return {
        "num_text_tokens": layout.num_text_tokens,
        "num_vision_tokens": layout.num_vision_tokens,
        "text_chunks": [list(chunk.token_indices) for chunk in layout.text_chunks],
        "vision_chunks": [list(chunk.token_indices) for chunk in layout.vision_chunks],
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    manifest = args.manifest or args.split_dir / f"{args.split}.json"
    subset, samples = load_subset(manifest, args.limit)
    samples = samples[args.shard_index::args.num_shards]
    retentions = args.retentions or config["retentions"]
    alphas = args.prefilter_fractions or config["prefilter_fractions"]
    states = [(float(r), float(a)) for r in retentions for a in alphas]
    batch_size = args.variant_batch_size or int(config["label_generation"]["variant_batch_size"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing label generation on CPU")
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=256 * 28 * 28, max_pixels=int(config["max_pixels"])
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": device.index or 0},
        attn_implementation=args.attn_implementation,
    ).eval()
    output_dir = args.output_root / args.split
    print(f"exp005 labels split={args.split} shard={args.shard_index}/{args.num_shards} samples={len(samples)} states={states}")
    with torch.inference_mode():
        for position, sample in enumerate(samples, start=1):
            sample_id = int(sample["sample_id"])
            cache_path = args.aligned_cache_root / args.split / f"{sample_id}.pt"
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            settings_payload = {
                "experiment": config["experiment"],
                "model_path": str(args.model_path.resolve()),
                "cache_fingerprint": cache["fingerprint"],
                "text_chunk": config["text_chunk"],
                "vision_window": config["vision_window"],
                "features": config["features"],
            }
            settings_fingerprint = hashlib.sha256(
                json.dumps(settings_payload, sort_keys=True).encode()
            ).hexdigest()
            pending = []
            for retention, alpha in states:
                output = output_dir / f"{sample_id}_{_state_name(retention, alpha)}.pt"
                if output.exists() and not args.overwrite:
                    existing = torch.load(output, map_location="cpu", weights_only=True)
                    if existing.get("settings_fingerprint") == settings_fingerprint:
                        continue
                    raise ValueError(f"stale conditional label exists: {output}")
                pending.append((retention, alpha, output))
            if not pending:
                continue
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sample_started = time.perf_counter()
            instruction, context, image_paths = resolve_sample(subset, sample, args.image_dir)
            spec = build_prompt_spec(instruction, context, image_paths)
            encoded = encode_sample(
                model, processor, spec, device=device,
                min_text_tokens=int(config["text_chunk"]["min_tokens"]),
                max_text_tokens=int(config["text_chunk"]["max_tokens"]),
                vision_window=int(config["vision_window"]),
            )
            if cache["layout"] != _layout_payload(encoded.layout):
                raise ValueError(f"aligned cache layout mismatch for sample {sample_id}")
            relevance = build_relevance_from_token_scores(
                encoded.layout, cache["text_scores"], cache["vision_scores"],
                token_top_fraction=float(config["token_top_fraction"]),
                chunk_top_fraction=float(config["chunk_top_fraction"]),
            )
            text, vision, question = candidate_embeddings(encoded)
            gold = answer_token_ids(processor, str(sample["response"]), device=device)
            for retention, alpha, output in pending:
                started = time.perf_counter()
                state = build_aligned_prefilter_state(
                    encoded.layout, relevance, retention=retention,
                    prefilter_fraction=alpha,
                )
                features = build_conditional_utility_features(
                    encoded.layout, state, text, vision, question, relevance,
                    aligned_top_fraction=float(config["features"]["aligned_top_fraction"]),
                )
                nlls = score_gold_answer_selections(
                    model, encoded, gold, conditional_selections(state),
                    variant_batch_size=batch_size,
                )
                labels = conditional_utility_from_nll(
                    nlls[0].mean_nll, [value.mean_nll for value in nlls[1:]]
                )
                if len(labels) != len(state.original_chunk_ids):
                    raise RuntimeError("conditional label count does not match state")
                record = {
                    "format_version": 2,
                    "experiment": config["experiment"],
                    "settings_fingerprint": settings_fingerprint,
                    "cache_fingerprint": cache["fingerprint"],
                    "split": args.split,
                    "sample_id": sample_id,
                    "state_id": _state_name(retention, alpha),
                    "retention": retention,
                    "prefilter_fraction": alpha,
                    "original_capacity": state.original_capacity,
                    "prefilter_budget": state.prefilter_budget,
                    "final_budget": state.final_budget,
                    "selected_text_indices": state.selected_text_indices,
                    "selected_vision_indices": state.selected_vision_indices,
                    "text_chunk_indices": state.text_chunk_indices,
                    "vision_chunk_indices": state.vision_chunk_indices,
                    "original_chunk_ids": state.original_chunk_ids,
                    "partial_chunk_ids": state.partial_chunk_ids,
                    "state_mean_nll": nlls[0].mean_nll,
                    "deleted_mean_nlls": torch.tensor([value.mean_nll for value in nlls[1:]]),
                    "utility_labels": labels,
                    "features": conditional_features_to_record(features),
                    "answer_token_count": len(gold),
                    "timing": {
                        "state_ms": (time.perf_counter() - started) * 1000,
                        "qwen_vision_encode_ms": encoded.vision_encode_ms,
                    },
                    "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "attention_implementation": model.config._attn_implementation,
                    "variant_batch_size": batch_size,
                }
                _atomic_save(record, output)
                print(f"labels {position}/{len(samples)} sample={sample_id} state={record['state_id']} chunks={len(labels)} time={record['timing']['state_ms']/1000:.1f}s")
            print(f"sample={sample_id} all_states_time={(time.perf_counter()-sample_started):.1f}s")


if __name__ == "__main__":
    main()
