"""Select the fixed 50-sample MMCoQA subset for experiment 001.

Input is the official ``MMCoQA.json`` annotation. Output is a compact JSON file
that preserves the task instruction and full sample records for later inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from mllm_prune.config import path_from_env


def parse_args() -> argparse.Namespace:
    """Parse dataset selection arguments."""

    parser = argparse.ArgumentParser()
    annotation = path_from_env("MMCOQA_ANNOTATION_PATH")
    output = path_from_env("EXPERIMENT_001_SUBSET_PATH")
    parser.add_argument(
        "--annotation",
        type=Path,
        default=annotation,
        required=annotation is None,
        help="Defaults to MMCOQA_ANNOTATION_PATH from .env.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output,
        required=output is None,
        help="Defaults to EXPERIMENT_001_SUBSET_PATH from .env.",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--min-context-words", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_samples(
    annotation: dict,
    num_samples: int,
    min_context_words: int,
    seed: int,
) -> list[dict]:
    """Randomly select sufficiently long samples using a fixed seed."""

    eligible = [
        sample
        for sample in annotation["data"]
        if len(sample["task_instance"]["context"].split()) >= min_context_words
    ]
    if len(eligible) < num_samples:
        raise ValueError(
            f"Only {len(eligible)} samples contain at least {min_context_words} words; "
            f"cannot select {num_samples}."
        )
    return random.Random(seed).sample(eligible, num_samples)


def main() -> None:
    """Create the deterministic experiment subset without modifying source data."""

    args = parse_args()
    with args.annotation.open(encoding="utf-8") as handle:
        annotation = json.load(handle)
    selected = select_samples(
        annotation,
        num_samples=args.num_samples,
        min_context_words=args.min_context_words,
        seed=args.seed,
    )
    result = {
        "meta_data": annotation["meta_data"],
        "selection": {
            "source": str(args.annotation),
            "num_samples": args.num_samples,
            "min_context_words": args.min_context_words,
            "seed": args.seed,
            "sample_ids": [sample["sample_id"] for sample in selected],
        },
        "data": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(selected)} samples to {args.output}")


if __name__ == "__main__":
    main()
