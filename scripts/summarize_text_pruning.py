"""Summarize experiment 001 JSONL predictions into a compact JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mllm_prune.config import path_from_env


def parse_args() -> argparse.Namespace:
    """Parse prediction and summary paths."""

    parser = argparse.ArgumentParser()
    predictions = path_from_env("EXPERIMENT_001_PREDICTIONS_PATH")
    output = path_from_env("EXPERIMENT_001_SUMMARY_PATH")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=predictions,
        required=predictions is None,
    )
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    return parser.parse_args()


def main() -> None:
    """Aggregate accuracy, compression, timing, and agreement by condition."""

    args = parse_args()
    frame = pd.read_json(args.predictions, lines=True)
    baseline = frame[frame["strategy"] == "baseline"].set_index("sample_id")
    rows = []
    group_columns = ["strategy", "target_retention", "seed"]
    for keys, group in frame.groupby(group_columns, sort=False):
        strategy, retention, seed = keys
        baseline_predictions = baseline.loc[group["sample_id"], "prediction"].to_numpy()
        rows.append(
            {
                "strategy": strategy,
                "target_retention": float(retention),
                "seed": int(seed),
                "samples": len(group),
                "accuracy": float(group["correct"].mean()),
                "actual_context_retention": float(group["actual_retention"].mean()),
                "mean_input_tokens": float(group["input_tokens"].mean()),
                "mean_visual_tokens": float(group["visual_tokens"].mean()),
                "mean_inference_ms": float(group["inference_ms"].mean()),
                "mean_pruning_ms": float(group["pruning_ms"].mean()),
                "peak_memory_gb": float(group["peak_memory_gb"].max()),
                "agreement": float((group["prediction"].to_numpy() == baseline_predictions).mean()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
