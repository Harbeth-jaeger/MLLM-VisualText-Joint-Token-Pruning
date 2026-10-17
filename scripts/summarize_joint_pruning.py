"""Aggregate experiment 002 predictions into condition-level diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mllm_prune.config import path_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    predictions = path_from_env("EXPERIMENT_002_PREDICTIONS_PATH")
    output = path_from_env("EXPERIMENT_002_SUMMARY_PATH")
    parser.add_argument("--predictions", type=Path, nargs="+", default=[predictions] if predictions else None, required=predictions is None)
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.concat(
        [pd.read_json(path, lines=True) for path in args.predictions], ignore_index=True
    )
    duplicate_keys = ["sample_id", "method", "retention", "seed"]
    duplicates = frame.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        rows = frame.loc[duplicates, duplicate_keys].to_dict("records")[:5]
        raise ValueError(f"duplicate experiment keys across prediction files: {rows}")
    if frame.empty:
        raise ValueError("prediction file is empty")
    baseline = frame[frame["method"] == "baseline"].drop_duplicates("sample_id").set_index("sample_id")
    rows = []
    for keys, group in frame.groupby(["method", "retention", "seed"], sort=False):
        method, retention, seed = keys
        paired = baseline.reindex(group["sample_id"])
        valid = paired["prediction"].notna().to_numpy()
        predictions = group["prediction"].to_numpy()
        correct = group["correct"].to_numpy(dtype=bool)
        baseline_correct = paired["correct"].fillna(False).to_numpy(dtype=bool)
        rows.append(
            {
                "method": method,
                "retention": float(retention),
                "seed": int(seed),
                "samples": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "mean_candidate_retention": float(group["actual_candidate_retention"].mean()),
                "mean_budget_deviation": float(group["plan"].map(lambda value: value["budget_deviation"]).mean()),
                "samples_with_budget_deviation": int(group["plan"].map(lambda value: value["budget_deviation"] > 0).sum()),
                "mean_compressed_input_tokens": float(group["compressed_input_tokens"].mean()),
                "mean_text_budget": float(group["plan"].map(lambda value: value["text_budget"]).mean()),
                "mean_vision_budget": float(group["plan"].map(lambda value: value["vision_budget"]).mean()),
                "mean_vision_encode_ms": float(group["vision_encode_ms"].mean()),
                "mean_relevance_ms": float(group["relevance_ms"].mean()),
                "mean_planning_ms": float(group["planning_ms"].mean()),
                "mean_pruning_ms": float(group["pruning_ms"].mean()),
                "mean_prefill_ms": float(group["prefill_ms"].mean()),
                "mean_decode_ms": float(group["decode_ms"].mean()),
                "peak_memory_gb": float(group["peak_memory_gb"].max()),
                "paired_samples": int(valid.sum()),
                "agreement_with_baseline": float((predictions[valid] == paired["prediction"].to_numpy()[valid]).mean()) if valid.any() else None,
                "correct_to_wrong": int((valid & baseline_correct & ~correct).sum()),
                "wrong_to_correct": int((valid & ~baseline_correct & correct).sum()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
