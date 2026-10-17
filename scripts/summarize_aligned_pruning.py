"""Aggregate experiment-003 shards and validate strict-budget results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mllm_prune.config import path_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    predictions = path_from_env("EXPERIMENT_003_PREDICTIONS_PATH")
    output = path_from_env("EXPERIMENT_003_SUMMARY_PATH")
    parser.add_argument("--predictions", type=Path, nargs="+", default=[predictions] if predictions else None, required=predictions is None)
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.concat([pd.read_json(path, lines=True) for path in args.predictions], ignore_index=True)
    if frame.empty:
        raise ValueError("prediction files are empty")
    keys = ["sample_id", "method", "retention", "seed"]
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        raise ValueError(f"duplicate result keys: {frame.loc[duplicate, keys].head().to_dict('records')}")
    bad_budget = frame[frame["plan"].map(lambda p: p["budget_deviation"] != 0)]
    if not bad_budget.empty:
        raise ValueError(f"strict-budget violations: {bad_budget[keys].head().to_dict('records')}")
    baseline = frame[frame.method == "baseline"].drop_duplicates("sample_id").set_index("sample_id")
    rows = []
    for (method, retention, seed), group in frame.groupby(["method", "retention", "seed"], sort=False):
        paired = baseline.reindex(group.sample_id)
        valid = paired.prediction.notna().to_numpy()
        correct = group.correct.to_numpy(dtype=bool)
        robust = group.correct_robust.to_numpy(dtype=bool)
        baseline_correct = paired.correct.fillna(False).to_numpy(dtype=bool)
        predictions = group.prediction.to_numpy()
        rows.append({
            "method": method, "retention": float(retention), "seed": int(seed),
            "samples": len(group), "accuracy": float(correct.mean()),
            "robust_accuracy": float(robust.mean()),
            "mean_candidate_retention": float(group.actual_candidate_retention.mean()),
            "mean_input_tokens": float(group.compressed_input_tokens.mean()),
            "mean_text_budget": float(group.plan.map(lambda p: p["text_budget"]).mean()),
            "mean_vision_budget": float(group.plan.map(lambda p: p["vision_budget"]).mean()),
            "mean_external_text_ms": float(group.external_text_encode_ms.mean()),
            "mean_external_vision_ms": float(group.external_vision_encode_ms.mean()),
            "mean_external_mapping_ms": float(group.external_mapping_ms.mean()),
            "mean_prefill_ms": float(group.prefill_ms.mean()),
            "mean_decode_ms": float(group.decode_ms.mean()),
            "peak_memory_gb": float(group.peak_memory_gb.max()),
            "agreement_with_baseline": float((predictions[valid] == paired.prediction.to_numpy()[valid]).mean()) if valid.any() else None,
            "correct_to_wrong": int((valid & baseline_correct & ~correct).sum()),
            "wrong_to_correct": int((valid & ~baseline_correct & correct).sum()),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
