"""Validate and aggregate experiment-004 prediction shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, pstdev

from mllm_prune.config import path_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    predictions = path_from_env("EXPERIMENT_004_PREDICTIONS_PATH")
    summary = path_from_env("EXPERIMENT_004_SUMMARY_PATH")
    parser.add_argument(
        "--predictions", type=Path, nargs="+",
        default=[predictions] if predictions else None,
        required=predictions is None,
    )
    parser.add_argument("--output", type=Path, default=summary, required=summary is None)
    parser.add_argument("--expected-samples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.predictions:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("prediction files are empty")
    keys = set()
    for row in rows:
        key = (int(row["sample_id"]), row["method"], float(row["retention"]), int(row["seed"]))
        if key in keys:
            raise ValueError(f"duplicate result key: {key}")
        keys.add(key)
        plan = row["plan"]
        selected = len(plan["selected_text_indices"]) + len(plan["selected_vision_indices"])
        if plan["budget_deviation"] != 0 or selected != plan["target_total_budget"]:
            raise ValueError(f"strict-budget violation: {key}")
    baseline = {
        int(row["sample_id"]): row for row in rows if row["method"] == "baseline"
    }
    groups: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], float(row["retention"]), int(row["seed"]))].append(row)
    summary = []
    for (method, retention, seed), group in sorted(groups.items()):
        if len(group) != args.expected_samples:
            raise ValueError(
                f"incomplete condition {(method, retention, seed)}: "
                f"{len(group)} != {args.expected_samples}"
            )
        paired = [baseline.get(int(row["sample_id"])) for row in group]
        correct_to_wrong = sum(
            int(base is not None and base["correct"] and not row["correct"])
            for row, base in zip(group, paired, strict=True)
        )
        wrong_to_correct = sum(
            int(base is not None and not base["correct"] and row["correct"])
            for row, base in zip(group, paired, strict=True)
        )
        summary.append(
            {
                "method": method,
                "retention": retention,
                "seed": seed,
                "samples": len(group),
                "accuracy": mean(float(row["correct"]) for row in group),
                "robust_accuracy": mean(float(row["correct_robust"]) for row in group),
                "mean_candidate_retention": mean(row["actual_candidate_retention"] for row in group),
                "mean_input_tokens": mean(row["compressed_input_tokens"] for row in group),
                "mean_text_budget": mean(row["plan"]["text_budget"] for row in group),
                "mean_vision_budget": mean(row["plan"]["vision_budget"] for row in group),
                "mean_planning_ms": mean(row["planning_ms"] for row in group),
                "mean_prefill_ms": mean(row["prefill_ms"] for row in group),
                "mean_decode_ms": mean(row["decode_ms"] for row in group),
                "peak_memory_gb": max(row["peak_memory_gb"] for row in group),
                "correct_to_wrong": correct_to_wrong,
                "wrong_to_correct": wrong_to_correct,
            }
        )
    learned_by_retention: dict[float, list[dict]] = defaultdict(list)
    for row in summary:
        if row["method"] == "learned_utility_joint":
            learned_by_retention[row["retention"]].append(row)
    seed_aggregates = [
        {
            "method": "learned_utility_joint",
            "retention": retention,
            "training_seeds": len(values),
            "accuracy_mean": mean(value["accuracy"] for value in values),
            "accuracy_std": pstdev(value["accuracy"] for value in values),
            "robust_accuracy_mean": mean(value["robust_accuracy"] for value in values),
            "robust_accuracy_std": pstdev(value["robust_accuracy"] for value in values),
        }
        for retention, values in sorted(learned_by_retention.items())
    ]
    payload = {"conditions": summary, "learned_seed_aggregates": seed_aggregates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

