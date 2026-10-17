"""Create a deterministic experiment-003 calibration subset disjoint from test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mllm_prune.config import path_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    annotation = path_from_env("MMCOQA_ANNOTATION_PATH")
    test_subset = path_from_env("MMCOQA_SUBSET_PATH")
    output = path_from_env("EXPERIMENT_003_CALIBRATION_SUBSET_PATH")
    parser.add_argument("--annotation", type=Path, default=annotation, required=annotation is None)
    parser.add_argument("--test-subset", type=Path, default=test_subset, required=test_subset is None)
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    parser.add_argument("--size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = json.loads(args.annotation.read_text(encoding="utf-8"))
    test = json.loads(args.test_subset.read_text(encoding="utf-8"))
    test_ids = {int(row["sample_id"]) for row in test["data"]}
    test_images = {
        name for row in test["data"] for name in row["task_instance"]["images_path"]
    }
    eligible = [
        row
        for row in full["data"]
        if int(row["sample_id"]) not in test_ids
        and not (set(row["task_instance"]["images_path"]) & test_images)
    ]
    eligible.sort(key=lambda row: int(row["sample_id"]))
    if args.size < 1 or len(eligible) < args.size:
        raise ValueError(f"requested {args.size} samples but only {len(eligible)} are disjoint")
    selected = eligible[: args.size]
    payload = {
        "meta_data": full["meta_data"],
        "selection": {
            "experiment": "003_aligned_encoder_scoring",
            "rule": "lowest sample IDs after excluding test IDs and shared image paths",
            "test_sample_count": len(test_ids),
            "test_image_count": len(test_images),
            "eligible_count": len(eligible),
            "selected_count": len(selected),
        },
        "data": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {len(selected)} calibration samples to {args.output}; "
        f"test ID overlap=0; shared image-path overlap=0"
    )


if __name__ == "__main__":
    main()
