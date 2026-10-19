"""Create group-aware train/validation manifests while preserving fixed test 50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mllm_prune.config import path_from_env
from mllm_prune.data.splits import split_train_validation, write_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    annotation = path_from_env("MMCOQA_ANNOTATION_PATH")
    test = path_from_env("MMCOQA_SUBSET_PATH")
    output = path_from_env("EXPERIMENT_004_SPLIT_DIR")
    parser.add_argument("--annotation", type=Path, default=annotation, required=annotation is None)
    parser.add_argument("--test", type=Path, default=test, required=test is None)
    parser.add_argument(
        "--output-dir", type=Path, default=output, required=output is None
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = json.loads(args.annotation.read_text(encoding="utf-8"))
    fixed_test = json.loads(args.test.read_text(encoding="utf-8"))
    test_ids = {int(sample["sample_id"]) for sample in fixed_test["data"]}
    train, validation, test, audit = split_train_validation(
        full["data"],
        test_ids,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    expected = sorted(test_ids)
    actual = sorted(int(sample["sample_id"]) for sample in test)
    if actual != expected:
        raise AssertionError("generated test split differs from fixed test manifest")
    write_subset(args.output_dir / "train.json", full["meta_data"], train, "train", audit)
    write_subset(
        args.output_dir / "validation.json", full["meta_data"], validation, "validation", audit
    )
    # Preserve the original fixed-test order and metadata exactly where possible.
    write_subset(
        args.output_dir / "test.json",
        fixed_test["meta_data"],
        fixed_test["data"],
        "test",
        audit,
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()

