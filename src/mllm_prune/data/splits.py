"""Leakage-aware group splitting for MMCoQA utility-scorer training."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


class _DisjointSet:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _sample_id(sample: dict) -> int:
    return int(sample["sample_id"])


def connected_image_groups(samples: list[dict]) -> list[list[dict]]:
    """Group samples transitively when they share an image/table source path."""

    ids = [_sample_id(sample) for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("sample IDs must be unique")
    dsu = _DisjointSet(ids)
    owners: dict[str, int] = {}
    for sample in samples:
        sample_id = _sample_id(sample)
        paths = sample["task_instance"].get("images_path", [])
        for path in set(map(str, paths)):
            if path in owners:
                dsu.union(sample_id, owners[path])
            else:
                owners[path] = sample_id
    groups: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[dsu.find(_sample_id(sample))].append(sample)
    return [sorted(group, key=_sample_id) for group in groups.values()]


def split_train_validation(
    samples: list[dict],
    test_ids: set[int],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Reserve every test-connected group, then split remaining whole groups."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    all_ids = {_sample_id(sample) for sample in samples}
    missing = test_ids - all_ids
    if missing:
        raise ValueError(f"test IDs missing from full annotation: {sorted(missing)}")
    groups = connected_image_groups(samples)
    test_groups = [group for group in groups if any(_sample_id(s) in test_ids for s in group)]
    held_out_ids = {_sample_id(sample) for group in test_groups for sample in group}
    pool = [group for group in groups if not any(_sample_id(s) in held_out_ids for s in group)]
    rng = random.Random(seed)
    tie_break = {_sample_id(group[0]): rng.random() for group in pool}
    target_validation = round(sum(map(len, pool)) * validation_fraction)
    validation_groups: list[list[dict]] = []
    train_groups: list[list[dict]] = []
    validation_count = 0
    for group in sorted(
        pool, key=lambda value: (-len(value), tie_break[_sample_id(value[0])])
    ):
        if validation_count < target_validation:
            validation_groups.append(group)
            validation_count += len(group)
        else:
            train_groups.append(group)
    train = sorted((s for group in train_groups for s in group), key=_sample_id)
    validation = sorted((s for group in validation_groups for s in group), key=_sample_id)
    test = sorted((s for s in samples if _sample_id(s) in test_ids), key=_sample_id)
    audit = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "all_samples": len(samples),
        "all_image_connected_groups": len(groups),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "test_connected_samples_excluded_from_training": len(held_out_ids - test_ids),
        "test_connected_excluded_ids": sorted(held_out_ids - test_ids),
    }
    return train, validation, test, audit


def manifest_fingerprint(data: list[dict]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_subset(path: Path, meta_data: dict, data: list[dict], split: str, audit: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta_data": meta_data,
        "selection": {
            "experiment": "004_learned_utility_scorer",
            "split": split,
            "fingerprint": manifest_fingerprint(data),
            "audit": audit,
        },
        "data": data,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

