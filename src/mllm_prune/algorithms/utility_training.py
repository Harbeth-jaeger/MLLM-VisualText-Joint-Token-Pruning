"""Losses, pair sampling, and validation metrics for utility ranking."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def stratified_pairs_v2(
    labels: torch.Tensor,
    modality_ids: torch.Tensor,
    *,
    gap_threshold: float,
    max_per_type: int,
    high_gap_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a configurable mixture of largest-gap and random valid pairs."""

    if max_per_type < 1:
        raise ValueError("max_per_type must be positive")
    if not 0 <= high_gap_fraction <= 1:
        raise ValueError("high_gap_fraction must be in [0, 1]")
    labels = labels.float().cpu()
    modality_ids = modality_ids.cpu()
    if labels.ndim != 1 or modality_ids.shape != labels.shape:
        raise ValueError("labels and modality_ids must be matching rank-1 tensors")
    strata: dict[str, list[tuple[int, int, float]]] = {"tt": [], "vv": [], "tv": []}
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            gap = float(labels[left] - labels[right])
            if abs(gap) <= gap_threshold:
                continue
            high, low = (left, right) if gap > 0 else (right, left)
            left_modality = int(modality_ids[left])
            right_modality = int(modality_ids[right])
            key = (
                "tt"
                if left_modality == right_modality == 0
                else "vv"
                if left_modality == right_modality == 1
                else "tv"
            )
            strata[key].append((high, low, abs(gap)))
    rng = random.Random(seed)
    selected: list[tuple[int, int]] = []
    for key in ("tt", "vv", "tv"):
        values = strata[key]
        if len(values) > max_per_type:
            values.sort(key=lambda item: (-item[2], item[0], item[1]))
            fixed_count = min(max_per_type, round(max_per_type * high_gap_fraction))
            fixed = values[:fixed_count]
            tail = values[fixed_count:]
            random_count = max_per_type - fixed_count
            sampled = rng.sample(tail, random_count) if random_count else []
            values = fixed + sampled
        selected.extend((high, low) for high, low, _ in values)
    if not selected:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    return (
        torch.tensor([pair[0] for pair in selected], dtype=torch.long),
        torch.tensor([pair[1] for pair in selected], dtype=torch.long),
    )


def stratified_pairs(
    labels: torch.Tensor,
    modality_ids: torch.Tensor,
    *,
    gap_threshold: float,
    max_per_type: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ordered high/low pairs across TT, VV, and TV strata."""

    labels = labels.float().cpu()
    modality_ids = modality_ids.cpu()
    if labels.ndim != 1 or modality_ids.shape != labels.shape:
        raise ValueError("labels and modality_ids must be matching rank-1 tensors")
    strata: dict[str, list[tuple[int, int, float]]] = {"tt": [], "vv": [], "tv": []}
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            gap = float(labels[left] - labels[right])
            if abs(gap) <= gap_threshold:
                continue
            high, low = (left, right) if gap > 0 else (right, left)
            left_modality = int(modality_ids[left])
            right_modality = int(modality_ids[right])
            key = "tt" if left_modality == right_modality == 0 else (
                "vv" if left_modality == right_modality == 1 else "tv"
            )
            strata[key].append((high, low, abs(gap)))
    rng = random.Random(seed)
    selected: list[tuple[int, int]] = []
    for key in ("tt", "vv", "tv"):
        values = strata[key]
        if len(values) > max_per_type:
            # Retain half high-gap pairs and sample the rest for broad coverage.
            values.sort(key=lambda item: (-item[2], item[0], item[1]))
            fixed = values[: max_per_type // 2]
            tail = values[max_per_type // 2 :]
            sampled = rng.sample(tail, max_per_type - len(fixed))
            values = fixed + sampled
        selected.extend((high, low) for high, low, _ in values)
    if not selected:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    return (
        torch.tensor([pair[0] for pair in selected], dtype=torch.long),
        torch.tensor([pair[1] for pair in selected], dtype=torch.long),
    )


def utility_loss(
    predictions: torch.Tensor,
    normalized_labels: torch.Tensor,
    pair_high: torch.Tensor,
    pair_low: torch.Tensor,
    *,
    huber_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine pairwise softplus ranking with robust scalar regression."""

    if predictions.shape != normalized_labels.shape:
        raise ValueError("predictions and labels must have identical shape")
    regression = F.huber_loss(predictions, normalized_labels)
    if len(pair_high):
        ranking = F.softplus(
            -(predictions[pair_high] - predictions[pair_low])
        ).mean()
    else:
        ranking = predictions.sum() * 0
    total = ranking + huber_weight * regression
    return total, {
        "loss": float(total.detach()),
        "ranking_loss": float(ranking.detach()),
        "huber_loss": float(regression.detach()),
        "pairs": len(pair_high),
    }


def pairwise_accuracy(
    predictions: torch.Tensor, labels: torch.Tensor, *, gap_threshold: float
) -> float:
    correct = total = 0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            gap = float(labels[left] - labels[right])
            if abs(gap) <= gap_threshold:
                continue
            predicted_gap = float(predictions[left] - predictions[right])
            correct += int((predicted_gap > 0) == (gap > 0))
            total += 1
    return correct / total if total else float("nan")

