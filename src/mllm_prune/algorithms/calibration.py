"""Weighted empirical-CDF calibration for cross-modal relevance scores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch


FORMAT_VERSION = 1


@dataclass(frozen=True)
class WeightedScores:
    """One sample's scores; every sample has equal total calibration weight."""

    values: tuple[float, ...]


@dataclass(frozen=True)
class EmpiricalDistribution:
    """Sorted support and mid-rank CDF values for one modality."""

    support: tuple[float, ...]
    mid_cdf: tuple[float, ...]
    upper_cdf: tuple[float, ...]
    sample_count: int
    candidate_count: int

    @classmethod
    def fit(cls, samples: Iterable[WeightedScores]) -> "EmpiricalDistribution":
        samples = [sample for sample in samples if sample.values]
        if not samples:
            raise ValueError("cannot fit an empirical distribution without scores")
        weighted: list[tuple[float, float]] = []
        sample_weight = 1.0 / len(samples)
        for sample in samples:
            if not all(torch.isfinite(torch.tensor(sample.values))):
                raise ValueError("calibration scores contain NaN or Inf")
            token_weight = sample_weight / len(sample.values)
            weighted.extend((float(value), token_weight) for value in sample.values)
        weighted.sort(key=lambda pair: pair[0])
        support: list[float] = []
        mid_cdf: list[float] = []
        upper_cdf: list[float] = []
        cumulative = 0.0
        cursor = 0
        while cursor < len(weighted):
            value = weighted[cursor][0]
            end = cursor
            tied_weight = 0.0
            while end < len(weighted) and weighted[end][0] == value:
                tied_weight += weighted[end][1]
                end += 1
            support.append(value)
            mid_cdf.append(cumulative + 0.5 * tied_weight)
            cumulative += tied_weight
            upper_cdf.append(cumulative)
            cursor = end
        epsilon = 0.5 / max(2, len(weighted))
        mid_cdf = [min(1.0 - epsilon, max(epsilon, value)) for value in mid_cdf]
        upper_cdf = [min(1.0 - epsilon, max(epsilon, value)) for value in upper_cdf]
        return cls(
            tuple(support), tuple(mid_cdf), tuple(upper_cdf), len(samples), len(weighted)
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the frozen monotone step CDF without changing device."""

        flat = values.float().flatten()
        if not torch.isfinite(flat).all():
            raise ValueError("scores contain NaN or Inf")
        support = torch.tensor(self.support, dtype=torch.float32, device=flat.device)
        mid_cdf = torch.tensor(self.mid_cdf, dtype=torch.float32, device=flat.device)
        upper_cdf = torch.tensor(self.upper_cdf, dtype=torch.float32, device=flat.device)
        indices = torch.searchsorted(support, flat, right=True) - 1
        indices = indices.clamp(0, len(support) - 1)
        result = upper_cdf[indices]
        exact = flat == support[indices]
        result = torch.where(exact, mid_cdf[indices], result)
        result = torch.where(flat < support[0], mid_cdf[0], result)
        result = torch.where(flat > support[-1], upper_cdf[-1], result)
        return result.reshape(values.shape)

    def to_dict(self) -> dict:
        return {
            "support": list(self.support),
            "mid_cdf": list(self.mid_cdf),
            "upper_cdf": list(self.upper_cdf),
            "sample_count": self.sample_count,
            "candidate_count": self.candidate_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmpiricalDistribution":
        return cls(
            tuple(float(value) for value in data["support"]),
            tuple(float(value) for value in data["mid_cdf"]),
            tuple(float(value) for value in data["upper_cdf"]),
            int(data["sample_count"]),
            int(data["candidate_count"]),
        )


@dataclass(frozen=True)
class EmpiricalCDFCalibrator:
    """Frozen text/vision score mappings plus reproducibility metadata."""

    text: EmpiricalDistribution
    vision: EmpiricalDistribution
    metadata: dict

    @classmethod
    def fit(
        cls,
        text_scores: Iterable[WeightedScores],
        vision_scores: Iterable[WeightedScores],
        metadata: dict,
    ) -> "EmpiricalCDFCalibrator":
        return cls(
            EmpiricalDistribution.fit(text_scores),
            EmpiricalDistribution.fit(vision_scores),
            {"format_version": FORMAT_VERSION, **metadata},
        )

    def transform(
        self, text_scores: torch.Tensor, vision_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text.transform(text_scores), self.vision.transform(vision_scores)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "text": self.text.to_dict(),
            "vision": self.vision.to_dict(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EmpiricalCDFCalibrator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload["metadata"].get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported calibration artifact version: {version}")
        return cls(
            EmpiricalDistribution.from_dict(payload["text"]),
            EmpiricalDistribution.from_dict(payload["vision"]),
            payload["metadata"],
        )

    def validate_compatibility(self, expected: dict) -> None:
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"calibration artifact is incompatible: {mismatches}")


def manifest_fingerprint(sample_ids: Iterable[int]) -> str:
    """Return a stable fingerprint for the ordered calibration sample IDs."""

    encoded = json.dumps([int(value) for value in sample_ids], separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
