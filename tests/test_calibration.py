"""Exact tests for sample-balanced empirical-CDF calibration."""

import torch

from mllm_prune.algorithms.calibration import (
    EmpiricalCDFCalibrator,
    EmpiricalDistribution,
    WeightedScores,
)


def test_empirical_cdf_is_monotone_and_uses_midrank() -> None:
    distribution = EmpiricalDistribution.fit(
        [WeightedScores((0.0, 1.0)), WeightedScores((1.0, 2.0))]
    )
    transformed = distribution.transform(torch.tensor([-1.0, 0.0, 0.5, 1.0, 3.0]))
    assert torch.all(transformed[1:] >= transformed[:-1])
    assert transformed[1].item() == 0.125
    assert transformed[2].item() == 0.25
    assert transformed[3].item() == 0.5
    assert 0 < transformed[0].item() < 1
    assert 0 < transformed[-1].item() < 1


def test_each_sample_has_equal_total_weight() -> None:
    distribution = EmpiricalDistribution.fit(
        [WeightedScores((0.0,)), WeightedScores((1.0,) * 9)]
    )
    transformed = distribution.transform(torch.tensor([0.0, 1.0]))
    assert transformed.tolist() == [0.25, 0.75]


def test_calibrator_roundtrip_and_compatibility(tmp_path) -> None:
    calibrator = EmpiricalCDFCalibrator.fit(
        [WeightedScores((0.1, 0.2))],
        [WeightedScores((0.3, 0.4))],
        {"encoder_identifier": "clip", "window_tokens": 16},
    )
    path = tmp_path / "calibration.json"
    calibrator.save(path)
    loaded = EmpiricalCDFCalibrator.load(path)
    loaded.validate_compatibility({"encoder_identifier": "clip", "window_tokens": 16})
    expected = calibrator.transform(torch.tensor([0.15]), torch.tensor([0.35]))
    actual = loaded.transform(torch.tensor([0.15]), torch.tensor([0.35]))
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
