"""Tests for group-aware experiment 004 splitting."""

from mllm_prune.data.splits import split_train_validation


def _sample(sample_id: int, images: list[str]) -> dict:
    return {"sample_id": sample_id, "task_instance": {"images_path": images}}


def test_test_connected_samples_are_excluded_from_train_and_validation() -> None:
    samples = [
        _sample(0, ["a.png"]),
        _sample(1, ["a.png", "b.png"]),
        _sample(2, ["c.png"]),
        _sample(3, ["d.png"]),
        _sample(4, ["e.png"]),
    ]
    train, validation, test, audit = split_train_validation(
        samples, {0}, validation_fraction=0.5, seed=7
    )
    assert [sample["sample_id"] for sample in test] == [0]
    remaining = {sample["sample_id"] for sample in train + validation}
    assert 0 not in remaining
    assert 1 not in remaining
    assert audit["test_connected_samples_excluded_from_training"] == 1
    assert {sample["sample_id"] for sample in train}.isdisjoint(
        {sample["sample_id"] for sample in validation}
    )

