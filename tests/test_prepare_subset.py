"""Tests for deterministic MMCoQA sample selection."""

from scripts.prepare_mmcoqa_subset import select_samples


def test_select_samples_is_deterministic_and_filters_short_contexts() -> None:
    """Selection should exclude short records and remain reproducible."""

    annotation = {
        "data": [
            {
                "sample_id": index,
                "task_instance": {"context": "word " * (20 + index)},
            }
            for index in range(10)
        ]
    }

    first = select_samples(annotation, num_samples=3, min_context_words=25, seed=42)
    second = select_samples(annotation, num_samples=3, min_context_words=25, seed=42)

    assert [sample["sample_id"] for sample in first] == [5, 9, 7]
    assert first == second
