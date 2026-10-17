"""Structural tests for text-window coverage and dense grid mapping."""

import torch

from mllm_prune.algorithms.aligned_scoring import _map_patch_scores, _text_windows
from mllm_prune.algorithms.joint_pruning import CandidateLayout, Chunk


class FakeEncoded:
    expanded_prompt = "abcdefghij"
    text_candidate_char_spans = tuple((index, index + 1) for index in range(10))
    layout = CandidateLayout(
        text_chunks=(Chunk(0, tuple(range(6))), Chunk(1, tuple(range(6, 10)))),
        vision_chunks=(Chunk(0, (0,)),),
        num_text_tokens=10,
        num_vision_tokens=1,
    )


def test_text_windows_cover_candidates_without_crossing_chunks() -> None:
    texts, coverage = _text_windows(FakeEncoded(), window_tokens=4, stride_tokens=2)
    assert texts == ["abcd", "cdef", "ghij"]
    assert set(index for window in coverage for index in window) == set(range(10))
    assert all(not (min(window) < 6 <= max(window)) for window in coverage)


def test_area_mapping_preserves_constant_and_orientation() -> None:
    constant = _map_patch_scores(torch.full((3, 5), 0.7), 7, 2)
    assert torch.allclose(constant, torch.full((14,), 0.7))
    gradient = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    mapped = _map_patch_scores(gradient, 2, 2).reshape(2, 2)
    assert torch.equal(mapped, gradient)
    assert mapped[0, 0] < mapped[-1, -1]
