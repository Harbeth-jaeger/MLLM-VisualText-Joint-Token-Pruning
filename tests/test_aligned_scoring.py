"""Structural tests for text-window coverage and dense grid mapping."""

import torch

from mllm_prune.algorithms.aligned_scoring import (
    _encode_text_window_scores,
    _map_patch_scores,
    _text_windows,
)
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


def test_text_windows_preserve_whitespace_only_candidate_coverage() -> None:
    class WhitespaceEncoded:
        expanded_prompt = "meaningful "
        text_candidate_char_spans = ((0, 10), (10, 11))
        layout = CandidateLayout(
            text_chunks=(Chunk(0, (0,)), Chunk(1, (1,))),
            vision_chunks=(Chunk(0, (0,)),),
            num_text_tokens=2,
            num_vision_tokens=1,
        )

    texts, coverage = _text_windows(
        WhitespaceEncoded(), window_tokens=16, stride_tokens=8
    )
    assert texts == ["meaningful", ""]
    assert coverage == [(0,), (1,)]


def test_blank_text_window_receives_cosine_lower_bound() -> None:
    class FakeEncoder:
        def encode_text(self, texts):
            assert texts == ["question", "meaningful"]
            return torch.tensor([[1.0, 0.0], [0.5, 0.5]])

    query, scores, _ = _encode_text_window_scores(
        FakeEncoder(), "question", ["meaningful", ""]
    )
    assert torch.equal(query, torch.tensor([1.0, 0.0]))
    assert torch.equal(scores, torch.tensor([0.5, -1.0]))


def test_area_mapping_preserves_constant_and_orientation() -> None:
    constant = _map_patch_scores(torch.full((3, 5), 0.7), 7, 2)
    assert torch.allclose(constant, torch.full((14,), 0.7))
    gradient = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    mapped = _map_patch_scores(gradient, 2, 2).reshape(2, 2)
    assert torch.equal(mapped, gradient)
    assert mapped[0, 0] < mapped[-1, -1]
