import torch

from mllm_prune.algorithms.cascaded_utility import build_aligned_prefilter_state
from mllm_prune.algorithms.conditional_utility import (
    build_conditional_utility_features,
    conditional_selections,
    conditional_utility_from_nll,
)
from mllm_prune.algorithms.joint_pruning import CandidateLayout, Chunk, RelevanceScores


def _fixture():
    layout = CandidateLayout(
        text_chunks=(Chunk(0, (0, 1)), Chunk(1, (2,))),
        vision_chunks=(Chunk(0, (0, 1)),),
        num_text_tokens=3,
        num_vision_tokens=2,
    )
    relevance = RelevanceScores(
        text_tokens=torch.tensor([0.1, 0.9, 0.4]),
        vision_tokens=torch.tensor([0.8, 0.2]),
        text_chunks=torch.tensor([0.9, 0.4]),
        vision_chunks=torch.tensor([0.8]),
        text_modality=0.9,
        vision_modality=0.8,
    )
    state = build_aligned_prefilter_state(
        layout, relevance, retention=0.6, prefilter_fraction=0.5
    )
    return layout, relevance, state


def test_conditional_features_follow_state_masks():
    layout, relevance, state = _fixture()
    text = torch.tensor([[1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
    vision = torch.tensor([[0.0, 2.0], [0.0, 4.0]])
    question = torch.tensor([[1.0, 1.0]])
    features = build_conditional_utility_features(
        layout,
        state,
        text,
        vision,
        question,
        relevance,
        aligned_top_fraction=0.5,
    )
    assert features.chunk_features.shape[0] == len(state.original_chunk_ids)
    assert features.structural_features.shape[1] == 8
    assert features.aligned_features.shape[1] == 4
    assert features.budget_features.shape[1] == 5
    assert tuple(features.original_chunk_ids.tolist()) == state.original_chunk_ids


def test_conditional_selections_remove_exactly_one_state_chunk():
    _, _, state = _fixture()
    selections = conditional_selections(state)
    assert len(selections) == 1 + len(state.original_chunk_ids)
    base_count = len(selections[0].text_indices) + len(selections[0].vision_indices)
    for selection, capacity in zip(selections[1:], state.chunk_capacities, strict=True):
        count = len(selection.text_indices) + len(selection.vision_indices)
        assert count == base_count - capacity


def test_conditional_utility_sign():
    labels = conditional_utility_from_nll(1.5, [3.5, 1.0])
    assert torch.equal(labels, torch.tensor([2.0, -0.5]))
