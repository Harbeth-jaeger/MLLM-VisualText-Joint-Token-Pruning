import torch

from mllm_prune.algorithms.cascaded_utility import (
    build_aligned_prefilter_state,
    plan_conditional_utility_pruning,
)
from mllm_prune.algorithms.joint_pruning import (
    CandidateLayout,
    Chunk,
    RelevanceScores,
)


def _fixture():
    layout = CandidateLayout(
        text_chunks=(Chunk(0, (0, 1, 2)), Chunk(1, (3, 4))),
        vision_chunks=(Chunk(0, (0, 1, 2, 3)), Chunk(1, (4,))),
        num_text_tokens=5,
        num_vision_tokens=5,
    )
    relevance = RelevanceScores(
        text_tokens=torch.tensor([0.1, 0.9, 0.2, 0.3, 0.4]),
        vision_tokens=torch.tensor([0.8, 0.7, 0.6, 0.5, 0.0]),
        text_chunks=torch.tensor([0.9, 0.4]),
        vision_chunks=torch.tensor([0.8, 0.0]),
        text_modality=0.9,
        vision_modality=0.8,
    )
    return layout, relevance


def test_prefilter_budget_and_partial_chunk():
    layout, relevance = _fixture()
    state = build_aligned_prefilter_state(
        layout, relevance, retention=0.5, prefilter_fraction=0.5
    )
    assert state.final_budget == 5
    assert state.prefilter_budget == 8
    assert len(state.selected_text_indices) + len(state.selected_vision_indices) == 8
    assert state.original_chunk_ids == (0, 1, 2)
    assert state.partial_chunk_ids == (1,)
    assert state.text_chunk_indices[1] == (4,)


def test_prefilter_alpha_zero_is_identity():
    layout, relevance = _fixture()
    state = build_aligned_prefilter_state(
        layout, relevance, retention=0.5, prefilter_fraction=0.0
    )
    assert state.prefilter_budget == 10
    assert state.selected_text_indices == tuple(range(5))
    assert state.selected_vision_indices == tuple(range(5))
    assert state.partial_chunk_ids == ()


def test_conditional_planner_uses_raw_utility_and_aligned_boundary():
    layout, relevance = _fixture()
    state = build_aligned_prefilter_state(
        layout, relevance, retention=0.5, prefilter_fraction=0.0
    )
    # Highest utility text chunk has capacity 3; next visual chunk only gets two slots.
    predictions = torch.tensor([3.0, -1.0, 1.0, 0.0])
    plan = plan_conditional_utility_pruning(layout, state, predictions, relevance)
    assert plan.total_budget == 5
    assert plan.selected_text_indices == (0, 1, 2)
    assert plan.selected_vision_indices == (0, 1)
    assert plan.text_chunk_budgets == (3, 0)
    assert plan.vision_chunk_budgets == (2, 0)


def test_prefilter_is_deterministic_on_ties():
    layout, relevance = _fixture()
    tied = RelevanceScores(
        relevance.text_tokens,
        relevance.vision_tokens,
        torch.zeros(2),
        torch.zeros(2),
        0.0,
        0.0,
    )
    first = build_aligned_prefilter_state(
        layout, tied, retention=0.5, prefilter_fraction=1.0
    )
    second = build_aligned_prefilter_state(
        layout, tied, retention=0.5, prefilter_fraction=1.0
    )
    assert first == second
    assert first.original_chunk_ids == (0, 1)
