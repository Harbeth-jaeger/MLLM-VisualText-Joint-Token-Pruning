"""Unit tests for exact joint budgeting and deterministic selection."""

import pytest
import torch

from mllm_prune.algorithms.joint_pruning import (
    CandidateLayout,
    Chunk,
    JointPruningConfig,
    allocate_capped_integer_budget,
    build_relevance_from_token_scores,
    plan_joint_pruning,
)


def chunks(sizes: list[int]) -> tuple[Chunk, ...]:
    result = []
    offset = 0
    for chunk_id, size in enumerate(sizes):
        result.append(Chunk(chunk_id, tuple(range(offset, offset + size))))
        offset += size
    return tuple(result)


@pytest.mark.parametrize("capacities", ([1], [1, 1], [1, 3, 2], [0, 2, 5]))
def test_capped_budget_is_exact_for_every_feasible_budget(capacities: list[int]) -> None:
    values = torch.arange(len(capacities), dtype=torch.float32)
    for budget in range(sum(capacities) + 1):
        allocation = allocate_capped_integer_budget(values, capacities, budget, 0.3)
        assert sum(allocation) == budget
        assert all(0 <= value <= cap for value, cap in zip(allocation, capacities, strict=True))


@pytest.mark.parametrize("method", ["baseline", "visual_only", "fixed_joint", "random_joint", "proposed_joint"])
@pytest.mark.parametrize("retention", [0.25, 0.5, 0.75, 1.0])
def test_plan_meets_budget_and_preserves_unique_indices(method: str, retention: float) -> None:
    layout = CandidateLayout(chunks([2, 3]), chunks([4, 2]), 5, 6)
    text = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]])
    vision = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4], [0.4, 0.6]])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    expected = 11 if method == "baseline" else int(retention * 11)
    plan = plan_joint_pruning(layout, text, vision, query, JointPruningConfig(retention=retention, method=method, seed=7))
    actual_expected = max(expected, layout.num_text_tokens) if method == "visual_only" else expected
    assert plan.target_total_budget == expected
    assert plan.total_budget == actual_expected
    assert plan.budget_deviation == actual_expected - expected
    assert plan.text_budget + plan.vision_budget == actual_expected
    assert len(plan.selected_text_indices) == plan.text_budget
    assert len(plan.selected_vision_indices) == plan.vision_budget
    assert len(set(plan.selected_text_indices)) == plan.text_budget
    assert len(set(plan.selected_vision_indices)) == plan.vision_budget
    assert tuple(sorted(plan.selected_text_indices)) == plan.selected_text_indices
    assert tuple(sorted(plan.selected_vision_indices)) == plan.selected_vision_indices


def test_random_joint_is_reproducible() -> None:
    layout = CandidateLayout(chunks([4]), chunks([4]), 4, 4)
    embeddings = torch.eye(4)
    query = torch.ones(1, 4)
    config = JointPruningConfig(retention=0.5, method="random_joint", seed=19)
    first = plan_joint_pruning(layout, embeddings, embeddings, query, config)
    second = plan_joint_pruning(layout, embeddings, embeddings, query, config)
    assert first == second


def test_known_similarity_selects_relevant_tokens() -> None:
    layout = CandidateLayout(chunks([2]), chunks([2]), 2, 2)
    text = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    vision = torch.tensor([[0.8, 0.2], [-0.8, -0.2]])
    query = torch.tensor([[1.0, 0.0]])
    plan = plan_joint_pruning(
        layout,
        text,
        vision,
        query,
        JointPruningConfig(retention=0.5, method="fixed_joint", fixed_text_fraction=0.5),
    )
    assert plan.selected_text_indices == (0,)
    assert plan.selected_vision_indices == (0,)


def test_build_relevance_from_precomputed_scores() -> None:
    layout = CandidateLayout(chunks([2, 1]), chunks([1, 2]), 3, 3)
    relevance = build_relevance_from_token_scores(
        layout,
        torch.tensor([0.1, 0.9, 0.4]),
        torch.tensor([0.7, 0.2, 0.3]),
        token_top_fraction=0.5,
        chunk_top_fraction=0.5,
    )
    assert relevance.text_chunks.tolist() == pytest.approx([0.9, 0.4])
    assert relevance.vision_chunks.tolist() == pytest.approx([0.7, 0.3])
    assert relevance.text_modality == pytest.approx(0.9)
    assert relevance.vision_modality == pytest.approx(0.7)


def test_build_relevance_rejects_nonfinite_scores() -> None:
    layout = CandidateLayout(chunks([1]), chunks([1]), 1, 1)
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_relevance_from_token_scores(
            layout,
            torch.tensor([float("nan")]),
            torch.tensor([0.2]),
            token_top_fraction=1.0,
            chunk_top_fraction=1.0,
        )

