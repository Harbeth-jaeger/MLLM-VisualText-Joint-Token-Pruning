"""Tests for learned-utility feature, loss, and strict planning logic."""

import pytest
import torch

from mllm_prune.algorithms.joint_pruning import (
    CandidateLayout,
    Chunk,
    build_relevance_from_token_scores,
)
from mllm_prune.algorithms.utility import (
    build_utility_features,
    plan_utility_pruning,
    utility_from_nll,
)
from mllm_prune.algorithms.utility_training import stratified_pairs, utility_loss
from mllm_prune.models.utility_scorer import JointUtilityScorer, UtilityScorerConfig


def _layout() -> CandidateLayout:
    return CandidateLayout(
        text_chunks=(Chunk(0, (0, 1), 0), Chunk(1, (2,), 0)),
        vision_chunks=(Chunk(0, (0, 1), 0), Chunk(1, (2, 3), 0)),
        num_text_tokens=3,
        num_vision_tokens=4,
    )


def test_build_features_has_one_fixed_width_row_per_chunk() -> None:
    layout = _layout()
    text = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    vision = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    question = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 1.0, 0.0]])
    features = build_utility_features(layout, text, vision, question)
    assert features.chunk_features.shape == (4, 4)
    assert features.question_feature.tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert features.modality_ids.tolist() == [0, 0, 1, 1]
    assert features.structural_features.shape == (4, 8)
    assert features.chunk_capacities == (2, 1, 2, 2)


def test_scorer_maps_variable_chunk_count_to_one_scalar_each() -> None:
    model = JointUtilityScorer(
        UtilityScorerConfig(input_dim=4, structural_dim=8, projection_dim=3, hidden_dim=8)
    )
    output = model(
        torch.randn(5, 4), torch.randn(4), torch.tensor([0, 0, 1, 1, 1]), torch.randn(5, 8)
    )
    assert output.shape == (5,)


def test_utility_sign_is_deleted_minus_full() -> None:
    values = utility_from_nll(0.5, [1.5, 0.4])
    assert values.tolist() == pytest.approx([1.0, -0.1])


@pytest.mark.parametrize("retention", [0.25, 0.5, 0.75, 1.0])
def test_utility_plan_fills_strict_budget(retention: float) -> None:
    layout = _layout()
    relevance = build_relevance_from_token_scores(
        layout,
        torch.tensor([0.9, 0.8, 0.1]),
        torch.tensor([0.7, 0.6, 0.5, 0.4]),
        token_top_fraction=0.5,
        chunk_top_fraction=0.5,
    )
    plan = plan_utility_pruning(
        layout,
        torch.tensor([0.2, -0.1, 1.0, 0.1]),
        relevance,
        retention=retention,
    )
    expected = 7 if retention == 1 else int(7 * retention)
    assert plan.total_budget == expected
    assert len(plan.selected_text_indices) + len(plan.selected_vision_indices) == expected
    assert sum(plan.text_chunk_budgets) + sum(plan.vision_chunk_budgets) == expected


def test_pair_sampling_covers_cross_modal_order() -> None:
    labels = torch.tensor([2.0, 0.0, 1.0, -1.0])
    modalities = torch.tensor([0, 0, 1, 1])
    high, low = stratified_pairs(
        labels, modalities, gap_threshold=0.01, max_per_type=10, seed=3
    )
    pairs = set(zip(high.tolist(), low.tolist()))
    assert (0, 2) in pairs
    assert (2, 1) in pairs
    predictions = torch.tensor([2.0, 0.0, 1.0, -1.0], requires_grad=True)
    loss, metrics = utility_loss(
        predictions, labels, high, low, huber_weight=0.25
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["pairs"] == len(pairs)

