"""Two-stage aligned prefiltering and conditional-utility planning."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from mllm_prune.algorithms.joint_pruning import (
    CandidateLayout,
    PruningPlan,
    RelevanceScores,
)


@dataclass(frozen=True)
class ConditionalPruningState:
    """Surviving original candidate indices after the aligned prefilter."""

    retention: float
    prefilter_fraction: float
    original_capacity: int
    prefilter_budget: int
    final_budget: int
    text_chunk_indices: tuple[tuple[int, ...], ...]
    vision_chunk_indices: tuple[tuple[int, ...], ...]
    original_chunk_ids: tuple[int, ...]
    partial_chunk_ids: tuple[int, ...]

    @property
    def selected_text_indices(self) -> tuple[int, ...]:
        return tuple(sorted(index for chunk in self.text_chunk_indices for index in chunk))

    @property
    def selected_vision_indices(self) -> tuple[int, ...]:
        return tuple(sorted(index for chunk in self.vision_chunk_indices for index in chunk))

    @property
    def chunk_capacities(self) -> tuple[int, ...]:
        return tuple(len(chunk) for chunk in (*self.text_chunk_indices, *self.vision_chunk_indices))

    def validate(self, layout: CandidateLayout) -> None:
        layout.validate()
        if not 0 < self.retention <= 1:
            raise ValueError("retention must be in (0, 1]")
        if not 0 <= self.prefilter_fraction <= 1:
            raise ValueError("prefilter_fraction must be in [0, 1]")
        total = layout.num_text_tokens + layout.num_vision_tokens
        if self.original_capacity != total:
            raise ValueError("state original capacity does not match layout")
        if not self.final_budget <= self.prefilter_budget <= total:
            raise ValueError("state budgets are inconsistent")
        selected_text = self.selected_text_indices
        selected_vision = self.selected_vision_indices
        if len(selected_text) != len(set(selected_text)) or len(selected_vision) != len(
            set(selected_vision)
        ):
            raise ValueError("state candidate indices cannot overlap")
        if any(index < 0 or index >= layout.num_text_tokens for index in selected_text):
            raise ValueError("state text index is outside the original layout")
        if any(index < 0 or index >= layout.num_vision_tokens for index in selected_vision):
            raise ValueError("state vision index is outside the original layout")
        if len(selected_text) + len(selected_vision) != self.prefilter_budget:
            raise ValueError("state selection does not fill its prefilter budget")
        chunks = (*layout.text_chunks, *layout.vision_chunks)
        expected_ids = tuple(
            index
            for index, chunk in enumerate(chunks)
            if (set(chunk.token_indices) & set(
                selected_text if index < len(layout.text_chunks) else selected_vision
            ))
        )
        if expected_ids != self.original_chunk_ids:
            raise ValueError("state original chunk IDs do not match selected candidates")


def _top_indices(
    indices: tuple[int, ...], scores: torch.Tensor, count: int
) -> tuple[int, ...]:
    ranked = sorted(indices, key=lambda index: (-float(scores[index]), index))
    return tuple(sorted(ranked[:count]))


def build_aligned_prefilter_state(
    layout: CandidateLayout,
    aligned_relevance: RelevanceScores,
    *,
    retention: float,
    prefilter_fraction: float,
) -> ConditionalPruningState:
    """Keep the highest aligned-relevance chunks up to a conservative budget."""

    layout.validate()
    if not 0 < retention <= 1:
        raise ValueError("retention must be in (0, 1]")
    if not 0 <= prefilter_fraction <= 1:
        raise ValueError("prefilter_fraction must be in [0, 1]")
    if len(aligned_relevance.text_tokens) != layout.num_text_tokens:
        raise ValueError("aligned text token scores do not match layout")
    if len(aligned_relevance.vision_tokens) != layout.num_vision_tokens:
        raise ValueError("aligned vision token scores do not match layout")
    if len(aligned_relevance.text_chunks) != len(layout.text_chunks):
        raise ValueError("aligned text chunk scores do not match layout")
    if len(aligned_relevance.vision_chunks) != len(layout.vision_chunks):
        raise ValueError("aligned vision chunk scores do not match layout")

    total = layout.num_text_tokens + layout.num_vision_tokens
    final_budget = total if retention == 1 else math.floor(retention * total)
    prefilter_delete = math.floor(prefilter_fraction * (total - final_budget))
    prefilter_budget = total - prefilter_delete
    chunks = (*layout.text_chunks, *layout.vision_chunks)
    scores = torch.cat(
        (aligned_relevance.text_chunks.float(), aligned_relevance.vision_chunks.float())
    ).cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("aligned chunk scores contain NaN or Inf")
    order = sorted(range(len(chunks)), key=lambda index: (-float(scores[index]), index))
    kept: dict[int, tuple[int, ...]] = {}
    partial: list[int] = []
    remaining = prefilter_budget
    text_count = len(layout.text_chunks)
    for chunk_id in order:
        if not remaining:
            break
        chunk = chunks[chunk_id]
        count = min(len(chunk.token_indices), remaining)
        token_scores = (
            aligned_relevance.text_tokens
            if chunk_id < text_count
            else aligned_relevance.vision_tokens
        )
        kept[chunk_id] = _top_indices(chunk.token_indices, token_scores.cpu(), count)
        if count < len(chunk.token_indices):
            partial.append(chunk_id)
        remaining -= count
    if remaining:
        raise RuntimeError("aligned prefilter failed to fill its exact budget")
    text_chunks = tuple(kept[index] for index in range(text_count) if index in kept)
    vision_chunks = tuple(
        kept[index] for index in range(text_count, len(chunks)) if index in kept
    )
    state = ConditionalPruningState(
        retention=float(retention),
        prefilter_fraction=float(prefilter_fraction),
        original_capacity=total,
        prefilter_budget=prefilter_budget,
        final_budget=final_budget,
        text_chunk_indices=text_chunks,
        vision_chunk_indices=vision_chunks,
        original_chunk_ids=tuple(sorted(kept)),
        partial_chunk_ids=tuple(sorted(partial)),
    )
    state.validate(layout)
    return state


def plan_conditional_utility_pruning(
    layout: CandidateLayout,
    state: ConditionalPruningState,
    predicted_utilities: torch.Tensor,
    aligned_relevance: RelevanceScores,
    *,
    seed: int = 0,
) -> PruningPlan:
    """Fill the final budget by raw predicted utility and aligned boundary scores."""

    state.validate(layout)
    predicted = predicted_utilities.float().flatten().cpu()
    if len(predicted) != len(state.original_chunk_ids):
        raise ValueError("prediction count does not match surviving state chunks")
    if not torch.isfinite(predicted).all():
        raise ValueError("predicted utilities contain NaN or Inf")
    chunks = (*layout.text_chunks, *layout.vision_chunks)
    text_count = len(layout.text_chunks)
    retained_by_id = dict(
        zip(
            state.original_chunk_ids,
            (*state.text_chunk_indices, *state.vision_chunk_indices),
            strict=True,
        )
    )
    prediction_by_id = dict(zip(state.original_chunk_ids, predicted.tolist(), strict=True))
    order = sorted(
        state.original_chunk_ids,
        key=lambda chunk_id: (-prediction_by_id[chunk_id], chunk_id),
    )
    budgets = [0] * len(chunks)
    selected_text: list[int] = []
    selected_vision: list[int] = []
    remaining = state.final_budget
    for chunk_id in order:
        if not remaining:
            break
        available = retained_by_id[chunk_id]
        count = min(len(available), remaining)
        token_scores = (
            aligned_relevance.text_tokens.cpu()
            if chunk_id < text_count
            else aligned_relevance.vision_tokens.cpu()
        )
        selected = _top_indices(available, token_scores, count)
        budgets[chunk_id] = count
        if chunk_id < text_count:
            selected_text.extend(selected)
        else:
            selected_vision.extend(selected)
        remaining -= count
    if remaining:
        raise RuntimeError("conditional utility planner failed to fill exact budget")
    full_scores = [float("-inf")] * len(chunks)
    for chunk_id, value in prediction_by_id.items():
        full_scores[chunk_id] = float(value)
    text_budget = len(selected_text)
    vision_budget = len(selected_vision)
    return PruningPlan(
        method="conditional_utility_joint",
        retention=state.retention,
        seed=seed,
        target_total_budget=state.final_budget,
        total_budget=state.final_budget,
        budget_deviation=0,
        text_budget=text_budget,
        vision_budget=vision_budget,
        selected_text_indices=tuple(sorted(selected_text)),
        selected_vision_indices=tuple(sorted(selected_vision)),
        text_token_scores=tuple(float(value) for value in aligned_relevance.text_tokens.cpu()),
        vision_token_scores=tuple(float(value) for value in aligned_relevance.vision_tokens.cpu()),
        text_chunk_scores=tuple(full_scores[:text_count]),
        vision_chunk_scores=tuple(full_scores[text_count:]),
        text_chunk_budgets=tuple(budgets[:text_count]),
        vision_chunk_budgets=tuple(budgets[text_count:]),
        text_modality_score=max(full_scores[:text_count], default=float("-inf")),
        vision_modality_score=max(full_scores[text_count:], default=float("-inf")),
        text_budget_fraction=text_budget / state.final_budget if state.final_budget else 0.0,
        reallocations=0,
    )
