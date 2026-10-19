"""State-aware features and records for conditional chunk utility."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from mllm_prune.algorithms.cascaded_utility import ConditionalPruningState
from mllm_prune.algorithms.joint_pruning import CandidateLayout, RelevanceScores
from mllm_prune.models.qwen25vl import CandidateSelection


@dataclass(frozen=True)
class ConditionalUtilityFeatures:
    chunk_features: torch.Tensor
    question_feature: torch.Tensor
    modality_ids: torch.Tensor
    structural_features: torch.Tensor
    aligned_features: torch.Tensor
    budget_features: torch.Tensor
    original_chunk_ids: torch.Tensor
    chunk_capacities: tuple[int, ...]

    def validate(self, state: ConditionalPruningState) -> None:
        count = len(state.original_chunk_ids)
        matrices = (
            self.chunk_features,
            self.structural_features,
            self.aligned_features,
            self.budget_features,
        )
        if any(value.ndim != 2 or value.shape[0] != count for value in matrices):
            raise ValueError("conditional feature matrices need one row per state chunk")
        if self.question_feature.ndim != 1:
            raise ValueError("question feature must be rank 1")
        if self.modality_ids.shape != (count,) or self.original_chunk_ids.shape != (count,):
            raise ValueError("conditional feature IDs need one value per state chunk")
        if tuple(int(value) for value in self.original_chunk_ids) != state.original_chunk_ids:
            raise ValueError("conditional feature chunk IDs do not match state")
        if self.chunk_capacities != state.chunk_capacities:
            raise ValueError("conditional feature capacities do not match state")
        for value in (*matrices, self.question_feature):
            if not torch.isfinite(value.float()).all():
                raise ValueError("conditional features contain NaN or Inf")


def _pool(features: torch.Tensor, groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    if not groups:
        return features.new_empty((0, features.shape[-1]), dtype=torch.float32)
    return torch.stack([features[list(group)].float().mean(dim=0) for group in groups])


def _redundancy(features: torch.Tensor) -> torch.Tensor:
    if len(features) <= 1:
        return features.new_zeros((len(features), 2), dtype=torch.float32)
    unit = F.normalize(features.float(), dim=-1)
    similarity = unit @ unit.T
    mask = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    others = similarity[mask].reshape(len(features), len(features) - 1)
    return torch.stack((others.max(dim=1).values, others.mean(dim=1)), dim=1)


def _aligned_statistics(
    scores: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
    top_fraction: float,
) -> torch.Tensor:
    if not 0 < top_fraction <= 1:
        raise ValueError("aligned top fraction must be in (0, 1]")
    rows: list[torch.Tensor] = []
    for group in groups:
        values = scores[list(group)].float()
        count = max(1, math.ceil(len(values) * top_fraction))
        rows.append(
            torch.stack(
                (
                    values.max(),
                    values.mean(),
                    values.std(unbiased=False),
                    torch.topk(values, count).values.mean(),
                )
            )
        )
    return torch.stack(rows) if rows else scores.new_empty((0, 4), dtype=torch.float32)


def build_conditional_utility_features(
    layout: CandidateLayout,
    state: ConditionalPruningState,
    text_features: torch.Tensor,
    vision_features: torch.Tensor,
    question_features: torch.Tensor,
    aligned_relevance: RelevanceScores,
    *,
    aligned_top_fraction: float,
) -> ConditionalUtilityFeatures:
    """Build state-aware Qwen, aligned, redundancy, and budget features."""

    state.validate(layout)
    if text_features.ndim != 2 or len(text_features) != layout.num_text_tokens:
        raise ValueError("text features do not match layout")
    if vision_features.ndim != 2 or len(vision_features) != layout.num_vision_tokens:
        raise ValueError("vision features do not match layout")
    if question_features.ndim != 2 or not len(question_features):
        raise ValueError("question features must be a non-empty matrix")
    text_pooled = _pool(text_features, state.text_chunk_indices)
    vision_pooled = _pool(vision_features, state.vision_chunk_indices)
    chunk_features = torch.cat((text_pooled, vision_pooled), dim=0)
    question = question_features.float().mean(dim=0)
    if chunk_features.shape[1] != question.shape[0]:
        raise ValueError("chunk and question hidden dimensions differ")
    text_redundancy = _redundancy(text_pooled)
    vision_redundancy = _redundancy(vision_pooled)
    redundancy = torch.cat((text_redundancy, vision_redundancy), dim=0)
    text_aligned = _aligned_statistics(
        aligned_relevance.text_tokens, state.text_chunk_indices, aligned_top_fraction
    )
    vision_aligned = _aligned_statistics(
        aligned_relevance.vision_tokens, state.vision_chunk_indices, aligned_top_fraction
    )
    aligned = torch.cat((text_aligned, vision_aligned), dim=0).to(chunk_features.device)
    text_count = len(state.text_chunk_indices)
    vision_count = len(state.vision_chunk_indices)
    total_chunks = max(text_count + vision_count, 1)
    total_tokens = max(state.prefilter_budget, 1)
    text_tokens = len(state.selected_text_indices)
    vision_tokens = len(state.selected_vision_indices)
    structural: list[list[float]] = []
    for modality, groups, token_count in (
        (0, state.text_chunk_indices, text_tokens),
        (1, state.vision_chunk_indices, vision_tokens),
    ):
        for order, group in enumerate(groups):
            structural.append(
                [
                    math.log1p(len(group)),
                    len(group) / max(token_count, 1),
                    order / max(len(groups) - 1, 1),
                    len(groups) / total_chunks,
                    token_count / total_tokens,
                    float(modality),
                ]
            )
    base = torch.tensor(structural, dtype=torch.float32, device=chunk_features.device)
    budget_row = torch.tensor(
        [
            state.retention,
            state.prefilter_fraction,
            state.prefilter_budget / max(state.original_capacity, 1),
            text_tokens / total_tokens,
            vision_tokens / total_tokens,
        ],
        dtype=torch.float32,
        device=chunk_features.device,
    )
    result = ConditionalUtilityFeatures(
        chunk_features=chunk_features,
        question_feature=question,
        modality_ids=torch.tensor(
            [0] * text_count + [1] * vision_count,
            dtype=torch.long,
            device=chunk_features.device,
        ),
        structural_features=torch.cat((base, redundancy), dim=1),
        aligned_features=aligned,
        budget_features=budget_row.unsqueeze(0).expand(total_chunks, -1),
        original_chunk_ids=torch.tensor(
            state.original_chunk_ids, dtype=torch.long, device=chunk_features.device
        ),
        chunk_capacities=state.chunk_capacities,
    )
    result.validate(state)
    return result


def conditional_selections(
    state: ConditionalPruningState,
) -> list[CandidateSelection]:
    """Return S followed by S minus each surviving chunk in record order."""

    base_text = set(state.selected_text_indices)
    base_vision = set(state.selected_vision_indices)
    selections = [
        CandidateSelection(tuple(sorted(base_text)), tuple(sorted(base_vision)))
    ]
    for group in state.text_chunk_indices:
        selections.append(
            CandidateSelection(tuple(sorted(base_text - set(group))), tuple(sorted(base_vision)))
        )
    for group in state.vision_chunk_indices:
        selections.append(
            CandidateSelection(tuple(sorted(base_text)), tuple(sorted(base_vision - set(group))))
        )
    return selections


def conditional_utility_from_nll(
    state_nll: float, deleted_nlls: list[float]
) -> torch.Tensor:
    values = torch.tensor(deleted_nlls, dtype=torch.float32) - float(state_nll)
    if not torch.isfinite(values).all():
        raise ValueError("conditional utility labels contain NaN or Inf")
    return values


def conditional_features_to_record(features: ConditionalUtilityFeatures) -> dict:
    return {
        "chunk_features": features.chunk_features.detach().cpu().to(torch.bfloat16),
        "question_feature": features.question_feature.detach().cpu().to(torch.bfloat16),
        "modality_ids": features.modality_ids.detach().cpu(),
        "structural_features": features.structural_features.detach().cpu(),
        "aligned_features": features.aligned_features.detach().cpu(),
        "budget_features": features.budget_features.detach().cpu(),
        "original_chunk_ids": features.original_chunk_ids.detach().cpu(),
        "chunk_capacities": features.chunk_capacities,
    }
