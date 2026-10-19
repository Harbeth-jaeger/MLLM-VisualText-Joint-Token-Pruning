"""Feature construction, labels, and exact planning for learned chunk utility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from mllm_prune.algorithms.joint_pruning import (
    CandidateLayout,
    Chunk,
    PruningPlan,
    RelevanceScores,
)


@dataclass(frozen=True)
class UtilityFeatures:
    """One fixed-width feature row per text/vision chunk."""

    chunk_features: torch.Tensor
    question_feature: torch.Tensor
    modality_ids: torch.Tensor
    structural_features: torch.Tensor
    chunk_capacities: tuple[int, ...]

    def validate(self, layout: CandidateLayout) -> None:
        chunks = (*layout.text_chunks, *layout.vision_chunks)
        count = len(chunks)
        if self.chunk_features.ndim != 2 or self.chunk_features.shape[0] != count:
            raise ValueError("chunk_features must have one rank-2 row per chunk")
        if self.question_feature.ndim != 1:
            raise ValueError("question_feature must be rank 1")
        if self.question_feature.shape[0] != self.chunk_features.shape[1]:
            raise ValueError("question and chunk hidden dimensions differ")
        if self.modality_ids.shape != (count,):
            raise ValueError("modality_ids must have one value per chunk")
        if self.structural_features.ndim != 2 or self.structural_features.shape[0] != count:
            raise ValueError("structural_features must have one row per chunk")
        if self.chunk_capacities != tuple(len(chunk.token_indices) for chunk in chunks):
            raise ValueError("chunk capacities do not match layout")
        if not torch.isfinite(self.chunk_features.float()).all():
            raise ValueError("chunk features contain NaN or Inf")
        if not torch.isfinite(self.question_feature.float()).all():
            raise ValueError("question feature contains NaN or Inf")
        if not torch.isfinite(self.structural_features.float()).all():
            raise ValueError("structural features contain NaN or Inf")


def _chunk_means(token_features: torch.Tensor, chunks: Sequence[Chunk]) -> torch.Tensor:
    if not chunks:
        return token_features.new_empty((0, token_features.shape[-1]))
    return torch.stack(
        [token_features[list(chunk.token_indices)].float().mean(dim=0) for chunk in chunks]
    )


def _redundancy(features: torch.Tensor) -> torch.Tensor:
    """Return maximum and mean cosine to other chunks in the same modality."""

    count = features.shape[0]
    if count <= 1:
        return features.new_zeros((count, 2), dtype=torch.float32)
    unit = F.normalize(features.float(), dim=-1)
    similarity = unit @ unit.T
    mask = ~torch.eye(count, dtype=torch.bool, device=features.device)
    others = similarity[mask].reshape(count, count - 1)
    return torch.stack((others.max(dim=1).values, others.mean(dim=1)), dim=1)


def build_utility_features(
    layout: CandidateLayout,
    text_features: torch.Tensor,
    vision_features: torch.Tensor,
    question_features: torch.Tensor,
) -> UtilityFeatures:
    """Pool frozen Qwen embeddings into compact, inspectable chunk features."""

    layout.validate()
    if text_features.ndim != 2 or len(text_features) != layout.num_text_tokens:
        raise ValueError("text feature shape does not match layout")
    if vision_features.ndim != 2 or len(vision_features) != layout.num_vision_tokens:
        raise ValueError("vision feature shape does not match layout")
    if question_features.ndim != 2 or not len(question_features):
        raise ValueError("question features must be a non-empty rank-2 tensor")
    hidden = question_features.shape[1]
    if text_features.shape[1] != hidden or vision_features.shape[1] != hidden:
        raise ValueError("text, vision, and question hidden dimensions must match")

    text_pooled = _chunk_means(text_features, layout.text_chunks)
    vision_pooled = _chunk_means(vision_features, layout.vision_chunks)
    chunk_features = torch.cat((text_pooled, vision_pooled), dim=0)
    question = question_features.float().mean(dim=0)
    text_redundancy = _redundancy(text_pooled)
    vision_redundancy = _redundancy(vision_pooled)

    structural: list[list[float]] = []
    all_chunks = (layout.text_chunks, layout.vision_chunks)
    modality_sizes = (max(layout.num_text_tokens, 1), max(layout.num_vision_tokens, 1))
    total_chunks = max(len(layout.text_chunks) + len(layout.vision_chunks), 1)
    for modality, chunks in enumerate(all_chunks):
        source_max = max((chunk.source_id for chunk in chunks), default=0)
        for order, chunk in enumerate(chunks):
            structural.append(
                [
                    math.log1p(len(chunk.token_indices)),
                    len(chunk.token_indices) / modality_sizes[modality],
                    order / max(len(chunks) - 1, 1),
                    chunk.source_id / max(source_max, 1),
                    len(chunks) / total_chunks,
                    modality_sizes[modality] / max(
                        layout.num_text_tokens + layout.num_vision_tokens, 1
                    ),
                ]
            )
    base_structural = torch.tensor(structural, dtype=torch.float32, device=chunk_features.device)
    redundancy = torch.cat((text_redundancy, vision_redundancy), dim=0)
    result = UtilityFeatures(
        chunk_features=chunk_features,
        question_feature=question,
        modality_ids=torch.tensor(
            [0] * len(layout.text_chunks) + [1] * len(layout.vision_chunks),
            dtype=torch.long,
            device=chunk_features.device,
        ),
        structural_features=torch.cat((base_structural, redundancy), dim=1),
        chunk_capacities=tuple(
            len(chunk.token_indices) for chunk in (*layout.text_chunks, *layout.vision_chunks)
        ),
    )
    result.validate(layout)
    return result


def utility_from_nll(full_nll: float, deleted_nlls: Sequence[float]) -> torch.Tensor:
    """Compute signed leave-one-chunk-out utility labels."""

    values = torch.tensor(deleted_nlls, dtype=torch.float32) - float(full_nll)
    if not torch.isfinite(values).all():
        raise ValueError("utility labels contain NaN or Inf")
    return values


def _ranked_chunk_budgets(
    utilities: torch.Tensor,
    capacities: Sequence[int],
    budget: int,
    gamma: float,
) -> tuple[int, ...]:
    """Greedily fill an exact token budget by predicted utility density."""

    if utilities.ndim != 1 or len(utilities) != len(capacities):
        raise ValueError("utilities and capacities must have the same length")
    if not torch.isfinite(utilities).all():
        raise ValueError("utilities contain NaN or Inf")
    if gamma < 0:
        raise ValueError("density gamma must be non-negative")
    capacities = tuple(int(value) for value in capacities)
    if budget < 0 or budget > sum(capacities):
        raise ValueError("budget is outside total chunk capacity")
    positive = utilities.clamp_min(0)
    density = positive / torch.tensor(capacities, dtype=torch.float32).pow(gamma)
    if not bool(torch.any(density > 0)):
        density = utilities.float()
    order = sorted(
        range(len(capacities)),
        key=lambda index: (-float(density[index]), -float(utilities[index]), index),
    )
    allocations = [0] * len(capacities)
    remaining = budget
    for index in order:
        add = min(capacities[index], remaining)
        allocations[index] = add
        remaining -= add
        if not remaining:
            break
    if remaining:
        raise RuntimeError("utility planner failed to fill exact budget")
    return tuple(allocations)


def _select_tokens(
    chunks: Sequence[Chunk], budgets: Sequence[int], token_scores: torch.Tensor
) -> tuple[int, ...]:
    selected: list[int] = []
    for chunk, budget in zip(chunks, budgets, strict=True):
        ranked = sorted(
            chunk.token_indices,
            key=lambda index: (-float(token_scores[index]), index),
        )
        selected.extend(ranked[:budget])
    return tuple(sorted(selected))


def plan_utility_pruning(
    layout: CandidateLayout,
    predicted_utilities: torch.Tensor,
    token_relevance: RelevanceScores,
    *,
    retention: float,
    density_gamma: float = 1.0,
    seed: int = 0,
) -> PruningPlan:
    """Plan strict joint pruning from cross-modal chunk utility predictions."""

    layout.validate()
    if not 0 < retention <= 1:
        raise ValueError("retention must be in (0, 1]")
    chunks = (*layout.text_chunks, *layout.vision_chunks)
    predicted_utilities = predicted_utilities.float().flatten().cpu()
    if len(predicted_utilities) != len(chunks):
        raise ValueError("predicted utility count does not match chunks")
    total_capacity = layout.num_text_tokens + layout.num_vision_tokens
    budget = total_capacity if retention == 1 else math.floor(retention * total_capacity)
    budgets = _ranked_chunk_budgets(
        predicted_utilities,
        [len(chunk.token_indices) for chunk in chunks],
        budget,
        density_gamma,
    )
    text_count = len(layout.text_chunks)
    text_budgets = budgets[:text_count]
    vision_budgets = budgets[text_count:]
    selected_text = _select_tokens(
        layout.text_chunks, text_budgets, token_relevance.text_tokens.cpu()
    )
    selected_vision = _select_tokens(
        layout.vision_chunks, vision_budgets, token_relevance.vision_tokens.cpu()
    )
    text_budget = sum(text_budgets)
    vision_budget = sum(vision_budgets)
    if text_budget + vision_budget != budget:
        raise RuntimeError("learned utility plan violates the strict budget")
    text_utilities = predicted_utilities[:text_count]
    vision_utilities = predicted_utilities[text_count:]
    return PruningPlan(
        method="learned_utility_joint",
        retention=retention,
        seed=seed,
        target_total_budget=budget,
        total_budget=budget,
        budget_deviation=0,
        text_budget=text_budget,
        vision_budget=vision_budget,
        selected_text_indices=selected_text,
        selected_vision_indices=selected_vision,
        text_token_scores=tuple(float(value) for value in token_relevance.text_tokens.cpu()),
        vision_token_scores=tuple(float(value) for value in token_relevance.vision_tokens.cpu()),
        text_chunk_scores=tuple(float(value) for value in text_utilities),
        vision_chunk_scores=tuple(float(value) for value in vision_utilities),
        text_chunk_budgets=text_budgets,
        vision_chunk_budgets=vision_budgets,
        text_modality_score=float(text_utilities.max()) if len(text_utilities) else float("-inf"),
        vision_modality_score=(
            float(vision_utilities.max()) if len(vision_utilities) else float("-inf")
        ),
        text_budget_fraction=text_budget / budget if budget else 0.0,
        reallocations=0,
    )


def features_to_record(features: UtilityFeatures) -> dict:
    """Detach utility features into a compact CPU training record."""

    return {
        "chunk_features": features.chunk_features.detach().to("cpu", torch.bfloat16),
        "question_feature": features.question_feature.detach().to("cpu", torch.bfloat16),
        "modality_ids": features.modality_ids.detach().cpu(),
        "structural_features": features.structural_features.detach().cpu(),
        "chunk_capacities": features.chunk_capacities,
    }

