"""Chunk-based, query-aware joint pruning independent of any MLLM backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Sequence

import torch
import torch.nn.functional as F


Method = Literal[
    "baseline",
    "visual_only",
    "fixed_joint",
    "random_joint",
    "proposed_joint",
    "raw_qwen_joint",
    "aligned_uncalibrated",
    "aligned_calibrated",
]


@dataclass(frozen=True)
class Chunk:
    """Indices of candidate tokens that form one structural unit."""

    chunk_id: int
    token_indices: tuple[int, ...]
    source_id: int = 0


@dataclass(frozen=True)
class CandidateLayout:
    """Model-independent candidate grouping for both modalities."""

    text_chunks: tuple[Chunk, ...]
    vision_chunks: tuple[Chunk, ...]
    num_text_tokens: int
    num_vision_tokens: int

    def validate(self) -> None:
        """Reject missing, duplicated, or out-of-range candidate indices."""

        for name, chunks, size in (
            ("text", self.text_chunks, self.num_text_tokens),
            ("vision", self.vision_chunks, self.num_vision_tokens),
        ):
            flattened = [index for chunk in chunks for index in chunk.token_indices]
            if sorted(flattened) != list(range(size)):
                raise ValueError(
                    f"{name} chunks must partition 0..{size - 1}; got {flattened}"
                )
            if any(not chunk.token_indices for chunk in chunks):
                raise ValueError(f"{name} chunks cannot be empty")


@dataclass(frozen=True)
class JointPruningConfig:
    """Hyperparameters and experimental condition for one pruning plan."""

    retention: float
    method: Method = "proposed_joint"
    modality_temperature: float = 0.1
    chunk_temperature: float = 0.1
    token_top_fraction: float = 0.2
    chunk_top_fraction: float = 0.2
    min_text_fraction: float = 0.05
    min_vision_fraction: float = 0.10
    fixed_text_fraction: float = 0.5
    seed: int = 0

    def validate(self) -> None:
        if not 0 < self.retention <= 1:
            raise ValueError("retention must be in (0, 1]")
        if self.modality_temperature <= 0 or self.chunk_temperature <= 0:
            raise ValueError("temperatures must be positive")
        for name, value in (
            ("token_top_fraction", self.token_top_fraction),
            ("chunk_top_fraction", self.chunk_top_fraction),
            ("min_text_fraction", self.min_text_fraction),
            ("min_vision_fraction", self.min_vision_fraction),
            ("fixed_text_fraction", self.fixed_text_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PruningPlan:
    """Selections and diagnostics needed to audit and reproduce pruning."""

    method: str
    retention: float
    seed: int
    target_total_budget: int
    total_budget: int
    budget_deviation: int
    text_budget: int
    vision_budget: int
    selected_text_indices: tuple[int, ...]
    selected_vision_indices: tuple[int, ...]
    text_token_scores: tuple[float, ...]
    vision_token_scores: tuple[float, ...]
    text_chunk_scores: tuple[float, ...]
    vision_chunk_scores: tuple[float, ...]
    text_chunk_budgets: tuple[int, ...]
    vision_chunk_budgets: tuple[int, ...]
    text_modality_score: float
    vision_modality_score: float
    text_budget_fraction: float
    reallocations: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RelevanceScores:
    """Query relevance computed once and shared by all sample conditions."""

    text_tokens: torch.Tensor
    vision_tokens: torch.Tensor
    text_chunks: torch.Tensor
    vision_chunks: torch.Tensor
    text_modality: float
    vision_modality: float


def cosine_max_scores(candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Return each candidate's maximum cosine similarity to a query token."""

    if candidates.ndim != 2 or query.ndim != 2:
        raise ValueError("candidate and query embeddings must both be rank 2")
    if candidates.shape[1] != query.shape[1]:
        raise ValueError("candidate and query hidden dimensions must match")
    if not len(candidates):
        return torch.empty(0, dtype=torch.float32, device=candidates.device)
    if not len(query):
        raise ValueError("at least one query token is required")
    candidate_unit = F.normalize(candidates.float(), dim=-1)
    query_unit = F.normalize(query.float(), dim=-1)
    return (candidate_unit @ query_unit.T).amax(dim=-1)


def _top_mean(values: torch.Tensor, fraction: float) -> float:
    if not len(values):
        return float("-inf")
    count = max(1, math.ceil(fraction * len(values)))
    return float(torch.topk(values.float(), count).values.mean().item())


def score_chunks(
    token_scores: torch.Tensor, chunks: Sequence[Chunk], top_fraction: float
) -> torch.Tensor:
    """Aggregate token scores into chunk values using a top-fraction mean."""

    values = []
    for chunk in chunks:
        indices = torch.tensor(chunk.token_indices, device=token_scores.device)
        values.append(_top_mean(token_scores[indices], top_fraction))
    return torch.tensor(values, dtype=torch.float32, device=token_scores.device)


def compute_relevance_scores(
    layout: CandidateLayout,
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    *,
    token_top_fraction: float,
    chunk_top_fraction: float,
) -> RelevanceScores:
    """Compute reusable token, chunk, and modality query relevance."""

    text_scores = cosine_max_scores(text_embeddings, query_embeddings)
    vision_scores = cosine_max_scores(vision_embeddings, query_embeddings)
    text_chunks = score_chunks(text_scores, layout.text_chunks, token_top_fraction)
    vision_chunks = score_chunks(vision_scores, layout.vision_chunks, token_top_fraction)
    return RelevanceScores(
        text_tokens=text_scores,
        vision_tokens=vision_scores,
        text_chunks=text_chunks,
        vision_chunks=vision_chunks,
        text_modality=_top_mean(text_chunks, chunk_top_fraction),
        vision_modality=_top_mean(vision_chunks, chunk_top_fraction),
    )


def build_relevance_from_token_scores(
    layout: CandidateLayout,
    text_token_scores: torch.Tensor,
    vision_token_scores: torch.Tensor,
    *,
    token_top_fraction: float,
    chunk_top_fraction: float,
) -> RelevanceScores:
    """Aggregate precomputed token scores for the shared pruning planner."""

    layout.validate()
    text_token_scores = text_token_scores.float().flatten()
    vision_token_scores = vision_token_scores.float().flatten()
    if len(text_token_scores) != layout.num_text_tokens:
        raise ValueError("text score count does not match layout")
    if len(vision_token_scores) != layout.num_vision_tokens:
        raise ValueError("vision score count does not match layout")
    if not torch.isfinite(text_token_scores).all():
        raise ValueError("text scores contain NaN or Inf")
    if not torch.isfinite(vision_token_scores).all():
        raise ValueError("vision scores contain NaN or Inf")
    text_chunks = score_chunks(text_token_scores, layout.text_chunks, token_top_fraction)
    vision_chunks = score_chunks(vision_token_scores, layout.vision_chunks, token_top_fraction)
    return RelevanceScores(
        text_tokens=text_token_scores,
        vision_tokens=vision_token_scores,
        text_chunks=text_chunks,
        vision_chunks=vision_chunks,
        text_modality=_top_mean(text_chunks, chunk_top_fraction),
        vision_modality=_top_mean(vision_chunks, chunk_top_fraction),
    )


def _capacity_split(total: int, desired_text: int, text_cap: int, vision_cap: int) -> tuple[int, int, int]:
    """Clamp a two-modality integer split and reallocate overflow exactly."""

    if not 0 <= total <= text_cap + vision_cap:
        raise ValueError("total budget exceeds joint capacity")
    text = min(max(desired_text, 0), text_cap, total)
    vision = min(total - text, vision_cap)
    reallocations = int(text != desired_text)
    missing = total - text - vision
    if missing:
        add_text = min(missing, text_cap - text)
        text += add_text
        missing -= add_text
        reallocations += int(add_text > 0)
    if missing:
        add_vision = min(missing, vision_cap - vision)
        vision += add_vision
        missing -= add_vision
        reallocations += int(add_vision > 0)
    if missing or text + vision != total:
        raise RuntimeError("failed to allocate the exact joint budget")
    return text, vision, reallocations


def allocate_capped_integer_budget(
    values: torch.Tensor, capacities: Sequence[int], budget: int, temperature: float
) -> tuple[int, ...]:
    """Allocate an exact softmax budget with integer capacities and stable ties."""

    capacities = tuple(int(value) for value in capacities)
    if budget < 0 or budget > sum(capacities):
        raise ValueError("chunk budget is outside total capacity")
    if not capacities:
        if budget:
            raise ValueError("cannot assign budget without chunks")
        return ()
    allocation = [0] * len(capacities)
    remaining = budget
    active = {index for index, capacity in enumerate(capacities) if capacity > 0}
    while remaining and active:
        ordered = sorted(active)
        logits = values[ordered].float() / temperature
        weights = torch.softmax(logits, dim=0)
        quotas = weights * remaining
        floors = torch.floor(quotas).to(torch.int64).tolist()
        progress = 0
        for local, index in enumerate(ordered):
            add = min(floors[local], capacities[index] - allocation[index])
            allocation[index] += add
            remaining -= add
            progress += add
        active = {i for i in active if allocation[i] < capacities[i]}
        if not remaining:
            break
        # Largest remainder with chunk index as the deterministic tie breaker.
        remainder_order = sorted(
            range(len(ordered)),
            key=lambda local: (-(float(quotas[local]) - floors[local]), ordered[local]),
        )
        for local in remainder_order:
            index = ordered[local]
            if index in active and remaining:
                allocation[index] += 1
                remaining -= 1
                progress += 1
                if allocation[index] == capacities[index]:
                    active.discard(index)
        if progress == 0:
            raise RuntimeError("capped allocation made no progress")
    if remaining or sum(allocation) != budget:
        raise RuntimeError("capped allocation did not meet the exact budget")
    return tuple(allocation)


def _select_by_chunks(
    scores: torch.Tensor, chunks: Sequence[Chunk], budgets: Sequence[int]
) -> tuple[int, ...]:
    selected: list[int] = []
    for chunk, budget in zip(chunks, budgets, strict=True):
        ranked = sorted(chunk.token_indices, key=lambda index: (-float(scores[index]), index))
        selected.extend(ranked[:budget])
    return tuple(sorted(selected))


def _random_scores(size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(size, generator=generator, dtype=torch.float32).to(device)


def plan_joint_pruning(
    layout: CandidateLayout,
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    config: JointPruningConfig,
    relevance: RelevanceScores | None = None,
) -> PruningPlan:
    """Score, budget, and select text/vision candidates for one sample."""

    layout.validate()
    config.validate()
    if text_embeddings.shape[0] != layout.num_text_tokens:
        raise ValueError("text embedding count does not match layout")
    if vision_embeddings.shape[0] != layout.num_vision_tokens:
        raise ValueError("vision embedding count does not match layout")

    relevance = relevance or compute_relevance_scores(
        layout,
        text_embeddings,
        vision_embeddings,
        query_embeddings,
        token_top_fraction=config.token_top_fraction,
        chunk_top_fraction=config.chunk_top_fraction,
    )
    text_scores = relevance.text_tokens
    vision_scores = relevance.vision_tokens
    text_chunk_scores = relevance.text_chunks
    vision_chunk_scores = relevance.vision_chunks
    text_modality = relevance.text_modality
    vision_modality = relevance.vision_modality

    capacity = layout.num_text_tokens + layout.num_vision_tokens
    target_total_budget = capacity if config.method == "baseline" else math.floor(config.retention * capacity)
    total_budget = target_total_budget
    if config.method == "baseline":
        desired_text = layout.num_text_tokens
    elif config.method == "visual_only":
        # A visual-only control cannot remove text. When text alone exceeds the
        # target, expose the unavoidable budget deviation instead of silently
        # changing the method or aborting a long run midway.
        total_budget = max(total_budget, layout.num_text_tokens)
        desired_text = layout.num_text_tokens
    elif config.method == "fixed_joint":
        desired_text = round(total_budget * config.fixed_text_fraction)
    else:
        if not layout.num_text_tokens:
            alpha_text = 0.0
        elif not layout.num_vision_tokens:
            alpha_text = 1.0
        else:
            modality_logits = torch.tensor(
                [text_modality, vision_modality], dtype=torch.float32
            ) / config.modality_temperature
            alpha_text = float(torch.softmax(modality_logits, dim=0)[0].item())
        min_text = min(layout.num_text_tokens, round(config.min_text_fraction * total_budget))
        min_vision = min(layout.num_vision_tokens, round(config.min_vision_fraction * total_budget))
        if min_text + min_vision > total_budget:
            scale = total_budget / (min_text + min_vision)
            min_text = math.floor(min_text * scale)
            min_vision = min(total_budget - min_text, math.floor(min_vision * scale))
        desired_text = min_text + round(alpha_text * (total_budget - min_text - min_vision))

    text_budget, vision_budget, reallocations = _capacity_split(
        total_budget, desired_text, layout.num_text_tokens, layout.num_vision_tokens
    )

    selection_text_scores = text_scores
    selection_vision_scores = vision_scores
    allocation_text_scores = text_chunk_scores
    allocation_vision_scores = vision_chunk_scores
    if config.method == "random_joint":
        selection_text_scores = _random_scores(len(text_scores), config.seed, text_scores.device)
        selection_vision_scores = _random_scores(
            len(vision_scores), config.seed + 1_000_003, vision_scores.device
        )
        allocation_text_scores = score_chunks(selection_text_scores, layout.text_chunks, 1.0)
        allocation_vision_scores = score_chunks(selection_vision_scores, layout.vision_chunks, 1.0)

    text_chunk_budgets = allocate_capped_integer_budget(
        allocation_text_scores,
        [len(chunk.token_indices) for chunk in layout.text_chunks],
        text_budget,
        config.chunk_temperature,
    )
    vision_chunk_budgets = allocate_capped_integer_budget(
        allocation_vision_scores,
        [len(chunk.token_indices) for chunk in layout.vision_chunks],
        vision_budget,
        config.chunk_temperature,
    )
    selected_text = _select_by_chunks(selection_text_scores, layout.text_chunks, text_chunk_budgets)
    selected_vision = _select_by_chunks(
        selection_vision_scores, layout.vision_chunks, vision_chunk_budgets
    )
    if len(selected_text) + len(selected_vision) != total_budget:
        raise RuntimeError("selection does not match the requested total budget")

    return PruningPlan(
        method=config.method,
        retention=config.retention,
        seed=config.seed,
        target_total_budget=target_total_budget,
        total_budget=total_budget,
        budget_deviation=total_budget - target_total_budget,
        text_budget=text_budget,
        vision_budget=vision_budget,
        selected_text_indices=selected_text,
        selected_vision_indices=selected_vision,
        text_token_scores=tuple(float(value) for value in text_scores.cpu()),
        vision_token_scores=tuple(float(value) for value in vision_scores.cpu()),
        text_chunk_scores=tuple(float(value) for value in text_chunk_scores.cpu()),
        vision_chunk_scores=tuple(float(value) for value in vision_chunk_scores.cpu()),
        text_chunk_budgets=text_chunk_budgets,
        vision_chunk_budgets=vision_chunk_budgets,
        text_modality_score=text_modality,
        vision_modality_score=vision_modality,
        text_budget_fraction=text_budget / total_budget if total_budget else 0.0,
        reallocations=reallocations,
    )
