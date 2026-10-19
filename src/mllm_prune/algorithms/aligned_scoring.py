"""Map frozen aligned-encoder relevance onto Qwen text and vision candidates."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

from PIL import Image
import torch
import torch.nn.functional as F

from mllm_prune.models.aligned_encoder import CLIPAlignedEncoder
from mllm_prune.models.qwen25vl import EncodedSample


@dataclass(frozen=True)
class RawAlignedScores:
    """Raw token-level scores and external scoring timings for one sample."""

    text: torch.Tensor
    vision: torch.Tensor
    text_encode_ms: float
    vision_encode_ms: float
    mapping_ms: float


def _text_windows(
    encoded: EncodedSample, window_tokens: int, stride_tokens: int
) -> tuple[list[str], list[tuple[int, ...]]]:
    if window_tokens < 1 or stride_tokens < 1:
        raise ValueError("text scoring window and stride must be positive")
    texts: list[str] = []
    covered: list[tuple[int, ...]] = []
    for chunk in encoded.layout.text_chunks:
        indices = list(chunk.token_indices)
        max_start = max(0, len(indices) - window_tokens)
        starts = list(range(0, max_start + 1, stride_tokens))
        if starts[-1] != max_start:
            starts.append(max_start)
        seen: set[tuple[int, ...]] = set()
        for start in starts:
            local = tuple(indices[start : start + window_tokens])
            if not local or local in seen:
                continue
            seen.add(local)
            char_start = encoded.text_candidate_char_spans[local[0]][0]
            char_end = encoded.text_candidate_char_spans[local[-1]][1]
            text = encoded.expanded_prompt[char_start:char_end].strip()
            texts.append(text)
            covered.append(local)
    counts = [0] * encoded.layout.num_text_tokens
    for indices in covered:
        for index in indices:
            counts[index] += 1
    if any(count == 0 for count in counts):
        raise ValueError("text scoring windows do not cover every candidate token")
    return texts, covered


def _encode_text_window_scores(
    encoder: CLIPAlignedEncoder, question: str, windows: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Score semantic windows and assign cosine lower-bound to blank windows."""

    nonblank = [index for index, text in enumerate(windows) if text]
    started = time.perf_counter()
    text_features = encoder.encode_text([question, *(windows[index] for index in nonblank)])
    text_encode_ms = (time.perf_counter() - started) * 1000
    query = text_features[0]
    scores = torch.full((len(windows),), -1.0, dtype=torch.float32)
    if nonblank:
        semantic_scores = text_features[1:] @ query
        scores[torch.tensor(nonblank, dtype=torch.long)] = semantic_scores.float().cpu()
    return query, scores, text_encode_ms


def _map_patch_scores(score_map: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if height < 1 or width < 1:
        raise ValueError("Qwen vision grid dimensions must be positive")
    mapped = F.interpolate(
        score_map[None, None].float(), size=(height, width), mode="area"
    )[0, 0]
    return mapped.flatten()


def compute_aligned_raw_scores(
    encoder: CLIPAlignedEncoder,
    question: str,
    encoded: EncodedSample,
    images: Sequence[Image.Image],
    *,
    window_tokens: int,
    stride_tokens: int,
) -> RawAlignedScores:
    """Compute aligned token scores without changing Qwen representations."""

    windows, coverage = _text_windows(encoded, window_tokens, stride_tokens)
    query, window_scores, text_encode_ms = _encode_text_window_scores(
        encoder, question, windows
    )
    text_sum = torch.zeros(encoded.layout.num_text_tokens, dtype=torch.float32)
    text_count = torch.zeros_like(text_sum)
    for score, indices in zip(window_scores, coverage, strict=True):
        index = torch.tensor(indices, dtype=torch.long)
        text_sum[index] += float(score)
        text_count[index] += 1
    if (text_count == 0).any():
        raise RuntimeError("text score aggregation left uncovered tokens")
    text_scores = text_sum / text_count

    dense_images, vision_encode_ms = encoder.encode_image_patches(images)
    if len(dense_images) != len(encoded.image_grid_thw):
        raise ValueError("external image count does not match Qwen image grids")
    mapping_started = time.perf_counter()
    vision_scores: list[torch.Tensor] = []
    for dense, grid in zip(dense_images, encoded.image_grid_thw.tolist(), strict=True):
        temporal, raw_height, raw_width = (int(value) for value in grid)
        if temporal != 1:
            raise NotImplementedError("experiment 003 currently supports images only")
        height = raw_height // 2
        width = raw_width // 2
        score_map = torch.einsum("hwd,d->hw", dense.features, query)
        vision_scores.append(_map_patch_scores(score_map, height, width))
    vision = torch.cat(vision_scores) if vision_scores else torch.empty(0)
    mapping_ms = (time.perf_counter() - mapping_started) * 1000
    if len(vision) != encoded.layout.num_vision_tokens:
        raise ValueError("mapped external scores do not match Qwen visual candidates")
    if not torch.isfinite(text_scores).all() or not torch.isfinite(vision).all():
        raise ValueError("aligned scorer produced NaN or Inf")
    return RawAlignedScores(text_scores, vision, text_encode_ms, vision_encode_ms, mapping_ms)
