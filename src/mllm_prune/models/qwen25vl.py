"""Qwen2.5-VL adapter for projector-output pruning and explicit mRoPE."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

import torch
from qwen_vl_utils import process_vision_info

from mllm_prune.algorithms.joint_pruning import CandidateLayout, Chunk, PruningPlan
from mllm_prune.data.mmcoqa import PromptSpec


_BOUNDARY_RE = re.compile(r"[.!?;]\s*$|\n\s*$")
_PUNCT_ONLY_RE = re.compile(r"^\W+$", re.UNICODE)


@dataclass
class EncodedSample:
    """Full Qwen input and all mappings required for safe compression."""

    input_ids: torch.Tensor
    input_embeddings: torch.Tensor
    position_ids: torch.Tensor
    rope_delta: torch.Tensor
    attention_mask: torch.Tensor
    layout: CandidateLayout
    text_sequence_indices: torch.Tensor
    vision_sequence_indices: torch.Tensor
    query_sequence_indices: torch.Tensor
    protected_sequence_indices: torch.Tensor
    image_grid_thw: torch.Tensor
    pixel_values: torch.Tensor
    vision_encode_ms: float
    expanded_prompt: str
    text_candidate_char_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class GenerationResult:
    """Decoded tokens and timing from compressed greedy generation."""

    token_ids: tuple[int, ...]
    prefill_ms: float
    decode_ms: float
    compressed_input_tokens: int


def _find_in_order(text: str, values: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for value in values:
        start = text.find(value, cursor)
        if start < 0:
            raise ValueError(f"Could not align prompt fragment beginning {value[:40]!r}")
        spans.append((start, start + len(value)))
        cursor = start + len(value)
    return spans


def _expand_image_pads(prompt: str, counts: list[int]) -> str:
    token = "<|image_pad|>"
    pieces = prompt.split(token)
    if len(pieces) - 1 != len(counts):
        raise ValueError("prompt image placeholders do not match image grids")
    output = [pieces[0]]
    for count, tail in zip(counts, pieces[1:], strict=True):
        output.extend((token * count, tail))
    return "".join(output)


def _token_groups(
    prompt: str,
    tokenizer: Any,
    candidate_fragments: tuple[str, ...],
    query_text: str,
    final_input_ids: torch.Tensor,
) -> tuple[list[int], list[int], list[int], list[tuple[int, int]]]:
    """Align prompt character spans to final processor token positions."""

    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("experiment 002 requires a fast tokenizer for offset mapping")
    fragment_spans = _find_in_order(prompt, [*candidate_fragments, query_text])
    candidate_spans = fragment_spans[:-1]
    query_span = fragment_spans[-1]
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    if encoded["input_ids"] != final_input_ids.tolist():
        raise ValueError("expanded prompt tokenization differs from processor input_ids")
    text_positions: list[int] = []
    query_positions: list[int] = []
    source_ids: list[int] = []
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for source_id, (span_start, span_end) in enumerate(candidate_spans):
            if start >= span_start and end <= span_end:
                text_positions.append(token_index)
                source_ids.append(source_id)
                break
        if start >= query_span[0] and end <= query_span[1]:
            piece = prompt[start:end]
            if piece.strip() and not _PUNCT_ONLY_RE.fullmatch(piece.strip()):
                query_positions.append(token_index)
    if not text_positions or not query_positions:
        raise ValueError("prompt alignment produced empty text or query candidates")
    return text_positions, query_positions, source_ids, offsets


def _make_text_chunks(
    candidate_positions: list[int],
    source_ids: list[int],
    offsets: list[tuple[int, int]],
    prompt: str,
    min_tokens: int,
    max_tokens: int,
) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for source_id in sorted(set(source_ids)):
        local_indices = [i for i, value in enumerate(source_ids) if value == source_id]
        cursor = 0
        source_chunks: list[list[int]] = []
        while cursor < len(local_indices):
            upper = min(cursor + max_tokens, len(local_indices))
            cut = upper
            for position in range(cursor + min_tokens, upper + 1):
                sequence_index = candidate_positions[local_indices[position - 1]]
                start = offsets[sequence_index][0]
                end = offsets[sequence_index][1]
                if _BOUNDARY_RE.search(prompt[start:end]):
                    cut = position
            source_chunks.append(local_indices[cursor:cut])
            cursor = cut
        if len(source_chunks) > 1 and len(source_chunks[-1]) < min_tokens:
            if len(source_chunks[-2]) + len(source_chunks[-1]) <= max_tokens:
                source_chunks[-2].extend(source_chunks.pop())
        for indices in source_chunks:
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    token_indices=tuple(indices),
                    source_id=source_id,
                )
            )
    return tuple(chunks)


def _make_vision_chunks(image_grid_thw: torch.Tensor, window: int) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    offset = 0
    for image_index, grid in enumerate(image_grid_thw.tolist()):
        temporal, height, width = grid
        height //= 2
        width //= 2
        for t in range(temporal):
            for row in range(0, height, window):
                for column in range(0, width, window):
                    indices = []
                    for dr in range(window):
                        for dc in range(window):
                            if row + dr < height and column + dc < width:
                                indices.append(offset + t * height * width + (row + dr) * width + column + dc)
                    chunks.append(Chunk(len(chunks), tuple(indices), image_index))
        offset += temporal * height * width
    return tuple(chunks)


def encode_sample(
    model: Any,
    processor: Any,
    prompt_spec: PromptSpec,
    *,
    device: torch.device,
    min_text_tokens: int = 16,
    max_text_tokens: int = 64,
    vision_window: int = 2,
) -> EncodedSample:
    """Encode images once, build full mRoPE, and derive pruning candidates."""

    prompt = processor.apply_chat_template(
        prompt_spec.messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(prompt_spec.messages)
    if video_inputs:
        raise NotImplementedError("experiment 002 currently supports images only")
    inputs = processor(
        text=[prompt], images=image_inputs, videos=None, padding=True, return_tensors="pt"
    ).to(device)
    image_grid = inputs["image_grid_thw"]
    merge = int(model.config.vision_config.spatial_merge_size)
    image_counts = (image_grid.prod(-1) // merge**2).tolist()
    expanded_prompt = _expand_image_pads(prompt, image_counts)
    ids = inputs["input_ids"][0]
    text_positions, query_positions, source_ids, offsets = _token_groups(
        expanded_prompt,
        processor.tokenizer,
        prompt_spec.candidate_fragments,
        prompt_spec.query_text,
        ids.cpu(),
    )
    image_token_id = int(model.config.image_token_id)
    vision_positions = torch.nonzero(ids == image_token_id, as_tuple=False).flatten().tolist()
    if len(vision_positions) != sum(image_counts):
        raise ValueError("visual placeholder count does not match projected image tokens")

    started = time.perf_counter()
    image_features = model.model.get_image_features(inputs["pixel_values"], image_grid)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    vision_encode_ms = (time.perf_counter() - started) * 1000
    vision_embeddings = torch.cat(image_features, dim=0)
    embeddings = model.get_input_embeddings()(ids.unsqueeze(0)).clone()
    embeddings[0, torch.tensor(vision_positions, device=device)] = vision_embeddings.to(
        device=device, dtype=embeddings.dtype
    )
    position_ids, rope_delta = model.model.get_rope_index(
        inputs["input_ids"], image_grid, attention_mask=inputs["attention_mask"]
    )
    text_positions = sorted(text_positions)
    # source_ids tracks text_positions before sorting; both are already prompt ordered.
    text_chunks = _make_text_chunks(
        text_positions, source_ids, offsets, expanded_prompt, min_text_tokens, max_text_tokens
    )
    vision_chunks = _make_vision_chunks(image_grid.cpu(), vision_window)
    layout = CandidateLayout(
        text_chunks=text_chunks,
        vision_chunks=vision_chunks,
        num_text_tokens=len(text_positions),
        num_vision_tokens=len(vision_positions),
    )
    layout.validate()
    candidate_set = set(text_positions) | set(vision_positions)
    protected = [index for index in range(len(ids)) if index not in candidate_set]
    return EncodedSample(
        input_ids=inputs["input_ids"],
        input_embeddings=embeddings,
        position_ids=position_ids,
        rope_delta=rope_delta,
        attention_mask=inputs["attention_mask"],
        layout=layout,
        text_sequence_indices=torch.tensor(text_positions, device=device),
        vision_sequence_indices=torch.tensor(vision_positions, device=device),
        query_sequence_indices=torch.tensor(query_positions, device=device),
        protected_sequence_indices=torch.tensor(protected, device=device),
        image_grid_thw=image_grid,
        pixel_values=inputs["pixel_values"],
        vision_encode_ms=vision_encode_ms,
        expanded_prompt=expanded_prompt,
        text_candidate_char_spans=tuple(offsets[index] for index in text_positions),
    )


def candidate_embeddings(encoded: EncodedSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return text, visual, and filtered query embeddings in a shared space."""

    hidden = encoded.input_embeddings[0]
    return (
        hidden[encoded.text_sequence_indices],
        hidden[encoded.vision_sequence_indices],
        hidden[encoded.query_sequence_indices],
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_from_plan(
    model: Any,
    encoded: EncodedSample,
    plan: PruningPlan,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> GenerationResult:
    """Greedily decode from gathered embeddings and gathered full mRoPE IDs."""

    device = encoded.input_embeddings.device
    selected = set(encoded.protected_sequence_indices.tolist())
    selected.update(encoded.text_sequence_indices[list(plan.selected_text_indices)].tolist())
    selected.update(encoded.vision_sequence_indices[list(plan.selected_vision_indices)].tolist())
    keep = torch.tensor(sorted(selected), device=device)
    compressed_embeddings = encoded.input_embeddings[:, keep]
    compressed_positions = encoded.position_ids[:, :, keep]
    length = compressed_embeddings.shape[1]
    attention_mask = torch.ones((1, length), dtype=torch.long, device=device)
    cache_position = torch.arange(length, device=device)

    _synchronize(device)
    started = time.perf_counter()
    outputs = model(
        inputs_embeds=compressed_embeddings,
        position_ids=compressed_positions,
        attention_mask=attention_mask,
        use_cache=True,
        cache_position=cache_position,
        logits_to_keep=1,
        return_dict=True,
    )
    _synchronize(device)
    prefill_ms = (time.perf_counter() - started) * 1000
    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    past = outputs.past_key_values
    generated: list[int] = []
    next_position = int(encoded.position_ids.max().item()) + 1

    _synchronize(device)
    started = time.perf_counter()
    for step in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id in eos_token_ids:
            break
        attention_mask = torch.ones((1, length + step + 1), dtype=torch.long, device=device)
        step_positions = torch.full(
            (3, 1, 1), next_position + step, dtype=torch.long, device=device
        )
        outputs = model(
            input_ids=next_token,
            position_ids=step_positions,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            cache_position=torch.tensor([length + step], device=device),
            logits_to_keep=1,
            return_dict=True,
        )
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        past = outputs.past_key_values
    _synchronize(device)
    decode_ms = (time.perf_counter() - started) * 1000
    return GenerationResult(tuple(generated), prefill_ms, decode_ms, length)
