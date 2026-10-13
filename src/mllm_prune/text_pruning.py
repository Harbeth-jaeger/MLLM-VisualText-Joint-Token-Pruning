"""Prune text chunks while preserving MileBench image/table placeholders.

The module implements the deliberately small set of selection strategies used by
the first exploratory experiment. A caller supplies the model tokenizer so that
budgets and reported token counts use the same tokenization as inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
from typing import Callable, Literal, Sequence

from rank_bm25 import BM25Okapi


Strategy = Literal["bm25", "random", "uniform"]
Tokenize = Callable[[str], Sequence[int]]

_PLACEHOLDER_RE = re.compile(r"(\{(?:image|table)#\d+\})", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class TextChunk:
    """A selectable text span and its location in the interleaved context."""

    chunk_id: int
    segment_index: int
    text: str
    token_count: int


@dataclass(frozen=True)
class PruningResult:
    """Pruned context plus enough diagnostics to audit the selection."""

    context: str
    strategy: str
    target_retention: float
    original_tokens: int
    kept_tokens: int
    actual_retention: float
    selected_chunk_ids: tuple[int, ...]
    chunk_scores: tuple[float, ...]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return asdict(self)


def lexical_tokens(text: str) -> list[str]:
    """Tokenize English text for BM25 scoring."""

    return [token.lower() for token in _WORD_RE.findall(text)]


def split_question(context: str) -> tuple[str, str]:
    """Separate the final question from the prunable preceding context."""

    question_start = context.rfind("Question:")
    if question_start < 0:
        raise ValueError("MileBench context does not contain a final 'Question:' marker")
    return context[:question_start], context[question_start:]


def _split_piece(text: str, target_tokens: int, tokenize: Tokenize) -> list[str]:
    """Group sentences into chunks close to the requested model-token size."""

    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if current and len(tokenize(candidate)) > target_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(
    prunable_context: str,
    tokenize: Tokenize,
    target_tokens: int = 64,
) -> tuple[list[str], list[TextChunk]]:
    """Split text without crossing or removing image/table placeholders."""

    segments = _PLACEHOLDER_RE.split(prunable_context)
    chunks: list[TextChunk] = []
    next_id = 0
    for segment_index, segment in enumerate(segments):
        if not segment or _PLACEHOLDER_RE.fullmatch(segment):
            continue
        for text in _split_piece(segment, target_tokens, tokenize):
            chunks.append(
                TextChunk(
                    chunk_id=next_id,
                    segment_index=segment_index,
                    text=text,
                    token_count=len(tokenize(text)),
                )
            )
            next_id += 1
    return segments, chunks


def _rank_chunks(
    chunks: Sequence[TextChunk],
    query: str,
    strategy: Strategy,
    seed: int,
) -> tuple[list[int], list[float]]:
    """Return chunk IDs in selection order and per-chunk diagnostic scores."""

    if not chunks:
        return [], []
    if strategy == "bm25":
        corpus = [lexical_tokens(chunk.text) for chunk in chunks]
        scores = BM25Okapi(corpus).get_scores(lexical_tokens(query)).tolist()
        order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        return order, [float(score) for score in scores]
    if strategy == "random":
        order = list(range(len(chunks)))
        random.Random(seed).shuffle(order)
        return order, [0.0] * len(chunks)
    if strategy == "uniform":
        # Alternating evenly spaced positions gives broad coverage at any budget.
        count = len(chunks)
        order: list[int] = []
        level = 1
        while len(order) < count:
            for position in range(level):
                index = min(count - 1, math.floor((position + 0.5) * count / level))
                if index not in order:
                    order.append(index)
            level *= 2
        return order, [0.0] * len(chunks)
    raise ValueError(f"Unsupported pruning strategy: {strategy}")


def _select_to_budget(
    chunks: Sequence[TextChunk], order: Sequence[int], budget: int
) -> set[int]:
    """Select whole chunks in ranking order without exceeding the budget."""

    selected: set[int] = set()
    used = 0
    for chunk_id in order:
        length = chunks[chunk_id].token_count
        if used + length <= budget:
            selected.add(chunk_id)
            used += length
    return selected


def _rebuild_context(
    segments: Sequence[str], chunks: Sequence[TextChunk], selected: set[int]
) -> str:
    """Reconstruct context in original order with every placeholder preserved."""

    kept_by_segment: dict[int, list[str]] = {}
    for chunk in chunks:
        if chunk.chunk_id in selected:
            kept_by_segment.setdefault(chunk.segment_index, []).append(chunk.text)

    output: list[str] = []
    for segment_index, segment in enumerate(segments):
        if _PLACEHOLDER_RE.fullmatch(segment):
            output.append(segment)
        elif segment_index in kept_by_segment:
            output.append(" ".join(kept_by_segment[segment_index]))
    return " ".join(part.strip() for part in output if part.strip())


def prune_context(
    context: str,
    tokenize: Tokenize,
    strategy: Strategy,
    retention: float,
    *,
    seed: int = 0,
    chunk_tokens: int = 64,
) -> PruningResult:
    """Prune MileBench context text and retain the final question verbatim."""

    if not 0 < retention <= 1:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    prunable, question = split_question(context)
    segments, chunks = build_chunks(prunable, tokenize, chunk_tokens)
    original_tokens = sum(chunk.token_count for chunk in chunks)
    if retention == 1 or not chunks:
        selected = {chunk.chunk_id for chunk in chunks}
        scores = [0.0] * len(chunks)
    else:
        order, scores = _rank_chunks(chunks, question, strategy, seed)
        selected = _select_to_budget(chunks, order, math.floor(original_tokens * retention))

    kept_tokens = sum(chunks[index].token_count for index in selected)
    rebuilt = _rebuild_context(segments, chunks, selected)
    final_context = f"{rebuilt} {question}".strip()
    return PruningResult(
        context=final_context,
        strategy=strategy,
        target_retention=retention,
        original_tokens=original_tokens,
        kept_tokens=kept_tokens,
        actual_retention=kept_tokens / original_tokens if original_tokens else 1.0,
        selected_chunk_ids=tuple(sorted(selected)),
        chunk_scores=tuple(scores),
    )
