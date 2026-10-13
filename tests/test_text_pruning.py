"""Tests for the text-only pruning rules used in experiment 001."""

from mllm_prune.text_pruning import prune_context


def whitespace_tokens(text: str) -> list[str]:
    """Provide deterministic token counts for unit tests."""

    return text.split()


CONTEXT = (
    "History one contains irrelevant red words and details. {table#1} "
    "History two says the winning driver was Alice Smith. {image#2} "
    "History three contains unrelated blue words and details. "
    "Question: Who was the winning driver? Answer:"
)


def test_bm25_keeps_relevant_text_and_all_placeholders() -> None:
    """BM25 should retain relevant evidence without dropping visual markers."""

    result = prune_context(
        CONTEXT, whitespace_tokens, "bm25", 0.5, chunk_tokens=8
    )

    assert "winning driver was Alice Smith" in result.context
    assert "{table#1}" in result.context
    assert "{image#2}" in result.context
    assert result.context.endswith("Question: Who was the winning driver? Answer:")
    assert result.actual_retention <= 0.5


def test_random_is_reproducible() -> None:
    """The same seed must produce the same random chunk selection."""

    first = prune_context(CONTEXT, whitespace_tokens, "random", 0.5, seed=7)
    second = prune_context(CONTEXT, whitespace_tokens, "random", 0.5, seed=7)

    assert first == second
