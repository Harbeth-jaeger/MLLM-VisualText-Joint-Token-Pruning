"""Tests for experiment 002 prompt construction."""

from pathlib import Path

import pytest

from mllm_prune.data.mmcoqa import build_prompt_spec, split_context_question


def test_prompt_spec_separates_candidates_query_and_images() -> None:
    context = "First fact. {image#1} Second fact. Question: What happened? Answer:"
    spec = build_prompt_spec("Answer briefly.", context, [Path("image.jpg")])
    assert spec.candidate_fragments == ("First fact. ", " Second fact. ")
    assert spec.query_text == "Question: What happened? Answer:"
    assert [item["type"] for item in spec.messages[0]["content"]] == ["text", "text", "image", "text", "text"]


def test_question_marker_is_required() -> None:
    with pytest.raises(ValueError):
        split_context_question("No explicit question marker")
