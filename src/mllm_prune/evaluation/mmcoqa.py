"""MMCoQA answer normalization and substring accuracy."""

from __future__ import annotations

import re


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize_answer(text: str) -> str:
    """Apply the normalization used by the existing MMCoQA experiment."""

    text = text.replace("\n", " ").replace("\t", " ").strip().lower()
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    return " ".join(text.split())


def is_correct(prediction: str, answer: str) -> bool:
    """Use normalized ground-truth substring matching."""

    normalized_answer = normalize_answer(answer)
    return bool(normalized_answer) and normalized_answer in normalize_answer(prediction)


def normalize_answer_robust(text: str) -> str:
    """Add conservative article and simple number-word normalization."""

    tokens = normalize_answer(text).split()
    tokens = [_NUMBER_WORDS.get(token, token) for token in tokens]
    tokens = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def is_correct_robust(prediction: str, answer: str) -> bool:
    """Score with the additive robust adapter while preserving the legacy metric."""

    normalized_answer = normalize_answer_robust(answer)
    return bool(normalized_answer) and normalized_answer in normalize_answer_robust(prediction)

