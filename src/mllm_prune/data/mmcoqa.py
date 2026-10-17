"""MileBench/MMCoQA loading and Qwen-compatible interleaved prompts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from PIL import Image


_PLACEHOLDER_RE = re.compile(r"\{(?:image|table)#(\d+)\}", re.IGNORECASE)


@dataclass(frozen=True)
class PromptSpec:
    """A chat message plus exact text fragments used for token alignment."""

    messages: list[dict]
    candidate_fragments: tuple[str, ...]
    query_text: str


def load_subset(path: Path, limit: int | None = None) -> tuple[dict, list[dict]]:
    """Read a fixed MMCoQA subset without changing its stored order."""

    with path.open(encoding="utf-8") as handle:
        subset = json.load(handle)
    samples = subset["data"]
    return subset, samples if limit is None else samples[:limit]


def task_instruction(subset: dict, sample: dict) -> str:
    """Resolve both official and preview-style task instruction formats."""

    if "task_instruction" in sample:
        return str(sample["task_instruction"])
    return str(subset["meta_data"]["task_instruction"][sample["task_instruction_id"]])


def resolve_sample(subset: dict, sample: dict, image_dir: Path) -> tuple[str, str, list[Path]]:
    """Resolve instruction, context, and validated image paths for one sample."""

    instance = sample["task_instance"]
    paths = [image_dir / item for item in instance["images_path"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images for sample {sample['sample_id']}: {missing}")
    for path in paths:
        with Image.open(path) as image:
            image.verify()
    return task_instruction(subset, sample), str(instance["context"]), paths


def split_context_question(context: str) -> tuple[str, str]:
    """Separate prunable history from the protected final question."""

    start = context.rfind("Question:")
    if start < 0:
        raise ValueError("MMCoQA context has no final 'Question:' marker")
    return context[:start], context[start:]


def build_prompt_spec(instruction: str, context: str, image_paths: list[Path]) -> PromptSpec:
    """Replace placeholders with image blocks while retaining alignment fragments."""

    prunable, query = split_context_question(context)
    content: list[dict] = [{"type": "text", "text": instruction + "\n"}]
    fragments: list[str] = []
    cursor = 0
    seen: list[int] = []
    for match in _PLACEHOLDER_RE.finditer(prunable):
        if match.start() > cursor:
            fragment = prunable[cursor : match.start()]
            content.append({"type": "text", "text": fragment})
            fragments.append(fragment)
        image_index = int(match.group(1)) - 1
        if not 0 <= image_index < len(image_paths):
            raise IndexError(
                f"Placeholder {match.group(0)} has no matching image; "
                f"sample supplies {len(image_paths)} images"
            )
        content.append({"type": "image", "image": str(image_paths[image_index])})
        seen.append(image_index)
        cursor = match.end()
    if cursor < len(prunable):
        fragment = prunable[cursor:]
        content.append({"type": "text", "text": fragment})
        fragments.append(fragment)
    content.append({"type": "text", "text": query})
    if sorted(seen) != list(range(len(image_paths))):
        raise ValueError(
            f"Context references image indices {seen}, expected 0..{len(image_paths) - 1}"
        )
    return PromptSpec(
        messages=[{"role": "user", "content": content}],
        candidate_fragments=tuple(fragment for fragment in fragments if fragment),
        query_text=query,
    )

