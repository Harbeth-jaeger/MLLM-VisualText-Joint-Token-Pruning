"""Run experiment 001 with a local Qwen2.5-VL model.

The script loads the model once, evaluates the fixed MMCoQA subset under the
baseline and three pruning strategies, and appends one JSON record per run so a
long experiment can resume safely after interruption.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from mllm_prune.config import path_from_env, value_from_env
from mllm_prune.text_pruning import PruningResult, prune_context


_PLACEHOLDER_RE = re.compile(r"\{(?:image|table)#(\d+)\}", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """Parse paths and small-experiment settings."""

    parser = argparse.ArgumentParser()
    subset = path_from_env("EXPERIMENT_001_SUBSET_PATH")
    image_dir = path_from_env("MMCOQA_IMAGE_DIR")
    model_path = path_from_env("QWEN_MODEL_PATH")
    output = path_from_env("EXPERIMENT_001_PREDICTIONS_PATH")
    attention = value_from_env("MLLM_ATTN_IMPLEMENTATION", "sdpa")
    parser.add_argument("--subset", type=Path, default=subset, required=subset is None)
    parser.add_argument(
        "--image-dir", type=Path, default=image_dir, required=image_dir is None
    )
    parser.add_argument(
        "--model-path", type=Path, default=model_path, required=model_path is None
    )
    parser.add_argument("--output", type=Path, default=output, required=output is None)
    parser.add_argument(
        "--attn-implementation",
        default=attention,
        choices=("flash_attention_2", "sdpa", "eager"),
        help="Attention backend; defaults to MLLM_ATTN_IMPLEMENTATION from .env.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def experiment_conditions() -> list[tuple[str, float, int]]:
    """Return the deliberately small condition grid from the experiment plan."""

    conditions = [("baseline", 1.0, 0)]
    for strategy in ("bm25", "uniform"):
        conditions.extend((strategy, retention, 0) for retention in (0.5, 0.25))
    for seed in (0, 1, 2):
        conditions.extend(("random", retention, seed) for retention in (0.5, 0.25))
    return conditions


def normalize_answer(text: str) -> str:
    """Apply the simple normalization used by MileBench MMCoQA scoring."""

    text = text.replace("\n", " ").replace("\t", " ").strip().lower()
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    return " ".join(text.split())


def is_correct(prediction: str, answer: str) -> bool:
    """Score MMCoQA using normalized ground-truth substring matching."""

    normalized_answer = normalize_answer(answer)
    return bool(normalized_answer) and normalized_answer in normalize_answer(prediction)


def task_instruction(subset: dict, sample: dict) -> str:
    """Resolve either official JSON or preview-style task instructions."""

    if "task_instruction" in sample:
        return sample["task_instruction"]
    instructions = subset["meta_data"]["task_instruction"]
    return instructions[sample["task_instruction_id"]]


def build_messages(
    instruction: str,
    context: str,
    image_paths: list[Path],
) -> list[dict]:
    """Convert MileBench placeholders into Qwen's interleaved message format."""

    content: list[dict] = [{"type": "text", "text": instruction + "\n"}]
    cursor = 0
    seen: list[int] = []
    for match in _PLACEHOLDER_RE.finditer(context):
        if match.start() > cursor:
            content.append({"type": "text", "text": context[cursor : match.start()]})
        image_index = int(match.group(1)) - 1
        if not 0 <= image_index < len(image_paths):
            raise IndexError(
                f"Placeholder {match.group(0)} has no matching image; "
                f"sample supplies {len(image_paths)} images"
            )
        content.append({"type": "image", "image": str(image_paths[image_index])})
        seen.append(image_index)
        cursor = match.end()
    if cursor < len(context):
        content.append({"type": "text", "text": context[cursor:]})
    if sorted(seen) != list(range(len(image_paths))):
        raise ValueError(
            f"Context references image indices {seen}, expected 0..{len(image_paths) - 1}"
        )
    return [{"role": "user", "content": content}]


def load_completed(path: Path) -> set[tuple[int, str, float, int]]:
    """Load completed condition keys from an existing JSONL output."""

    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add(
                    (
                        int(row["sample_id"]),
                        row["strategy"],
                        float(row["target_retention"]),
                        int(row["seed"]),
                    )
                )
    return completed


def make_pruning_result(
    context: str,
    tokenize,
    strategy: str,
    retention: float,
    seed: int,
    chunk_tokens: int,
) -> PruningResult:
    """Construct the unpruned baseline or run the requested pruning method."""

    if strategy == "baseline":
        return prune_context(
            context, tokenize, "bm25", 1.0, seed=seed, chunk_tokens=chunk_tokens
        )
    return prune_context(
        context,
        tokenize,
        strategy,
        retention,
        seed=seed,
        chunk_tokens=chunk_tokens,
    )


def main() -> None:
    """Run all pending sample-condition pairs and append inspectable results."""

    args = parse_args()
    with args.subset.open(encoding="utf-8") as handle:
        subset = json.load(handle)
    samples = subset["data"][: args.limit]

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=args.attn_implementation,
    ).eval()
    tokenize = lambda text: processor.tokenizer.encode(text, add_special_tokens=False)
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output)
    total = len(samples) * len(experiment_conditions())
    finished = len(completed)
    print(f"Running {total - finished}/{total} pending generations")

    with args.output.open("a", encoding="utf-8") as output_handle:
        for sample in samples:
            sample_id = int(sample["sample_id"])
            instance = sample["task_instance"]
            context = instance["context"]
            paths = [args.image_dir / item for item in instance["images_path"]]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing images for sample {sample_id}: {missing}")
            # Detect corrupt images before loading the model inputs.
            for path in paths:
                with Image.open(path) as image:
                    image.verify()

            for strategy, retention, seed in experiment_conditions():
                key = (sample_id, strategy, retention, seed)
                if key in completed:
                    continue
                prune_start = time.perf_counter()
                pruning = make_pruning_result(
                    context,
                    tokenize,
                    strategy,
                    retention,
                    seed,
                    args.chunk_tokens,
                )
                pruning_ms = (time.perf_counter() - prune_start) * 1000
                messages = build_messages(
                    task_instruction(subset, sample), pruning.context, paths
                )
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda:0")
                visual_tokens = int((inputs["input_ids"] == image_pad_id).sum().item())

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - inference_start) * 1000
                new_tokens = generated[:, inputs["input_ids"].shape[1] :]
                prediction = processor.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                row = {
                    "sample_id": sample_id,
                    **pruning.to_dict(),
                    "strategy": strategy,
                    "target_retention": retention,
                    "seed": seed,
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "visual_tokens": visual_tokens,
                    "output_tokens": int(new_tokens.shape[1]),
                    "pruning_ms": pruning_ms,
                    "inference_ms": inference_ms,
                    "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "prediction": prediction,
                    "answer": str(sample["response"]),
                    "correct": is_correct(prediction, str(sample["response"])),
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
                finished += 1
                print(
                    f"[{finished}/{total}] sample={sample_id} {strategy} "
                    f"r={retention} seed={seed} correct={row['correct']}"
                )


if __name__ == "__main__":
    main()
