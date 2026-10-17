"""Small structural tests for Qwen prompt and grid mappings."""

import torch

from mllm_prune.models.qwen25vl import _expand_image_pads, _make_vision_chunks


def test_expand_image_pads_uses_per_image_counts() -> None:
    prompt = "a<|image_pad|>b<|image_pad|>c"
    expanded = _expand_image_pads(prompt, [2, 3])
    assert expanded == "a<|image_pad|><|image_pad|>b<|image_pad|><|image_pad|><|image_pad|>c"


def test_vision_chunks_partition_each_image_without_crossing() -> None:
    # After spatial merge by 2 these become 2x3 and 1x2 token grids.
    chunks = _make_vision_chunks(torch.tensor([[1, 4, 6], [1, 2, 4]]), window=2)
    flattened = [index for chunk in chunks for index in chunk.token_indices]
    assert sorted(flattened) == list(range(8))
    assert len(set(flattened)) == 8
    assert all(max(chunk.token_indices) < 6 for chunk in chunks if chunk.source_id == 0)
    assert all(min(chunk.token_indices) >= 6 for chunk in chunks if chunk.source_id == 1)
