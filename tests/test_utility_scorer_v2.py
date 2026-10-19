from pathlib import Path

import torch

from mllm_prune.models.utility_scorer_v2 import (
    JointUtilityScorerV2,
    UtilityScorerV2Config,
    load_scorer_v2_artifact,
    save_scorer_v2_artifact,
)


def test_scorer_v2_forward_and_roundtrip(tmp_path: Path):
    config = UtilityScorerV2Config(8, 6, 4, 5, projection_dim=4, hidden_dim=8)
    model = JointUtilityScorerV2(config).eval()
    inputs = (
        torch.randn(3, 8),
        torch.randn(8),
        torch.tensor([0, 1, 1]),
        torch.randn(3, 6),
        torch.randn(3, 4),
        torch.randn(3, 5),
    )
    with torch.inference_mode():
        expected = model(*inputs)
    path = tmp_path / "scorer.pt"
    save_scorer_v2_artifact(
        path, model, label_center=0.1, label_scale=2.0, metadata={"seed": 0}
    )
    loaded, artifact = load_scorer_v2_artifact(path, torch.device("cpu"))
    with torch.inference_mode():
        actual = loaded(*inputs)
    assert artifact["format_version"] == 2
    assert torch.equal(expected, actual)
