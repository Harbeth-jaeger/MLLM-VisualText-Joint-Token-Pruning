"""Lightweight shared scorer for text and vision chunk utility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class UtilityScorerConfig:
    input_dim: int
    structural_dim: int
    projection_dim: int = 256
    hidden_dim: int = 256
    modality_dim: int = 16
    dropout: float = 0.1


class JointUtilityScorer(nn.Module):
    """Project modality-specific inputs, then predict on one shared scale."""

    def __init__(self, config: UtilityScorerConfig) -> None:
        super().__init__()
        self.config = config
        self.text_projection = nn.Linear(config.input_dim, config.projection_dim)
        self.vision_projection = nn.Linear(config.input_dim, config.projection_dim)
        self.question_projection = nn.Linear(config.input_dim, config.projection_dim)
        self.modality_embedding = nn.Embedding(2, config.modality_dim)
        joined = config.projection_dim * 4 + config.structural_dim + config.modality_dim
        self.head = nn.Sequential(
            nn.LayerNorm(joined),
            nn.Linear(joined, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(
        self,
        chunk_features: torch.Tensor,
        question_feature: torch.Tensor,
        modality_ids: torch.Tensor,
        structural_features: torch.Tensor,
    ) -> torch.Tensor:
        if chunk_features.ndim != 2:
            raise ValueError("chunk_features must be rank 2")
        if question_feature.ndim == 1:
            question_feature = question_feature.unsqueeze(0).expand(len(chunk_features), -1)
        if question_feature.shape != chunk_features.shape:
            raise ValueError("expanded question feature shape must match chunks")
        text = self.text_projection(chunk_features)
        vision = self.vision_projection(chunk_features)
        modality_mask = modality_ids.bool().unsqueeze(1)
        chunk = torch.where(modality_mask, vision, text)
        question = self.question_projection(question_feature)
        joined = torch.cat(
            (
                chunk,
                question,
                chunk * question,
                torch.abs(chunk - question),
                structural_features,
                self.modality_embedding(modality_ids),
            ),
            dim=1,
        )
        return self.head(joined).squeeze(1)


def save_scorer_artifact(
    path: Path,
    model: JointUtilityScorer,
    *,
    label_center: float,
    label_scale: float,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "label_center": float(label_center),
            "label_scale": float(label_scale),
            "metadata": metadata,
        },
        path,
    )


def load_scorer_artifact(
    path: Path, device: torch.device
) -> tuple[JointUtilityScorer, dict]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported utility scorer artifact version")
    model = JointUtilityScorer(UtilityScorerConfig(**artifact["config"]))
    model.load_state_dict(artifact["state_dict"])
    model.to(device).eval()
    return model, artifact

