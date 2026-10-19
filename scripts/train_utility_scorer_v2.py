"""Train the state-conditioned utility scorer for experiment 005."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from mllm_prune.algorithms.utility_training import (
    pairwise_accuracy,
    stratified_pairs_v2,
    utility_loss,
)
from mllm_prune.config import path_from_env
from mllm_prune.models.utility_scorer_v2 import (
    JointUtilityScorerV2,
    UtilityScorerV2Config,
    save_scorer_v2_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/005_learned_utility_scorer_v2.yaml"))
    labels = path_from_env("EXPERIMENT_005_LABEL_DIR")
    checkpoints = path_from_env("EXPERIMENT_005_CHECKPOINT_DIR")
    parser.add_argument("--label-root", type=Path, default=labels, required=labels is None)
    parser.add_argument("--checkpoint-dir", type=Path, default=checkpoints, required=checkpoints is None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def _load_records(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no conditional utility records found in {directory}")
    records = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    if any(record.get("format_version") != 2 for record in records):
        raise ValueError(f"legacy or invalid conditional record in {directory}")
    settings = {record["settings_fingerprint"] for record in records}
    if len(settings) != 1:
        raise ValueError(f"mixed record settings in {directory}")
    keys = [(int(record["sample_id"]), record["state_id"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate sample/state records in {directory}")
    return records


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward(model: JointUtilityScorerV2, record: dict, device: torch.device) -> torch.Tensor:
    features = record["features"]
    return model(
        features["chunk_features"].to(device=device, dtype=torch.float32),
        features["question_feature"].to(device=device, dtype=torch.float32),
        features["modality_ids"].to(device),
        features["structural_features"].to(device),
        features["aligned_features"].to(device),
        features["budget_features"].to(device),
    )


@torch.inference_mode()
def _validate(model, records, device, *, center: float, scale: float, gap: float) -> dict:
    model.eval()
    overall = []
    by_state: dict[str, list[float]] = {}
    maes = []
    for record in records:
        prediction = _forward(model, record, device).cpu() * scale + center
        labels = record["utility_labels"].float()
        accuracy = pairwise_accuracy(prediction, labels, gap_threshold=gap)
        if not math.isnan(accuracy):
            overall.append(accuracy)
            by_state.setdefault(record["state_id"], []).append(accuracy)
        maes.append(float(torch.mean(torch.abs(prediction - labels))))
    return {
        "macro_pairwise_accuracy": sum(overall) / len(overall) if overall else 0.0,
        "mae": sum(maes) / len(maes),
        "state_pairwise_accuracy": {
            state: sum(values) / len(values) for state, values in sorted(by_state.items())
        },
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    _set_seed(args.seed)
    device = torch.device(args.device)
    train = _load_records(args.label_root / "train")
    validation = _load_records(args.label_root / "validation")
    settings = {record["settings_fingerprint"] for record in train + validation}
    if len(settings) != 1:
        raise ValueError("train and validation record settings differ")
    train_ids = {int(record["sample_id"]) for record in train}
    validation_ids = {int(record["sample_id"]) for record in validation}
    if train_ids & validation_ids:
        raise ValueError("train and validation sample IDs overlap")
    all_labels = torch.cat([record["utility_labels"].float() for record in train])
    center = float(torch.median(all_labels))
    q25, q75 = torch.quantile(all_labels, torch.tensor([0.25, 0.75]))
    scale = max(float(q75 - q25), 1e-6)
    first = train[0]["features"]
    feature_cfg = config["features"]
    scorer_cfg = UtilityScorerV2Config(
        qwen_input_dim=int(first["chunk_features"].shape[1]),
        structural_dim=int(first["structural_features"].shape[1]),
        aligned_dim=int(first["aligned_features"].shape[1]),
        budget_dim=int(first["budget_features"].shape[1]),
        projection_dim=int(feature_cfg["projection_dim"]),
        hidden_dim=int(feature_cfg["hidden_dim"]),
        modality_dim=int(feature_cfg["modality_dim"]),
        dropout=float(feature_cfg["dropout"]),
    )
    model = JointUtilityScorerV2(scorer_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = args.epochs or int(training["epochs"])
    gap = float(training["pair_gap_threshold"])
    max_pairs = int(training["max_pairs_per_type_per_sample"])
    high_fraction = float(training["high_gap_fraction"])
    patience = int(training["patience"])
    output = args.checkpoint_dir / f"seed_{args.seed}.pt"
    history = []
    best = -1.0
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for record_index in order:
            record = train[record_index]
            labels = record["utility_labels"].float().to(device)
            normalized = (labels - center) / scale
            high, low = stratified_pairs_v2(
                labels, record["features"]["modality_ids"], gap_threshold=gap,
                max_per_type=max_pairs, high_gap_fraction=high_fraction,
                seed=args.seed * 1_000_003 + epoch * len(train) + record_index,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, record, device)
            loss, metrics = utility_loss(
                prediction, normalized, high.to(device), low.to(device),
                huber_weight=float(training["huber_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(metrics["loss"])
        metrics = _validate(model, validation, device, center=center, scale=scale, gap=gap)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        score = metrics["macro_pairwise_accuracy"]
        if score > best:
            best = score
            stale = 0
            save_scorer_v2_artifact(
                output, model, label_center=center, label_scale=scale,
                metadata={
                    "seed": args.seed,
                    "best_epoch": epoch,
                    "best_validation": metrics,
                    "train_records": len(train),
                    "validation_records": len(validation),
                    "train_samples": len(train_ids),
                    "validation_samples": len(validation_ids),
                    "record_settings_fingerprint": next(iter(settings)),
                },
            )
        else:
            stale += 1
            if stale >= patience:
                break
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.checkpoint_dir / f"seed_{args.seed}_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"best={best:.6f} checkpoint={output} epochs={len(history)}")


if __name__ == "__main__":
    main()
