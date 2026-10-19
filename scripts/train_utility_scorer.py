"""Train the lightweight utility scorer from cached per-sample records."""

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
    stratified_pairs,
    utility_loss,
)
from mllm_prune.config import path_from_env
from mllm_prune.models.utility_scorer import (
    JointUtilityScorer,
    UtilityScorerConfig,
    save_scorer_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/004_learned_utility_scorer.yaml"),
    )
    labels = path_from_env("EXPERIMENT_004_LABEL_DIR")
    checkpoints = path_from_env("EXPERIMENT_004_CHECKPOINT_DIR")
    parser.add_argument("--label-root", type=Path, default=labels, required=labels is None)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=checkpoints, required=checkpoints is None
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def _load_records(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("*.pt"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"no utility records found in {directory}")
    records = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    fingerprints = {record["fingerprint"] for record in records}
    if len(fingerprints) != 1:
        raise ValueError(f"mixed record fingerprints in {directory}")
    return records


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward(model: JointUtilityScorer, record: dict, device: torch.device) -> torch.Tensor:
    features = record["features"]
    return model(
        features["chunk_features"].to(device=device, dtype=torch.float32),
        features["question_feature"].to(device=device, dtype=torch.float32),
        features["modality_ids"].to(device),
        features["structural_features"].to(device),
    )


@torch.inference_mode()
def _validate(
    model: JointUtilityScorer,
    records: list[dict],
    device: torch.device,
    *,
    center: float,
    scale: float,
    gap_threshold: float,
) -> dict[str, float]:
    model.eval()
    accuracies: list[float] = []
    cross_correct = cross_total = 0
    maes: list[float] = []
    for record in records:
        prediction = _forward(model, record, device).cpu() * scale + center
        labels = record["utility_labels"].float()
        accuracy = pairwise_accuracy(prediction, labels, gap_threshold=gap_threshold)
        if not math.isnan(accuracy):
            accuracies.append(accuracy)
        modalities = record["features"]["modality_ids"]
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                gap = float(labels[left] - labels[right])
                if modalities[left] == modalities[right] or abs(gap) <= gap_threshold:
                    continue
                cross_correct += int(
                    (float(prediction[left] - prediction[right]) > 0) == (gap > 0)
                )
                cross_total += 1
        maes.append(float(torch.mean(torch.abs(prediction - labels))))
    return {
        "macro_pairwise_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "cross_modal_pairwise_accuracy": cross_correct / cross_total if cross_total else 0.0,
        "mae": sum(maes) / len(maes),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    _set_seed(args.seed)
    device = torch.device(args.device)
    train_records = _load_records(args.label_root / "train")
    validation_records = _load_records(args.label_root / "validation")
    settings = {
        record.get("settings_fingerprint") for record in train_records + validation_records
    }
    if len(settings) != 1 or None in settings:
        raise ValueError("train/validation records use different or legacy feature settings")
    all_labels = torch.cat([record["utility_labels"].float() for record in train_records])
    center = float(torch.median(all_labels))
    q25, q75 = torch.quantile(all_labels, torch.tensor([0.25, 0.75]))
    scale = max(float(q75 - q25), 1e-6)
    first = train_records[0]["features"]
    feature_cfg = config["features"]
    scorer_config = UtilityScorerConfig(
        input_dim=int(first["chunk_features"].shape[1]),
        structural_dim=int(first["structural_features"].shape[1]),
        projection_dim=int(feature_cfg["projection_dim"]),
        hidden_dim=int(feature_cfg["hidden_dim"]),
        modality_dim=int(feature_cfg["modality_dim"]),
        dropout=float(feature_cfg["dropout"]),
    )
    model = JointUtilityScorer(scorer_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = args.epochs or int(training["epochs"])
    gap = float(training["pair_gap_threshold"])
    max_pairs = int(training["max_pairs_per_type_per_sample"])
    patience = int(training["patience"])
    output = args.checkpoint_dir / f"seed_{args.seed}.pt"
    history: list[dict] = []
    best = -1.0
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_records)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_losses: list[float] = []
        for record_index in order:
            record = train_records[record_index]
            labels = record["utility_labels"].float().to(device)
            normalized = (labels - center) / scale
            high, low = stratified_pairs(
                labels,
                record["features"]["modality_ids"],
                gap_threshold=gap,
                max_per_type=max_pairs,
                seed=args.seed * 1_000_003 + epoch * len(train_records) + record_index,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, record, device)
            loss, metrics = utility_loss(
                prediction,
                normalized,
                high.to(device),
                low.to(device),
                huber_weight=float(training["huber_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(metrics["loss"])
        validation = _validate(
            model,
            validation_records,
            device,
            center=center,
            scale=scale,
            gap_threshold=gap,
        )
        row = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            **validation,
        }
        history.append(row)
        print(json.dumps(row))
        score = validation["macro_pairwise_accuracy"]
        if score > best:
            best = score
            stale = 0
            save_scorer_artifact(
                output,
                model,
                label_center=center,
                label_scale=scale,
                metadata={
                    "seed": args.seed,
                    "best_epoch": epoch,
                    "best_validation": validation,
                    "train_samples": len(train_records),
                    "validation_samples": len(validation_records),
                    "record_format_version": 1,
                    "record_settings_fingerprint": next(iter(settings)),
                },
            )
        else:
            stale += 1
            if stale >= patience:
                break
    history_path = args.checkpoint_dir / f"seed_{args.seed}_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"best_checkpoint={output} best_macro_pairwise_accuracy={best:.6f}")


if __name__ == "__main__":
    main()

