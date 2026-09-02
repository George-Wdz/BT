#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from dataset import (
    MinuteRainDataset,
    TrainTransforms,
    collate_minutes,
    fit_train_transforms,
    load_npz,
)
from model import MinuteRainTransformer


class DynamicDryDownsampleSampler(Sampler[int]):
    """Keep every rainy minute and refresh a bounded dry subset each epoch."""
    def __init__(self, dataset: MinuteRainDataset, rain_threshold: float,
                 max_dry_to_rain_ratio: float, seed: int = 42):
        self.rain_indices = np.asarray([
            index for index, sample in enumerate(dataset.samples)
            if float(sample["minute_rainfall_mm"]) > rain_threshold
        ], dtype=np.int64)
        self.dry_indices = np.asarray([
            index for index, sample in enumerate(dataset.samples)
            if float(sample["minute_rainfall_mm"]) <= rain_threshold
        ], dtype=np.int64)
        if len(self.rain_indices) == 0:
            raise ValueError("Training split has no rainy samples")
        self.dry_per_epoch = min(
            len(self.dry_indices),
            int(round(len(self.rain_indices) * max_dry_to_rain_ratio)),
        )
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rain_indices) + self.dry_per_epoch

    def __iter__(self):
        dry = self.rng.choice(self.dry_indices, size=self.dry_per_epoch, replace=False)
        indices = np.concatenate([self.rain_indices, dry])
        self.rng.shuffle(indices)
        return iter(indices.tolist())


def serialize_transforms(transforms) -> dict:
    """Store preprocessing state using only weights-only-safe primitives."""
    dry_satellites = sorted(transforms.dry_by_satellite)
    dry_values = np.stack(
        [transforms.dry_by_satellite[satellite] for satellite in dry_satellites]
    ) if dry_satellites else np.empty((0, 4), dtype=np.float32)
    satellite_pairs = sorted(transforms.satellite_to_index.items())
    return {
        "feature_mean": torch.from_numpy(transforms.feature_mean),
        "feature_std": torch.from_numpy(transforms.feature_std),
        "satellite_ids": torch.tensor([pair[0] for pair in satellite_pairs], dtype=torch.int64),
        "satellite_indices": torch.tensor([pair[1] for pair in satellite_pairs], dtype=torch.int64),
        "dry_satellite_ids": torch.tensor(dry_satellites, dtype=torch.int64),
        "dry_values": torch.from_numpy(dry_values.astype(np.float32)),
        "global_dry": torch.from_numpy(transforms.global_dry),
    }


def load_local_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a checkpoint produced locally, including pre-2.6 object checkpoints."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # PyTorch 2.0 supports the legacy behavior but not this keyword.
        return torch.load(path, map_location=device)


def deserialize_transforms(state: dict) -> TrainTransforms:
    satellite_to_index = dict(zip(
        state["satellite_ids"].cpu().tolist(),
        state["satellite_indices"].cpu().tolist(),
    ))
    dry_by_satellite = {
        int(satellite): value.cpu().numpy().astype(np.float32)
        for satellite, value in zip(
            state["dry_satellite_ids"].cpu().tolist(), state["dry_values"]
        )
    }
    return TrainTransforms(
        feature_mean=state["feature_mean"].cpu().numpy().astype(np.float32),
        feature_std=state["feature_std"].cpu().numpy().astype(np.float32),
        satellite_to_index=satellite_to_index,
        dry_by_satellite=dry_by_satellite,
        global_dry=state["global_dry"].cpu().numpy().astype(np.float32),
    )


def initialize_from_checkpoint(
    model: MinuteRainTransformer,
    checkpoint: dict,
    source_transforms: TrainTransforms,
    target_transforms: TrainTransforms,
) -> dict[str, int]:
    """Transfer shared weights and raw-satellite embeddings by satellite ID."""
    source_state = checkpoint["model"]
    target_state = model.state_dict()
    copied_tensors = 0
    copied_parameters = 0
    for name, value in source_state.items():
        if name == "satellite_embedding.weight":
            continue
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name].copy_(value)
            copied_tensors += 1
            copied_parameters += value.numel()
    source_embedding = source_state["satellite_embedding.weight"]
    target_embedding = target_state["satellite_embedding.weight"]
    target_embedding[0].copy_(source_embedding[0])
    mapped_satellites = 0
    for satellite, target_index in target_transforms.satellite_to_index.items():
        source_index = source_transforms.satellite_to_index.get(satellite)
        if source_index is None or source_index >= len(source_embedding):
            continue
        target_embedding[target_index].copy_(source_embedding[source_index])
        mapped_satellites += 1
    model.load_state_dict(target_state)
    return {
        "copied_tensors": copied_tensors,
        "copied_parameters": copied_parameters,
        "mapped_satellite_embeddings": mapped_satellites,
    }


def configure_transfer_freeze(model: MinuteRainTransformer, mode: str) -> None:
    if mode == "full":
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "encoder_frozen":
        trainable_prefixes = (
            "summary_token", "position_embedding", "numeric_projection",
            "satellite_embedding", "fusion", "output_norm",
            "rain_classifier", "amount_head",
        )
    elif mode == "heads_only":
        trainable_prefixes = ("output_norm", "rain_classifier", "amount_head")
    else:
        raise ValueError(f"unsupported transfer freeze mode: {mode}")
    for name, parameter in model.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad = True


def evaluate(model, loader, device, rain_threshold: float = 0.005,
             probability_threshold: float = 0.5):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["features"].to(device), batch["satellite_ids"].to(device),
                           batch["valid_mask"].to(device), batch["raw_snr_db"].to(device))
            for idx, prediction in enumerate(output["prediction"].cpu().numpy()):
                valid = batch["valid_mask"][idx]
                quality = output["quality_weight"][idx].cpu()[valid]
                effective = output["attention_valid_mask"][idx].cpu()[valid]
                rows.append({
                    "anchor_time": pd.to_datetime(int(batch["anchor_time_ns"][idx]), unit="ns"),
                    "true_minute_rainfall_mm": float(batch["target"][idx]),
                    "pred_minute_rainfall_mm": float(prediction),
                    "rain_probability": float(torch.sigmoid(output["rain_logit"][idx]).cpu()),
                    "phy_point_count": int(batch["point_count"][idx]),
                    "effective_phy_point_count": int(effective.sum()),
                    "mean_snr_quality_weight": float(quality.mean()),
                    "satellite_count": int(batch["satellite_count"][idx]),
                })
    frame = pd.DataFrame(rows)
    # Gauge resolution is reported to two decimal places after rainfall / 10.
    frame["true_minute_rainfall_mm"] = frame["true_minute_rainfall_mm"].round(2)
    true = frame.true_minute_rainfall_mm.to_numpy()
    prediction = frame.pred_minute_rainfall_mm.to_numpy()
    probability = frame.rain_probability.to_numpy()
    rainy = true > rain_threshold
    dry = ~rainy
    predicted_rain = probability >= probability_threshold
    tp = int((predicted_rain & rainy).sum())
    fp = int((predicted_rain & dry).sum())
    fn = int((~predicted_rain & rainy).sum())
    tn = int((~predicted_rain & dry).sum())
    rainy_mae = float(np.abs(true[rainy] - prediction[rainy]).mean()) if rainy.any() else 0.0
    dry_mae = float(np.abs(true[dry] - prediction[dry]).mean()) if dry.any() else 0.0
    metrics = {
        "mae": float(np.abs(true - prediction).mean()),
        "mse": float(((true - prediction) ** 2).mean()),
        "balanced_mae": 0.5 * (rainy_mae + dry_mae),
        "rainy_mae": rainy_mae,
        "dry_mae": dry_mae,
        "n": len(frame),
        "rainy_n": int(rainy.sum()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    frame["true_minute_rainfall_mm"] = frame["true_minute_rainfall_mm"].map(
        lambda value: f"{value:.2f}"
    )
    return frame, metrics


def rainy_predictions(frame: pd.DataFrame, rain_threshold: float) -> pd.DataFrame:
    """Return an auditable rainy-only view with direct regression errors."""
    result = frame.copy()
    true = pd.to_numeric(result["true_minute_rainfall_mm"], errors="coerce")
    prediction = pd.to_numeric(result["pred_minute_rainfall_mm"], errors="coerce")
    result = result.loc[true > rain_threshold].copy()
    true = true.loc[result.index]
    prediction = prediction.loc[result.index]
    result.insert(
        result.columns.get_loc("pred_minute_rainfall_mm") + 1,
        "abs_error_mm", (true - prediction).abs(),
    )
    result.insert(
        result.columns.get_loc("abs_error_mm") + 1,
        "signed_error_mm", prediction - true,
    )
    result["true_minute_rainfall_mm"] = true.map(lambda value: f"{value:.2f}")
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minute-level rainfall retrieval")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--rain-threshold", type=float, default=0.005)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-metric", choices=("mae", "balanced_mae", "rainy_mae"),
        default="balanced_mae",
    )
    parser.add_argument("--heavy-rain-threshold", type=float, default=0.1)
    parser.add_argument("--heavy-rain-loss-weight", type=float, default=2.0)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument(
        "--snr-quality-mode", choices=("none", "hard_mask", "soft_gate"), default="none",
        help="none: unchanged; hard_mask: exclude low-SNR tokens; soft_gate: attenuate link and dry-delta features",
    )
    parser.add_argument("--snr-threshold-db", type=float, default=-10.0)
    parser.add_argument("--snr-gate-temperature-db", type=float, default=2.0)
    parser.add_argument(
        "--max-train-dry-ratio", type=float, default=-1.0,
        help=">=0 keeps all rainy samples and at most this many dry samples per rainy sample; 0 is rain-only",
    )
    parser.add_argument(
        "--evaluate-only", type=int, choices=(0, 1), default=0,
        help="1: skip training and export predictions from output-dir/best.pt",
    )
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--transfer-scaling", choices=("source", "target"), default="source",
        help="source preserves the 001 feature scaling while fitting target dry baselines and satellite IDs",
    )
    parser.add_argument(
        "--transfer-freeze", choices=("full", "encoder_frozen", "heads_only"),
        default="full",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    samples, splits = load_npz(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    checkpoint = None
    model_args = vars(args)
    init_checkpoint = None
    source_transforms = None
    if args.init_checkpoint:
        init_checkpoint = load_local_checkpoint(Path(args.init_checkpoint), device)
        source_transforms = deserialize_transforms(init_checkpoint["transforms"])
        model_args = init_checkpoint.get("args", model_args)
    if args.evaluate_only == 1:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"evaluate-only checkpoint not found: {checkpoint_path}")
        checkpoint = load_local_checkpoint(checkpoint_path, device)
        transforms = deserialize_transforms(checkpoint["transforms"])
        model_args = checkpoint.get("args", model_args)
    else:
        transforms = fit_train_transforms(samples, splits)
        if source_transforms is not None and args.transfer_scaling == "source":
            if transforms.input_dim != source_transforms.input_dim:
                raise ValueError(
                    f"transfer input dimensions differ: target={transforms.input_dim}, "
                    f"source={source_transforms.input_dim}"
                )
            transforms.feature_mean = source_transforms.feature_mean.copy()
            transforms.feature_std = source_transforms.feature_std.copy()
    dataset_max_points = int(model_args.get("max_points", args.max_points))
    datasets = {name: MinuteRainDataset(samples, splits, name, transforms, dataset_max_points)
                for name in ("train", "val", "test")}
    train_sampler = None
    if args.max_train_dry_ratio >= 0:
        train_sampler = DynamicDryDownsampleSampler(
            datasets["train"], args.rain_threshold, args.max_train_dry_ratio,
            seed=args.seed,
        )
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size,
            shuffle=train_sampler is None, sampler=train_sampler,
            collate_fn=collate_minutes, num_workers=0,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_minutes, num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_minutes, num_workers=0,
        ),
    }
    model = MinuteRainTransformer(
        transforms.input_dim, len(transforms.satellite_to_index), model_args.get("d_model", 192),
        model_args.get("num_heads", 8), model_args.get("num_layers", 3),
        model_args.get("d_ff", 512), max_points=model_args.get("max_points", 256),
        snr_quality_mode=model_args.get("snr_quality_mode", "none"),
        snr_threshold_db=model_args.get("snr_threshold_db", -10.0),
        snr_gate_temperature_db=model_args.get("snr_gate_temperature_db", 2.0),
    ).to(device)
    if args.evaluate_only == 0 and init_checkpoint is not None:
        transfer_summary = initialize_from_checkpoint(
            model, init_checkpoint, source_transforms, transforms
        )
        configure_transfer_freeze(model, args.transfer_freeze)
        print(f"transfer_initialization={json.dumps(transfer_summary)} "
              f"freeze={args.transfer_freeze} scaling={args.transfer_scaling}")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} trainable_parameters={trainable:,} "
        f"total_parameters={total_parameters:,} input_dim={transforms.input_dim}"
    )
    if args.evaluate_only == 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        rain_count = sum(float(sample["minute_rainfall_mm"]) > args.rain_threshold
                         for sample in datasets["train"].samples)
        full_dry_count = len(datasets["train"]) - rain_count
        sampled_dry_count = (
            train_sampler.dry_per_epoch if train_sampler is not None else full_dry_count
        )
        effective_dry_ratio = sampled_dry_count / max(rain_count, 1)
        pos_weight = torch.tensor(max(effective_dry_ratio, 1.0), device=device)
        print(
            f"train_sampling rainy={rain_count} dry_full={full_dry_count} "
            f"dry_per_epoch={sampled_dry_count} dry_to_rain={effective_dry_ratio:.3f} "
            f"classification_pos_weight={float(pos_weight):.3f}"
        )
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        smooth_l1 = nn.SmoothL1Loss()
        best_loss, stale = float("inf"), 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in loaders["train"]:
                target = batch["target"].to(device)
                output = model(batch["features"].to(device), batch["satellite_ids"].to(device),
                               batch["valid_mask"].to(device), batch["raw_snr_db"].to(device))
                rainy = (target > args.rain_threshold).float()
                classification = bce(output["rain_logit"], rainy)
                rainy_mask = rainy.bool()
                if rainy_mask.any():
                    rainy_target = target[rainy_mask]
                    amount_errors = F.smooth_l1_loss(
                        torch.log1p(output["conditional_amount"][rainy_mask] * 100),
                        torch.log1p(rainy_target * 100), reduction="none",
                    )
                    intensity_weights = torch.where(
                        rainy_target >= args.heavy_rain_threshold,
                        torch.full_like(rainy_target, args.heavy_rain_loss_weight),
                        torch.ones_like(rainy_target),
                    )
                    amount = (amount_errors * intensity_weights).sum() / intensity_weights.sum()
                else:
                    amount = output["conditional_amount"].sum() * 0
                all_sample = smooth_l1(output["prediction"], target)
                loss = all_sample + amount + args.classification_weight * classification
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            val_frame, val_metrics = evaluate(
                model, loaders["val"], device,
                args.rain_threshold, args.probability_threshold,
            )
            print(f"epoch={epoch:03d} train_loss={np.mean(losses):.6f} "
                  f"val_mae={val_metrics['mae']:.6f} "
                  f"val_rainy_mae={val_metrics['rainy_mae']:.6f} "
                  f"val_balanced_mae={val_metrics['balanced_mae']:.6f} "
                  f"val_f1={val_metrics['f1']:.4f}")
            selection_value = val_metrics[args.selection_metric]
            if selection_value < best_loss:
                best_loss, stale = selection_value, 0
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "transforms": serialize_transforms(transforms),
                }, checkpoint_path)
                val_frame.to_csv(output_dir / "best_val_predictions.csv", index=False)
            else:
                stale += 1
                if stale >= args.patience:
                    break
    if checkpoint is None:
        checkpoint = load_local_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    metrics = {}
    evaluation_frames = {}
    for name in ("train", "val", "test"):
        frame, metrics[name] = evaluate(
            model, loaders[name], device,
            args.rain_threshold, args.probability_threshold,
        )
        frame.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        evaluation_frames[name] = frame
        if name in ("val", "test"):
            rainy_predictions(frame, args.rain_threshold).to_csv(
                output_dir / f"{name}_rainy_predictions.csv", index=False
            )
    combined_rainy = []
    for split_name in ("val", "test"):
        split_frame = rainy_predictions(
            evaluation_frames[split_name], args.rain_threshold
        )
        split_frame.insert(1, "split", split_name)
        combined_rainy.append(split_frame)
    pd.concat(combined_rainy, ignore_index=True).sort_values("anchor_time").to_csv(
        output_dir / "val_test_rainy_predictions.csv", index=False
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
