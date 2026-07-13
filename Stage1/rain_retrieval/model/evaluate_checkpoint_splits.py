#!/usr/bin/env python3
"""Evaluate Stage1 checkpoints on train/val/test with rainy/dry breakdowns."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.data_factory import (
    _optional_feature_keys,
    attach_train_dry_baseline,
    enabled_feature_groups,
    feature_group_columns,
    feature_group_dims,
    load_all_passes,
    split_passes_by_time,
)
from data.dataset import PassDataset, SatelliteIDMapper
from patch_encoder_decoder import PatchEncoderDecoder


def _coerce_numeric(cfg: dict) -> None:
    b = cfg.get("dry_baseline", {})
    for k in (
        "rain_threshold",
        "image_rain_prob_threshold",
        "min_sunny_prob",
        "time_scale_hours",
        "time_weight",
        "position_weight",
    ):
        if isinstance(b.get(k), str):
            b[k] = float(b[k])
    t = cfg["training"]
    for k in (
        "lr",
        "weight_decay",
        "rainfall_loss_weight",
        "rain_threshold",
        "rainy_loss_weight",
        "rain_classification_loss_weight",
        "rain_classification_pos_weight",
        "grad_clip",
        "decay_fac",
    ):
        if k in t and isinstance(t[k], str):
            t[k] = float(t[k])
    for k in ("epochs", "batch_size", "warmup_epochs", "patience",
              "iterations", "seed", "tmax"):
        if k in t and isinstance(t[k], str):
            t[k] = int(t[k])


def _load_checkpoint(ckpt_dir: Path, device: torch.device):
    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(meta["cfg"])
    _coerce_numeric(cfg)

    sat_mapper = SatelliteIDMapper(known_ids=[])
    sat_mapper.id_to_idx = meta["sat_mapper"]
    sat_mapper.num_satellites = max(sat_mapper.id_to_idx.values()) + 1 if sat_mapper.id_to_idx else 1
    cfg["model"]["num_satellites"] = max(cfg["model"]["num_satellites"], sat_mapper.num_satellites)

    model = PatchEncoderDecoder(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pth", map_location=device))
    model.eval()
    return cfg, model, sat_mapper, meta["scaler_X"], meta["scaler_y"]


def _predict(model, dataset: PassDataset, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = []
    trues = []
    probs = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            sat_idx = batch["satellite_idx"].to(device).long()
            condition = batch["condition"].to(device)
            quality = batch["modal_quality"].to(device)
            rain_pred, _, rain_logit = model(features, mask, sat_idx, condition, quality)
            preds.append(rain_pred.detach().cpu().numpy().reshape(-1))
            probs.append(torch.sigmoid(rain_logit).cpu().numpy().reshape(-1))
            trues.append(batch["labels_phys"][:, 0].detach().cpu().numpy().reshape(-1))
    return (
        np.concatenate(preds).astype(np.float64),
        np.concatenate(trues).astype(np.float64),
        np.concatenate(probs).astype(np.float64),
    )


def _report(name: str, pred: np.ndarray, true: np.ndarray, threshold: float) -> None:
    if len(true) == 0:
        print(f"{name}: empty")
        return
    err = np.abs(pred - true)
    rainy = true > threshold
    dry = ~rainy
    print(
        f"{name}: n={len(true)} rainy={int(rainy.sum())} "
        f"mae={err.mean():.4f} mse={(err ** 2).mean():.4f} "
        f"true_mean={true.mean():.4f} pred_mean={pred.mean():.4f} "
        f"true_max={true.max():.4f} pred_max={pred.max():.4f}"
    )
    if rainy.any():
        re = err[rainy]
        print(
            f"  rainy: n={int(rainy.sum())} mae={re.mean():.4f} "
            f"true_mean={true[rainy].mean():.4f} pred_mean={pred[rainy].mean():.4f} "
            f"true_max={true[rainy].max():.4f} pred_max={pred[rainy].max():.4f}"
        )
    if dry.any():
        de = err[dry]
        print(
            f"  dry:   n={int(dry.sum())} mae={de.mean():.4f} "
            f"pred_mean={pred[dry].mean():.4f} pred_max={pred[dry].max():.4f}"
        )


def _metric_row(split: str, subset: str, pred: np.ndarray, true: np.ndarray) -> dict:
    if len(true) == 0:
        return {
            "split": split,
            "subset": subset,
            "n": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "true_mean": np.nan,
            "pred_mean": np.nan,
            "true_max": np.nan,
            "pred_max": np.nan,
        }
    err = pred - true
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    return {
        "split": split,
        "subset": subset,
        "n": int(len(true)),
        "mae": float(np.mean(abs_err)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "true_mean": float(np.mean(true)),
        "pred_mean": float(np.mean(pred)),
        "true_max": float(np.max(true)),
        "pred_max": float(np.max(pred)),
    }


def _metric_rows(split: str, pred: np.ndarray, true: np.ndarray, threshold: float) -> list[dict]:
    rainy = true > threshold
    dry = ~rainy
    return [
        _metric_row(split, "all", pred, true),
        _metric_row(split, "rainy", pred[rainy], true[rainy]),
        _metric_row(split, "dry", pred[dry], true[dry]),
    ]


def _classification_row(split: str, prob: np.ndarray, true: np.ndarray,
                        rain_threshold: float, probability_threshold: float) -> dict:
    actual = true > rain_threshold
    predicted = prob >= probability_threshold
    tp = int(np.sum(actual & predicted)); fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted)); tn = int(np.sum(~actual & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if actual.any() and (~actual).any():
        pr_auc = float(average_precision_score(actual, prob))
        roc_auc = float(roc_auc_score(actual, prob))
    else:
        pr_auc = roc_auc = np.nan
    return {
        "split": split, "subset": "rain_classification", "n": int(len(true)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "pr_auc": pr_auc, "roc_auc": roc_auc,
    }


def _slice_rows(split: str, pred: np.ndarray, true: np.ndarray,
                passes: list, threshold: float) -> list[dict]:
    rows = []
    # Rainfall severity bins, with the first boundary tied to the configured dry threshold.
    edges = [threshold, 0.1, 1.0, 5.0, 20.0, np.inf]
    labels = ["trace", "light", "moderate", "heavy", "extreme"]
    for label, lo, hi in zip(labels, edges[:-1], edges[1:]):
        keep = (true > lo) & (true <= hi)
        row = _metric_row(split, f"rain_bin:{label}", pred[keep], true[keep])
        rows.append(row)
    sat_ids = np.asarray([int(p["satellite_id"]) for p in passes])
    for sat_id in np.unique(sat_ids):
        keep = sat_ids == sat_id
        rows.append(_metric_row(split, f"satellite:{sat_id}", pred[keep], true[keep]))
    image_available = np.asarray([
        int(p.get("label_meta", {}).get("image_available", 0) or 0) for p in passes
    ])
    for available in (0, 1):
        keep = image_available == available
        rows.append(_metric_row(split, f"image_available:{available}", pred[keep], true[keep]))
    # Geometry slices are emitted only when the NPZ contains derived geometry columns.
    elevations = np.full(len(passes), np.nan)
    for i, p in enumerate(passes):
        cols = p.get("feature_columns", {}).get("position", [])
        pos = np.asarray(p.get("position_features", []), dtype=np.float64)
        if "elevation_deg" in cols and pos.ndim == 2:
            elevations[i] = np.nanmean(pos[:, cols.index("elevation_deg")])
    for label, lo, hi in (("low", -90, 20), ("mid", 20, 50), ("high", 50, 90.0001)):
        keep = np.isfinite(elevations) & (elevations >= lo) & (elevations < hi)
        if keep.any():
            rows.append(_metric_row(split, f"elevation:{label}", pred[keep], true[keep]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--rain-threshold", type=float, default=1e-6)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-csv", default="", help="Optional path to save per-pass predictions.")
    parser.add_argument("--rainy-csv", default="", help="Optional path to save rainy per-pass predictions.")
    parser.add_argument("--test-csv", default="", help="Optional path to save compact test-set predictions.")
    parser.add_argument("--metrics-csv", default="", help="Optional path to save MAE/MSE/RMSE metrics.")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, sat_mapper, scaler_X, scaler_y = _load_checkpoint(ckpt_dir, device)

    passes = load_all_passes(cfg)
    train_passes, val_passes, test_passes = split_passes_by_time(
        passes,
        cfg["data"]["data_split"],
        val_strategy=cfg["data"].get("val_strategy", "time"),
        seed=cfg["training"].get("seed", 42),
    )
    train_passes, val_passes, test_passes = attach_train_dry_baseline(
        train_passes, val_passes, test_passes, cfg
    )

    print(f"checkpoint={ckpt_dir}")
    print(f"device={device}")
    print(f"input_dim={cfg['model']['input_dim']} channel_attention={cfg['model'].get('use_channel_attention', False)}")
    print(f"rain_filter_min={cfg['data'].get('rain_filter_min', 0.0)}")

    csv_rows = []
    metrics_rows = []
    needs_pass_rows = bool(args.out_csv or args.rainy_csv or args.test_csv)
    for name, split in (("train", train_passes), ("val", val_passes), ("test", test_passes)):
        ds = PassDataset(
            split,
            sat_mapper,
            max_len=cfg["model"]["max_seq_len"],
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            fit_scalers=False,
            extra_feature_keys=_optional_feature_keys(cfg),
            feature_groups=enabled_feature_groups(cfg),
            feature_group_dims=feature_group_dims(cfg),
            feature_group_columns=feature_group_columns(cfg),
            target_names=list(cfg["targets"]["primary"]) + list(cfg["targets"].get("auxiliary", [])),
        )
        pred, true, rain_prob = _predict(model, ds, device, args.batch_size)
        _report(name, pred, true, args.rain_threshold)
        metrics_rows.extend(_metric_rows(name, pred, true, args.rain_threshold))
        metrics_rows.append(_classification_row(
            name, rain_prob, true, args.rain_threshold, args.probability_threshold
        ))
        metrics_rows.extend(_slice_rows(name, pred, true, split, args.rain_threshold))
        if needs_pass_rows:
            for p, y_pred, y_true, y_prob in zip(split, pred, true, rain_prob):
                ts = pd.DatetimeIndex(p["timestamps"])
                row = {
                    "split": name,
                    "satellite_id": int(p["satellite_id"]),
                    "pass_start": ts[0],
                    "pass_end": ts[-1],
                    "true_rainfall_mm": float(y_true),
                    "pred_rainfall_mm": float(y_pred),
                    "abs_error_mm": float(abs(y_pred - y_true)),
                    "is_rainy": bool(y_true > args.rain_threshold),
                    "pred_rain_probability": float(y_prob),
                }
                link = np.asarray(p["link_features"], dtype=np.float64)
                for i, col in enumerate(cfg.get("features", {}).get("link", [])):
                    if i < link.shape[1]:
                        row[f"{col}_mean"] = float(np.nanmean(link[:, i]))
                        row[f"{col}_min"] = float(np.nanmin(link[:, i]))
                        row[f"{col}_max"] = float(np.nanmax(link[:, i]))
                image = p.get("image_weather")
                if image is not None:
                    image = np.asarray(image, dtype=np.float64)
                    if image.ndim == 2 and image.shape[1] >= 4:
                        row["prob_sunny"] = float(np.nanmean(image[:, 0]))
                        row["prob_cloudy"] = float(np.nanmean(image[:, 1]))
                        row["prob_rain"] = float(np.nanmean(image[:, 2]))
                        row["image_available"] = int(np.nanmax(image[:, 3]) > 0)
                meta = p.get("label_meta", {})
                row.update({
                    "rain_rate_mean": float(meta.get("rain_rate_mean", 0.0) or 0.0),
                    "rain_rate_max": float(meta.get("rain_rate_max", 0.0) or 0.0),
                    "rainy_ratio": float(meta.get("rainy_ratio", 0.0) or 0.0),
                })
                csv_rows.append({
                    k: v for k, v in row.items()
                })

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(out_path, index=False)
        print(f"saved_predictions_csv={out_path}")

    if args.rainy_csv:
        rainy_path = Path(args.rainy_csv)
        rainy_path.parent.mkdir(parents=True, exist_ok=True)
        rainy_df = pd.DataFrame(csv_rows)
        rainy_df = rainy_df[rainy_df["is_rainy"]].copy()
        preferred = [
            "split",
            "satellite_id",
            "pass_start",
            "pass_end",
            "true_rainfall_mm",
            "pred_rainfall_mm",
            "abs_error_mm",
            "rain_rate_mean",
            "rain_rate_max",
            "rainy_ratio",
            "prob_sunny",
            "prob_cloudy",
            "prob_rain",
            "image_available",
        ]
        cols = [c for c in preferred if c in rainy_df.columns]
        cols += [c for c in rainy_df.columns if c not in cols]
        rainy_df[cols].to_csv(rainy_path, index=False)
        print(f"saved_rainy_predictions_csv={rainy_path}")

    if args.test_csv:
        test_path = Path(args.test_csv)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame(csv_rows)
        test_df = test_df[test_df["split"] == "test"].copy()
        cols = [
            "satellite_id",
            "pass_start",
            "pass_end",
            "true_rainfall_mm",
            "pred_rainfall_mm",
            "abs_error_mm",
        ]
        test_df[cols].to_csv(test_path, index=False)
        print(f"saved_test_predictions_csv={test_path}")

    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(f"saved_metrics_csv={metrics_path}")


if __name__ == "__main__":
    main()
