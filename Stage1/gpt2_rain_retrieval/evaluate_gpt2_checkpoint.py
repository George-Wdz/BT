#!/usr/bin/env python3
"""Evaluate GPT2 rainfall checkpoints on train/val/test splits."""
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

ROOT = Path(__file__).resolve().parent
STAGE1_MODEL_ROOT = ROOT.parent / "rain_retrieval" / "model"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE1_MODEL_ROOT))

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
from gpt2_rain_model import GPT2RainRegressor


def _coerce_numeric(cfg: dict) -> None:
    b = cfg.get("dry_baseline", {})
    for key in (
        "rain_threshold",
        "image_rain_prob_threshold",
        "min_sunny_prob",
        "time_scale_hours",
        "time_weight",
        "position_weight",
    ):
        if isinstance(b.get(key), str):
            b[key] = float(b[key])
    t = cfg["training"]
    for key in (
        "lr",
        "weight_decay",
        "rainfall_loss_weight",
        "rain_threshold",
        "rainy_loss_weight",
        "rain_classification_loss_weight",
        "rain_classification_pos_weight",
        "auxiliary_loss_weight",
        "grad_clip",
        "decay_fac",
    ):
        if key in t and isinstance(t[key], str):
            t[key] = float(t[key])
    for key in ("epochs", "batch_size", "patience", "iterations", "seed", "tmax"):
        if key in t and isinstance(t[key], str):
            t[key] = int(t[key])


def _load_checkpoint(ckpt_dir: Path, device: torch.device):
    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(meta["cfg"])
    _coerce_numeric(cfg)
    sat_mapper = SatelliteIDMapper(known_ids=[])
    sat_mapper.id_to_idx = meta["sat_mapper"]
    sat_mapper.num_satellites = max(sat_mapper.id_to_idx.values()) + 1 if sat_mapper.id_to_idx else 1
    cfg["model"]["num_satellites"] = max(int(cfg["model"]["num_satellites"]), sat_mapper.num_satellites)
    model = GPT2RainRegressor(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pth", map_location=device))
    model.eval()
    return cfg, model, sat_mapper, meta["scaler_X"], meta["scaler_y"]


def _predict(model, dataset: PassDataset, device: torch.device, batch_size: int):
    preds, trues = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device).bool()
            sat_idx = batch["satellite_idx"].to(device).long()
            rain_pred, _, _ = model(features, mask, sat_idx)
            preds.append(rain_pred.detach().cpu().numpy().reshape(-1))
            trues.append(batch["labels_phys"][:, 0].detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds).astype(np.float64), np.concatenate(trues).astype(np.float64)


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
    mse = float(np.mean(err ** 2))
    return {
        "split": split,
        "subset": subset,
        "n": int(len(true)),
        "mae": float(np.mean(np.abs(err))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "true_mean": float(np.mean(true)),
        "pred_mean": float(np.mean(pred)),
        "true_max": float(np.max(true)),
        "pred_max": float(np.max(pred)),
    }


def _metric_rows(split: str, pred: np.ndarray, true: np.ndarray, threshold: float):
    rainy = true > threshold
    dry = ~rainy
    return [
        _metric_row(split, "all", pred, true),
        _metric_row(split, "rainy", pred[rainy], true[rainy]),
        _metric_row(split, "dry", pred[dry], true[dry]),
    ]


def _print_report(name: str, pred: np.ndarray, true: np.ndarray, threshold: float) -> None:
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
        print(
            f"  rainy: n={int(rainy.sum())} mae={err[rainy].mean():.4f} "
            f"true_mean={true[rainy].mean():.4f} pred_mean={pred[rainy].mean():.4f} "
            f"true_max={true[rainy].max():.4f} pred_max={pred[rainy].max():.4f}"
        )
    if dry.any():
        print(
            f"  dry:   n={int(dry.sum())} mae={err[dry].mean():.4f} "
            f"pred_mean={pred[dry].mean():.4f} pred_max={pred[dry].max():.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--rain-threshold", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--metrics-csv", default="")
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
    print(
        f"input_dim={cfg['model']['input_dim']} "
        f"gpt2_layers={cfg['model'].get('gpt2_layers')} "
        f"freeze_gpt2={cfg['model'].get('freeze_gpt2')}"
    )

    csv_rows = []
    metrics_rows = []
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
        pred, true = _predict(model, ds, device, args.batch_size)
        _print_report(name, pred, true, args.rain_threshold)
        metrics_rows.extend(_metric_rows(name, pred, true, args.rain_threshold))
        if args.out_csv or args.test_csv:
            for p, y_pred, y_true in zip(split, pred, true):
                ts = pd.DatetimeIndex(p["timestamps"])
                csv_rows.append(
                    {
                        "split": name,
                        "satellite_id": int(p["satellite_id"]),
                        "pass_start": ts[0],
                        "pass_end": ts[-1],
                        "true_rainfall_mm": float(y_true),
                        "pred_rainfall_mm": float(y_pred),
                        "abs_error_mm": float(abs(y_pred - y_true)),
                        "is_rainy": bool(y_true > args.rain_threshold),
                    }
                )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(out_path, index=False)
        print(f"saved_predictions_csv={out_path}")
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
