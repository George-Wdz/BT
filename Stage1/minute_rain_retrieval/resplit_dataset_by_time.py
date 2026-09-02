#!/usr/bin/env python3
"""Apply explicit chronological train/validation/test boundaries to an NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_flow import save_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-end", required=True)
    args = parser.parse_args()

    archive = np.load(args.input_path, allow_pickle=True)
    samples = archive["samples"].tolist()
    timestamps = pd.to_datetime(
        [int(sample["anchor_time_ns"]) for sample in samples], unit="ns"
    )
    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)
    if val_end <= train_end:
        parser.error("--val-end must be later than --train-end")
    splits = np.where(
        timestamps < train_end,
        "train",
        np.where(timestamps < val_end, "val", "test"),
    )
    summary = json.loads(str(archive["summary_json"].item()))
    summary["split_strategy"] = "explicit_time_boundaries"
    summary["split_boundaries"] = {
        "train_end_exclusive": train_end.isoformat(),
        "val_end_exclusive": val_end.isoformat(),
    }
    summary["split_counts"] = {
        name: int((splits == name).sum()) for name in ("train", "val", "test")
    }
    summary["rainy_split_counts"] = {
        name: int(sum(
            float(sample["minute_rainfall_mm"]) > 0
            for sample, split in zip(samples, splits) if split == name
        ))
        for name in ("train", "val", "test")
    }
    for name in ("train", "val", "test"):
        if summary["split_counts"][name] == 0:
            raise ValueError(f"empty split: {name}")
        if summary["rainy_split_counts"][name] == 0:
            raise ValueError(f"split has no rainy sample: {name}")
    metadata = {
        "splits": splits.astype("<U5"),
        "summary": summary,
        "feature_columns": archive["feature_columns"].tolist(),
    }
    save_dataset(samples, metadata, str(args.output_path))
    print(json.dumps({
        "output_path": str(args.output_path),
        "split_counts": summary["split_counts"],
        "rainy_split_counts": summary["rainy_split_counts"],
        "split_boundaries": summary["split_boundaries"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
