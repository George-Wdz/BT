#!/usr/bin/env python3
"""Derive point-count ablations from one NPZ while preserving split membership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="minute_rainfall")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[3, 5, 8, 10])
    args = parser.parse_args()

    source = np.load(args.source, allow_pickle=True)
    samples = source["samples"]
    splits = source["splits"]
    feature_columns = source["feature_columns"]
    source_summary = json.loads(str(source["summary_json"].item()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for threshold in args.thresholds:
        keep = np.asarray(
            [int(sample["point_count"]) >= threshold for sample in samples], dtype=bool
        )
        selected_samples = samples[keep]
        selected_splits = splits[keep]
        rainy = np.asarray(
            [float(sample["minute_rainfall_mm"]) > 0 for sample in selected_samples]
        )
        summary = dict(source_summary)
        summary["derived_from"] = str(Path(args.source).resolve())
        summary["config"] = dict(summary.get("config", {}))
        summary["config"]["min_phy_points"] = threshold
        summary["samples"] = int(len(selected_samples))
        summary["rainy_samples"] = int(rainy.sum())
        summary["split_counts"] = {
            name: int((selected_splits == name).sum()) for name in ("train", "val", "test")
        }
        summary["rainy_split_counts"] = {
            name: int((rainy & (selected_splits == name)).sum())
            for name in ("train", "val", "test")
        }
        path = output_dir / f"{args.prefix}_minphy{threshold}.npz"
        np.savez_compressed(
            path,
            samples=selected_samples,
            splits=selected_splits,
            feature_columns=feature_columns,
            summary_json=np.asarray(json.dumps(summary, ensure_ascii=False)),
        )
        path.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        index = pd.DataFrame({
            "anchor_time": [
                pd.to_datetime(int(sample["anchor_time_ns"]), unit="ns")
                for sample in selected_samples
            ],
            "split": selected_splits,
            "minute_rainfall_mm": [
                f"{float(sample['minute_rainfall_mm']):.2f}" for sample in selected_samples
            ],
            "phy_point_count": [int(sample["point_count"]) for sample in selected_samples],
            "satellite_count": [int(sample["satellite_count"]) for sample in selected_samples],
        })
        index.to_csv(path.with_suffix(".index.csv"), index=False)
        report.append({"threshold": threshold, **summary["split_counts"],
                       "samples": len(selected_samples), "rainy": int(rainy.sum())})
    print(pd.DataFrame(report).to_string(index=False))


if __name__ == "__main__":
    main()
