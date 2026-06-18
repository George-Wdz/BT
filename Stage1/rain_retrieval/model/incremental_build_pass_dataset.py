#!/usr/bin/env python3
"""Incrementally update a Stage1 pass_dataset NPZ.

The script reads an existing NPZ, builds only a recent DB window with overlap,
then merges passes by (satellite_id, pass_start). This avoids rebuilding from
scratch when there is no new DB data and keeps both workflow scripts using the
same incremental behavior.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.preprocessing import (  # noqa: E402
    build_pass_dataset,
    dataset_summary,
    pass_index_frame,
    save_dataset_artifacts,
)


def latest_db_time(db_path: str) -> pd.Timestamp | None:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT max(localTime) FROM phy_data").fetchone()
    if not row or not row[0]:
        return None
    return pd.Timestamp(row[0])


def pass_start(p: dict) -> pd.Timestamp:
    return pd.DatetimeIndex(p["timestamps"])[0]


def pass_end(p: dict) -> pd.Timestamp:
    return pd.DatetimeIndex(p["timestamps"])[-1]


def pass_key(p: dict) -> tuple[int, str]:
    # Rebuilt overlapping passes should replace older copies that share the
    # same satellite and start time. End time may extend as new rows arrive.
    return int(p["satellite_id"]), pass_start(p).isoformat()


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    npz = np.load(path, allow_pickle=True)
    return list(npz["passes"])


def merge_passes(old: list[dict], new: list[dict]) -> list[dict]:
    merged = {pass_key(p): p for p in old}
    for p in new:
        merged[pass_key(p)] = p
    return sorted(merged.values(), key=lambda p: (pass_start(p), int(p["satellite_id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="/home/wdz/satellite_data/satellite_data.db")
    parser.add_argument("--existing-npz", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--image-csv", required=True)
    parser.add_argument("--image-tolerance", default="10min")
    parser.add_argument("--lookback-minutes", type=float, default=20.0)
    parser.add_argument("--strict-source-filters", action="store_true")
    args = parser.parse_args()

    existing_path = Path(args.existing_npz).expanduser()
    output_path = Path(args.output_path).expanduser()
    old_passes = load_existing(existing_path)
    db_latest = latest_db_time(args.db_path)

    feature_cols = {
        "link": ["phyRssi", "rssi", "snr", "lastCniValue"],
        "position": [
            "longitude",
            "latitude",
            "satAltitude",
            "posLongitude",
            "posLatitude",
            "altitude",
        ],
        "ground_weather": ["temperature", "humidity", "pressure"],
    }
    image_weather_cfg = {
        "enabled": True,
        "csv_path": args.image_csv,
        "tolerance": args.image_tolerance,
    }

    if not old_passes:
        print("No existing NPZ found; building full dataset.")
        dataset = build_pass_dataset(
            db_path=args.db_path,
            output_path=str(output_path),
            feature_cols=feature_cols,
            strict_source_filters=args.strict_source_filters,
            image_weather_cfg=image_weather_cfg,
        )
        print(json.dumps({"mode": "full", "passes": len(dataset)}, ensure_ascii=False))
        return

    old_latest = max(pass_end(p) for p in old_passes)
    if db_latest is not None and db_latest <= old_latest:
        print(f"No new phy_data rows after existing dataset end: {old_latest}")
        save_dataset_artifacts(
            old_passes,
            args.db_path,
            str(output_path),
            {
                "incremental": {
                    "mode": "reuse_existing_no_new_rows",
                    "existing_npz": str(existing_path),
                    "old_latest_pass_end": str(old_latest),
                    "db_latest_phy_time": str(db_latest),
                }
            },
            feature_cols,
        )
        print(json.dumps({"mode": "reuse", "passes": len(old_passes)}, ensure_ascii=False))
        return

    start = old_latest - pd.Timedelta(minutes=args.lookback_minutes)
    print(f"Incremental DB window: {start} ~ latest")
    new_passes = build_pass_dataset(
        db_path=args.db_path,
        output_path=None,
        feature_cols=feature_cols,
        strict_source_filters=args.strict_source_filters,
        image_weather_cfg=image_weather_cfg,
        start_time=start.isoformat(),
        end_time=None,
    )
    merged = merge_passes(old_passes, new_passes)
    save_dataset_artifacts(
        merged,
        args.db_path,
        str(output_path),
        {
            "incremental": {
                "mode": "merge_new_window",
                "existing_npz": str(existing_path),
                "old_passes": len(old_passes),
                "new_window_passes": len(new_passes),
                "merged_passes": len(merged),
                "old_latest_pass_end": str(old_latest),
                "db_latest_phy_time": str(db_latest),
                "window_start": str(start),
                "lookback_minutes": args.lookback_minutes,
            }
        },
        feature_cols,
    )
    print(json.dumps({
        "mode": "incremental",
        "old_passes": len(old_passes),
        "new_window_passes": len(new_passes),
        "merged_passes": len(merged),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
