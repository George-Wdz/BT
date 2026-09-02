#!/usr/bin/env python3
"""Create a reproducible feature-ablation copy of a minute NPZ dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_flow import IMAGE_COLUMNS, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--disable-image", action="store_true")
    args = parser.parse_args()
    archive = np.load(args.input_path, allow_pickle=True)
    samples = archive["samples"].tolist()
    columns = archive["feature_columns"].tolist()
    disabled: list[str] = []
    if args.disable_image:
        indices = [columns.index(column) for column in IMAGE_COLUMNS]
        for sample in samples:
            features = np.asarray(sample["features"], dtype=np.float32).copy()
            features[:, indices] = 0.0
            sample["features"] = features
        disabled.extend(IMAGE_COLUMNS)
    summary = json.loads(str(archive["summary_json"].item()))
    summary["disabled_feature_columns"] = disabled
    metadata = {
        "splits": archive["splits"].astype(str),
        "summary": summary,
        "feature_columns": columns,
    }
    save_dataset(samples, metadata, str(args.output_path))
    print(json.dumps({
        "output_path": str(args.output_path),
        "samples": len(samples),
        "disabled_feature_columns": disabled,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
