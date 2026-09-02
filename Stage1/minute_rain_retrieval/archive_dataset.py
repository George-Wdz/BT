#!/usr/bin/env python3
"""Archive one minute-rainfall dataset with reproducible split and quality views."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from data_flow import save_dataset


POINT_KEYS = ("features", "satellite_ids", "timestamps_ns")


def load_dataset(path: Path) -> tuple[list[dict], np.ndarray, list[str], dict]:
    with np.load(path, allow_pickle=True) as data:
        samples = data["samples"].tolist()
        splits = data["splits"].astype("<U5")
        feature_columns = data["feature_columns"].astype(str).tolist()
        summary = json.loads(str(data["summary_json"].item()))
    return samples, splits, feature_columns, summary


def metadata_for(samples: list[dict], splits: np.ndarray, columns: list[str],
                 source_summary: dict, view: str) -> dict:
    summary = dict(source_summary)
    summary.update({
        "archive_view": view,
        "samples": len(samples),
        "rainy_samples": int(sum(float(item["minute_rainfall_mm"]) > 1e-6
                                  for item in samples)),
        "split_counts": {
            name: int((splits == name).sum()) for name in ("train", "val", "test")
        },
        "rainy_split_counts": {
            name: int(sum(
                float(item["minute_rainfall_mm"]) > 1e-6
                for item, split in zip(samples, splits) if split == name
            )) for name in ("train", "val", "test")
        },
    })
    return {"splits": splits, "feature_columns": columns, "summary": summary}


def archive_summary(source_summary: dict, archive_dir: Path,
                    full_path: Path) -> dict:
    summary = copy.deepcopy(source_summary)
    config = summary.setdefault("config", {})
    previous_output = config.get("output_path")
    if previous_output and previous_output != str(full_path):
        config["source_output_path"] = previous_output
    config["output_path"] = str(full_path)
    archived_images = archive_dir / "raw" / "camera_weather_labels.csv"
    if archived_images.is_file():
        previous_images = config.get("image_csv")
        if previous_images and previous_images != str(archived_images):
            config["source_image_csv"] = previous_images
        config["image_csv"] = str(archived_images)
    return summary


def export_splits(samples: list[dict], splits: np.ndarray, columns: list[str],
                  summary: dict, output_dir: Path) -> None:
    for split_name in ("train", "val", "test"):
        selected = np.flatnonzero(splits == split_name)
        split_samples = [samples[index] for index in selected]
        split_labels = splits[selected]
        metadata = metadata_for(
            split_samples, split_labels, columns, summary, f"split:{split_name}"
        )
        save_dataset(
            split_samples, metadata,
            str(output_dir / split_name / f"minute_rainfall_{split_name}.npz"),
        )


def filter_sample(sample: dict, snr_index: int, threshold: float) -> dict | None:
    features = np.asarray(sample["features"])
    keep = features[:, snr_index] >= threshold
    if not keep.any():
        return None
    filtered = dict(sample)
    for key in POINT_KEYS:
        filtered[key] = np.asarray(sample[key])[keep]
    filtered["point_count"] = np.int32(keep.sum())
    filtered["satellite_count"] = np.int16(
        np.unique(filtered["satellite_ids"]).size
    )
    return filtered


def export_quality_view(samples: list[dict], splits: np.ndarray, columns: list[str],
                        summary: dict, threshold: float, minimum_points: int,
                        output: Path, view_name: str) -> dict:
    snr_index = columns.index("snr")
    selected_samples: list[dict] = []
    selected_splits: list[str] = []
    for sample, split in zip(samples, splits):
        filtered = filter_sample(sample, snr_index, threshold)
        if filtered is not None and int(filtered["point_count"]) >= minimum_points:
            selected_samples.append(filtered)
            selected_splits.append(str(split))
    quality_splits = np.asarray(selected_splits, dtype="<U5")
    metadata = metadata_for(
        selected_samples, quality_splits, columns, summary, view_name
    )
    metadata["summary"]["snr_quality"] = {
        "point_threshold_db": threshold,
        "minimum_points_per_minute": minimum_points,
        "label_independent": True,
    }
    save_dataset(selected_samples, metadata, str(output))
    return metadata["summary"]


def annotate_quality(samples: list[dict], columns: list[str], strong_threshold: float,
                     rain_resilient_threshold: float) -> list[dict]:
    snr_index = columns.index("snr")
    annotated = []
    for sample in samples:
        item = dict(sample)
        snr = np.asarray(sample["features"])[:, snr_index]
        strong = snr >= strong_threshold
        resilient = snr >= rain_resilient_threshold
        item["snr_strong_mask"] = strong.astype(np.bool_)
        item["snr_rain_resilient_mask"] = resilient.astype(np.bool_)
        item["snr_strong_point_count"] = np.int32(strong.sum())
        item["snr_rain_resilient_point_count"] = np.int32(resilient.sum())
        annotated.append(item)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--strong-snr-threshold-db", type=float, default=-10.0)
    parser.add_argument("--rain-resilient-snr-threshold-db", type=float, default=-25.0)
    parser.add_argument("--minimum-quality-points", type=int, default=3)
    parser.add_argument("--quality-archives", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    samples, splits, columns, source_summary = load_dataset(args.dataset_path)
    archive_dir = args.archive_dir.resolve()
    processed = archive_dir / "processed"
    full_path = processed / "minute_rainfall_full.npz"
    summary = archive_summary(source_summary, archive_dir, full_path)
    save_dataset(
        samples,
        metadata_for(samples, splits, columns, summary, "full"),
        str(full_path),
    )
    export_splits(samples, splits, columns, summary, archive_dir / "splits")

    quality_summaries = {}
    if args.quality_archives:
        quality_dir = archive_dir / "high_quality"
        annotated = annotate_quality(
            samples, columns, args.strong_snr_threshold_db,
            args.rain_resilient_snr_threshold_db,
        )
        annotated_metadata = metadata_for(
            annotated, splits, columns, summary, "quality_annotated_full"
        )
        annotated_metadata["summary"]["snr_quality"] = {
            "strong_threshold_db": args.strong_snr_threshold_db,
            "rain_resilient_threshold_db": args.rain_resilient_snr_threshold_db,
            "label_independent": True,
        }
        save_dataset(
            annotated, annotated_metadata,
            str(quality_dir / "minute_rainfall_quality_annotated.npz"),
        )
        quality_summaries["strong_link"] = export_quality_view(
            samples, splits, columns, summary, args.strong_snr_threshold_db,
            args.minimum_quality_points,
            quality_dir / "strong_link_snr_ge_neg10" / "minute_rainfall_full.npz",
            "strong_link",
        )
        quality_summaries["rain_resilient"] = export_quality_view(
            samples, splits, columns, summary, args.rain_resilient_snr_threshold_db,
            args.minimum_quality_points,
            quality_dir / "rain_resilient_snr_ge_neg25" / "minute_rainfall_full.npz",
            "rain_resilient_link",
        )

    manifest = {
        "source_dataset": str(args.dataset_path.resolve()),
        "archive_dir": str(archive_dir),
        "processed_dataset": str(full_path),
        "split_strategy": summary.get("config", {}).get("split_strategy"),
        "split_seed": summary.get("config", {}).get("split_seed"),
        "samples": len(samples),
        "quality_archives": quality_summaries,
    }
    (archive_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
