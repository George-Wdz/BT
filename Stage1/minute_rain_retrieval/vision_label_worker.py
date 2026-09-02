#!/usr/bin/env python3
"""Incrementally classify new camera images for online rain retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, default_collate

STAGE1_ROOT = Path(__file__).resolve().parents[1]
if str(STAGE1_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_ROOT))

from vision_weather.models import WeatherClassifier
from vision_weather.predict_weather_labels import (
    IMAGE_SUFFIXES,
    InferenceImageDataset,
    _parse_timestamp_from_filename,
    _to_repo_relative,
)


FULL_COLUMNS = [
    "image_path", "file_name", "timestamp", "pred_label", "pred_idx", "confidence",
    "prob_sunny", "prob_cloudy", "prob_rain",
]
SLIM_COLUMNS = [
    "timestamp", "pred_label", "pred_idx", "confidence",
    "prob_sunny", "prob_cloudy", "prob_rain",
]


class _SafeInferenceImageDataset(InferenceImageDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            return super().__getitem__(index)
        except (OSError, ValueError):
            return {"invalid_image_path": str(self.image_paths[index])}


def _collate_valid_images(batch: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_paths = [item["invalid_image_path"] for item in batch if "invalid_image_path" in item]
    valid = [item for item in batch if "pixel_values" in item]
    collated = default_collate(valid) if valid else {}
    collated["invalid_image_paths"] = invalid_paths
    return collated


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


class IncrementalVisionLabeler:
    """Keep camera-weather CSV files synchronized with an image directory."""

    def __init__(
        self,
        *,
        input_dir: Path,
        weights_path: Path,
        full_csv_path: Path,
        slim_csv_path: Path,
        device: torch.device,
        batch_size: int = 256,
        num_workers: int = 8,
        max_images_per_refresh: int = 8192,
        refresh_interval_s: float = 60.0,
    ) -> None:
        self.input_dir = input_dir.resolve()
        self.weights_path = weights_path.resolve()
        self.full_csv_path = full_csv_path.resolve()
        self.slim_csv_path = slim_csv_path.resolve()
        self.device = device
        self.batch_size = max(int(batch_size), 1)
        self.num_workers = max(int(num_workers), 0)
        self.max_images_per_refresh = max(int(max_images_per_refresh), 0)
        self.refresh_interval_s = max(float(refresh_interval_s), 0.0)
        self._lock = threading.Lock()
        self._last_scan_monotonic = 0.0
        self._known_files: set[str] = set()
        self._invalid_files: set[str] = set()
        self._invalid_retry_at: dict[str, float] = {}
        self._frame = self._read_existing()

        checkpoint = _load_checkpoint(self.weights_path)
        self.class_names = [str(value) for value in checkpoint.get("class_names", [])]
        if self.class_names != ["sunny", "cloudy", "rain"]:
            raise ValueError(
                "online vision checkpoint must use classes sunny, cloudy, rain; "
                f"received {self.class_names}"
            )
        self.image_size = int(checkpoint.get("image_size", 224))
        self.model = WeatherClassifier(
            num_classes=len(self.class_names),
            dropout=float(checkpoint.get("dropout", 0.2)),
            resnet_width=int(checkpoint.get("resnet_width", 32)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.state: dict[str, Any] = {
            "status": "ready",
            "known_labels": int(len(self._frame)),
            "last_scan": None,
            "last_labeled": 0,
            "latest_label_time": self._latest_label_time(),
        }

    def _read_existing(self) -> pd.DataFrame:
        if self.full_csv_path.exists():
            frame = pd.read_csv(self.full_csv_path)
        elif self.slim_csv_path.exists():
            frame = pd.read_csv(self.slim_csv_path)
        else:
            frame = pd.DataFrame(columns=FULL_COLUMNS)
        for column in FULL_COLUMNS:
            if column not in frame:
                frame[column] = ""
        frame = frame[FULL_COLUMNS].copy()
        self._known_files = {
            str(value) for value in frame["file_name"].dropna() if str(value)
        }
        return frame

    def _latest_label_time(self) -> str | None:
        if self._frame.empty:
            return None
        values = pd.to_datetime(self._frame["timestamp"], errors="coerce")
        return values.max().isoformat() if values.notna().any() else None

    def _pending_images(self) -> list[Path]:
        wall_time = time.time()
        monotonic_time = time.monotonic()
        pending = []
        for path in self.input_dir.rglob("*"):
            try:
                if (
                    not path.is_file()
                    or path.suffix.lower() not in IMAGE_SUFFIXES
                    or path.name in self._known_files
                    or wall_time - path.stat().st_mtime < 2.0
                    or monotonic_time < self._invalid_retry_at.get(path.name, 0.0)
                ):
                    continue
            except FileNotFoundError:
                continue
            pending.append(path)
        pending.sort(key=lambda path: (_parse_timestamp_from_filename(path.name) or datetime.min, path.name))
        if self.max_images_per_refresh > 0:
            pending = pending[: self.max_images_per_refresh]
        return pending

    @torch.inference_mode()
    def _predict(self, image_paths: list[Path]) -> tuple[pd.DataFrame, set[str]]:
        dataset = _SafeInferenceImageDataset(image_paths, self.image_size)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=_collate_valid_images,
        )
        rows = []
        invalid_files: set[str] = set()
        for batch in loader:
            invalid_files.update(Path(path).name for path in batch["invalid_image_paths"])
            if "pixel_values" not in batch:
                continue
            pixels = batch["pixel_values"].to(self.device, non_blocking=True)
            probabilities = torch.softmax(self.model(pixels), dim=1)
            confidence, prediction = probabilities.max(dim=1)
            for index, source in enumerate(batch["image_path"]):
                path = Path(source)
                predicted_index = int(prediction[index].item())
                timestamp = _parse_timestamp_from_filename(path.name)
                rows.append(
                    {
                        "image_path": _to_repo_relative(path),
                        "file_name": path.name,
                        "timestamp": timestamp.isoformat() if timestamp else "",
                        "pred_label": self.class_names[predicted_index],
                        "pred_idx": predicted_index,
                        "confidence": round(float(confidence[index].item()), 6),
                        "prob_sunny": round(float(probabilities[index, 0].item()), 6),
                        "prob_cloudy": round(float(probabilities[index, 1].item()), 6),
                        "prob_rain": round(float(probabilities[index, 2].item()), 6),
                    }
                )
        return pd.DataFrame(rows, columns=FULL_COLUMNS), invalid_files

    def refresh(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not force and now - self._last_scan_monotonic < self.refresh_interval_s:
            return {**self.state, "last_labeled": 0, "skipped": True}
        with self._lock:
            self._last_scan_monotonic = now
            try:
                pending = self._pending_images()
                if pending:
                    additions, invalid_files = self._predict(pending)
                    self._invalid_files.update(invalid_files)
                    retry_time = time.monotonic() + 60.0
                    self._invalid_retry_at.update(
                        {file_name: retry_time for file_name in invalid_files}
                    )
                    if not additions.empty:
                        labeled_files = set(additions["file_name"].astype(str))
                        self._invalid_files.difference_update(labeled_files)
                        for file_name in labeled_files:
                            self._invalid_retry_at.pop(file_name, None)
                        self._frame = pd.concat([self._frame, additions], ignore_index=True)
                        self._frame = self._frame.drop_duplicates("file_name", keep="last")
                        order = pd.to_datetime(self._frame["timestamp"], errors="coerce")
                        self._frame = self._frame.assign(_order=order).sort_values(
                            ["_order", "file_name"], na_position="last"
                        ).drop(columns="_order")
                        _atomic_csv(self._frame[FULL_COLUMNS], self.full_csv_path)
                        _atomic_csv(self._frame[SLIM_COLUMNS], self.slim_csv_path)
                        self._known_files.update(labeled_files)
                self.state = {
                    "status": "ok",
                    "known_labels": int(len(self._frame)),
                    "last_scan": datetime.now().isoformat(timespec="seconds"),
                    "last_labeled": int(len(additions)) if pending else 0,
                    "invalid_images": int(len(self._invalid_files)),
                    "latest_label_time": self._latest_label_time(),
                    "weights": str(self.weights_path),
                }
            except Exception as exc:
                self.state = {
                    **self.state,
                    "status": "error",
                    "last_scan": datetime.now().isoformat(timespec="seconds"),
                    "last_labeled": 0,
                    "error": repr(exc),
                }
            return dict(self.state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--full-csv", type=Path, required=True)
    parser.add_argument("--slim-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    labeler = IncrementalVisionLabeler(
        input_dir=args.input_dir,
        weights_path=args.weights,
        full_csv_path=args.full_csv,
        slim_csv_path=args.slim_csv,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images_per_refresh=args.max_images,
        refresh_interval_s=0,
    )
    print(json.dumps(labeler.refresh(force=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
