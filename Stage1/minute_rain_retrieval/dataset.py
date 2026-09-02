"""Train-only preprocessing for variable-length minute samples."""
from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TrainTransforms:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    satellite_to_index: dict[int, int]
    dry_by_satellite: dict[int, np.ndarray]
    global_dry: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(len(self.feature_mean))


def load_npz(path: str) -> tuple[list[dict], np.ndarray]:
    # NumPy 2 writes nested ndarray pickle references under ``numpy._core``.
    # NumPy 1.x exposes the same implementation as ``numpy.core`` instead.
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    archive = np.load(path, allow_pickle=True)
    return archive["samples"].tolist(), archive["splits"].astype(str)


def fit_train_transforms(samples: list[dict], splits: np.ndarray) -> TrainTransforms:
    train_samples = [sample for sample, split in zip(samples, splits) if split == "train"]
    if not train_samples:
        raise ValueError("Dataset has no training samples")
    satellites = sorted({int(sat) for sample in train_samples for sat in sample["satellite_ids"]})
    satellite_to_index = {satellite: index + 1 for index, satellite in enumerate(satellites)}

    dry_sums: dict[int, np.ndarray] = {}
    dry_counts: dict[int, int] = {}
    global_parts: list[np.ndarray] = []
    for sample in train_samples:
        if float(sample["minute_rainfall_mm"]) > 1e-6:
            continue
        links = np.asarray(sample["features"], dtype=np.float32)[:, :4]
        sat_ids = np.asarray(sample["satellite_ids"], dtype=np.int64)
        global_parts.append(links)
        for satellite in np.unique(sat_ids):
            values = links[sat_ids == satellite]
            dry_sums[int(satellite)] = dry_sums.get(int(satellite), np.zeros(4)) + values.sum(axis=0)
            dry_counts[int(satellite)] = dry_counts.get(int(satellite), 0) + len(values)
    if not global_parts:
        raise ValueError("Training split has no dry minute for dry-baseline estimation")
    global_dry = np.concatenate(global_parts).mean(axis=0).astype(np.float32)
    dry_by_satellite = {
        satellite: (total / dry_counts[satellite]).astype(np.float32)
        for satellite, total in dry_sums.items()
    }

    augmented: list[np.ndarray] = []
    for sample in train_samples:
        base = np.asarray(sample["features"], dtype=np.float32)
        sat_ids = np.asarray(sample["satellite_ids"], dtype=np.int64)
        baseline = np.stack([dry_by_satellite.get(int(sat), global_dry) for sat in sat_ids])
        augmented.append(np.concatenate([base, base[:, :4] - baseline], axis=1))
    stacked = np.concatenate(augmented, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return TrainTransforms(mean, std, satellite_to_index, dry_by_satellite, global_dry)


class MinuteRainDataset(Dataset):
    def __init__(self, samples: list[dict], splits: np.ndarray, split: str,
                 transforms: TrainTransforms, max_points: int = 256):
        self.samples = [sample for sample, value in zip(samples, splits) if value == split]
        self.transforms = transforms
        self.max_points = max_points

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        base = np.asarray(sample["features"], dtype=np.float32)
        raw_sat_ids = np.asarray(sample["satellite_ids"], dtype=np.int64)
        if len(base) > self.max_points:
            keep = np.linspace(0, len(base) - 1, self.max_points).round().astype(int)
            base, raw_sat_ids = base[keep], raw_sat_ids[keep]
        raw_snr_db = base[:, 2].copy()
        baseline = np.stack([
            self.transforms.dry_by_satellite.get(int(sat), self.transforms.global_dry)
            for sat in raw_sat_ids
        ])
        features = np.concatenate([base, base[:, :4] - baseline], axis=1)
        features = (features - self.transforms.feature_mean) / self.transforms.feature_std
        sat_ids = np.asarray([
            self.transforms.satellite_to_index.get(int(sat), 0) for sat in raw_sat_ids
        ], dtype=np.int64)
        return {
            "features": torch.from_numpy(features.astype(np.float32)),
            "raw_snr_db": torch.from_numpy(raw_snr_db.astype(np.float32)),
            "satellite_ids": torch.from_numpy(sat_ids),
            "target": torch.tensor(float(sample["minute_rainfall_mm"]), dtype=torch.float32),
            "anchor_time_ns": torch.tensor(int(sample["anchor_time_ns"]), dtype=torch.int64),
            "point_count": torch.tensor(int(sample["point_count"]), dtype=torch.int64),
            "satellite_count": torch.tensor(int(sample["satellite_count"]), dtype=torch.int64),
        }


def collate_minutes(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(len(item["features"]) for item in batch)
    feature_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_length, feature_dim)
    satellite_ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    raw_snr_db = torch.full((len(batch), max_length), float("-inf"))
    valid_mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for index, item in enumerate(batch):
        length = len(item["features"])
        features[index, :length] = item["features"]
        raw_snr_db[index, :length] = item["raw_snr_db"]
        satellite_ids[index, :length] = item["satellite_ids"]
        valid_mask[index, :length] = True
    result = {
        "features": features,
        "raw_snr_db": raw_snr_db,
        "satellite_ids": satellite_ids,
        "valid_mask": valid_mask,
    }
    for key in ("target", "anchor_time_ns", "point_count", "satellite_count"):
        result[key] = torch.stack([item[key] for item in batch])
    return result
