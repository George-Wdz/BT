from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = ["sunny", "cloudy", "rain"]
DEFAULT_LABEL_ZH = {
    "sunny": "晴天",
    "cloudy": "多云",
    "rain": "下雨",
}


def preprocess_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


class WeatherInstructionDataset(Dataset):
    """Image-folder dataset turned into a simple visual instruction task."""

    def __init__(
        self,
        *,
        split_root: str,
        split: str,
        image_size: int,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        label_zh: dict[str, str] | None = None,
        max_samples: int = 0,
    ) -> None:
        self.split_dir = Path(split_root).expanduser() / split
        self.image_size = int(image_size)
        self.class_names = list(class_names)
        self.label_zh = dict(DEFAULT_LABEL_ZH)
        if label_zh:
            self.label_zh.update(label_zh)

        samples: list[tuple[Path, str]] = []
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    samples.append((path, class_name))

        if max_samples and max_samples > 0:
            samples = samples[: int(max_samples)]
        if not samples:
            raise ValueError(f"no image samples found under {self.split_dir}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        label_text = self.label_zh.get(label, label)
        return {
            "pixel_values": preprocess_image(path, self.image_size),
            "label": label,
            "label_text": label_text,
            "image_path": str(path),
            "answer": f"这张图像的天气是{label_text}。",
        }


def weather_collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "label": [item["label"] for item in batch],
        "label_text": [item["label_text"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }

