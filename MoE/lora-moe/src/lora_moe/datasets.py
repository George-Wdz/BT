from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

STAGE1_MODEL_ROOT = Path("/home/wdz/BT/Stage1/rain_retrieval/model")
if str(STAGE1_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_MODEL_ROOT))

from data.data_factory import _optional_feature_keys, attach_train_dry_baseline, enabled_feature_groups, split_passes_by_time  # noqa: E402
from data.dataset import PassDataset, SatelliteIDMapper  # noqa: E402

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


def _stage1_sat_mapper_from_meta(meta: dict) -> SatelliteIDMapper:
    mapper = SatelliteIDMapper([])
    mapper.id_to_idx = {int(k): int(v) for k, v in meta["sat_mapper"].items()}
    mapper.num_satellites = max(mapper.id_to_idx.values(), default=0) + 1
    return mapper


def _stage1_pass_start(p: dict) -> str:
    return str(p["timestamps"][0])


def _stage1_pass_end(p: dict) -> str:
    return str(p["timestamps"][-1])


def normalize_stage1_rainfall(value: float, no_rain_threshold: float = 0.05) -> float:
    value = max(float(value), 0.0)
    return 0.0 if value < float(no_rain_threshold) else value


def stage1_rain_level(value: float, no_rain_threshold: float = 0.05) -> str:
    value = normalize_stage1_rainfall(value, no_rain_threshold)
    if value <= 0.0:
        return "dry"
    if value < 0.1:
        return "weak"
    if value < 1.0:
        return "light"
    return "moderate"


def format_stage1_rainfall(value: float, no_rain_threshold: float = 0.05) -> str:
    value = normalize_stage1_rainfall(value, no_rain_threshold)
    if value <= 0.0:
        return "0"
    if value < 1.0:
        return f"{value:.3f}"
    return f"{value:.2f}"


def build_stage1_metadata_prompt(
    *,
    satellite_id: int,
    pass_start: str,
    pass_end: str,
    points: int,
) -> str:
    return (
        "最新卫星过境信息：\n"
        f"- 卫星ID：{satellite_id}\n"
        f"- 过境开始：{pass_start}\n"
        f"- 过境结束：{pass_end}\n"
        f"- 采样点数：{points}\n"
        "下面是卫星链路反演专家表示，请根据该专家表示判断本次过境降雨量。"
    )


def build_stage1_rain_answer(
    *,
    satellite_id: int,
    pass_start: str,
    pass_end: str,
    points: int,
    pred_rainfall_mm: float,
    rain_probability: float | None = None,
    no_rain_threshold: float = 0.05,
    template_idx: int = 0,
) -> str:
    display_rain = normalize_stage1_rainfall(pred_rainfall_mm, no_rain_threshold)
    rain_text = format_stage1_rainfall(pred_rainfall_mm, no_rain_threshold)
    level = stage1_rain_level(pred_rainfall_mm, no_rain_threshold)

    templates = {
        "dry": [
            "卫星{satellite_id}本次过境时间为{pass_start}至{pass_end}，采样点数为{points}。链路反演结果显示无明显降雨，本次降雨量记为0毫米。",
            "最新过境卫星为{satellite_id}，时间窗口为{pass_start}至{pass_end}。卫星链路反演未检测到有效降雨信号，估计降雨量为0毫米。",
            "根据卫星链路反演专家，卫星{satellite_id}在{pass_start}至{pass_end}过境期间降雨信号很弱，估计降雨量为0毫米。",
        ],
        "weak": [
            "卫星{satellite_id}本次过境时间为{pass_start}至{pass_end}，链路反演显示存在微弱降雨信号，估计降雨量约为{rain_text}毫米。",
            "最新过境卫星为{satellite_id}，采样点数为{points}。链路反演结果显示降雨量约为{rain_text}毫米，属于微弱降雨信号。",
        ],
        "light": [
            "根据卫星链路反演专家，卫星{satellite_id}在{pass_start}至{pass_end}的过境片段对应小雨信号，估计降雨量约为{rain_text}毫米。",
            "卫星{satellite_id}最新过境窗口为{pass_start}至{pass_end}，链路反演降雨量约为{rain_text}毫米，可判断为小雨量级。",
        ],
        "moderate": [
            "链路反演结果显示，卫星{satellite_id}于{pass_start}至{pass_end}过境期间存在较明显降雨，估计降雨量约为{rain_text}毫米。",
            "最新卫星{satellite_id}过境片段显示明显降雨信号，时间为{pass_start}至{pass_end}，反演降雨量约为{rain_text}毫米。",
        ],
    }
    choices = templates[level]
    template = choices[int(template_idx) % len(choices)]
    return template.format(
        satellite_id=satellite_id,
        pass_start=pass_start,
        pass_end=pass_end,
        points=points,
        rain_text=rain_text,
        display_rain=display_rain,
    )


class Stage1RainInstructionDataset(Dataset):
    """Stage1 pass dataset turned into a rainfall retrieval instruction task."""

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        split: str,
        pass_dataset_path: str = "",
        max_samples: int = 0,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split: {split}")

        checkpoint_path = Path(checkpoint_dir).expanduser()
        meta = torch.load(checkpoint_path / "meta.pt", map_location="cpu", weights_only=False)
        cfg = meta["cfg"]
        npz_path = Path(pass_dataset_path or cfg["data"]["pass_dataset_path"]).expanduser()
        if not npz_path.exists():
            raise FileNotFoundError(f"pass dataset not found: {npz_path}")

        npz = np.load(npz_path, allow_pickle=True)
        all_passes = list(npz["passes"])
        train_passes, val_passes, test_passes = split_passes_by_time(
            all_passes,
            cfg["data"]["data_split"],
            val_strategy=cfg["data"].get("val_strategy", "time"),
            seed=cfg["training"].get("seed", 42),
        )
        train_passes, val_passes, test_passes = attach_train_dry_baseline(
            train_passes, val_passes, test_passes, cfg
        )
        split_passes = {
            "train": train_passes,
            "val": val_passes,
            "test": test_passes,
        }[split]
        if max_samples and max_samples > 0:
            split_passes = split_passes[: int(max_samples)]
        if not split_passes:
            raise ValueError(f"no Stage1 pass samples for split={split}")

        self.cfg = cfg
        self.meta = meta
        self.split = split
        self.pass_dataset_path = str(npz_path)
        self.passes = split_passes
        self.inner = PassDataset(
            split_passes,
            _stage1_sat_mapper_from_meta(meta),
            max_len=cfg["model"]["max_seq_len"],
            scaler_X=meta["scaler_X"],
            scaler_y=meta["scaler_y"],
            fit_scalers=False,
            extra_feature_keys=_optional_feature_keys(cfg),
            feature_groups=enabled_feature_groups(cfg),
            target_names=list(cfg["targets"]["primary"]) + list(cfg["targets"].get("auxiliary", [])),
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        sample = self.inner[idx]
        pass_obj = self.passes[idx]
        rainfall = float(sample["labels_phys"][0])
        satellite_id = int(pass_obj["satellite_id"])
        pass_start = _stage1_pass_start(pass_obj)
        pass_end = _stage1_pass_end(pass_obj)
        points = len(pass_obj["timestamps"])
        return {
            **sample,
            "satellite_id": satellite_id,
            "pass_start": pass_start,
            "pass_end": pass_end,
            "points": points,
            "true_rainfall_mm": rainfall,
            "answer": "",
        }


def stage1_rain_collate(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "satellite_idx": torch.tensor([item["satellite_idx"] for item in batch], dtype=torch.long),
        "labels_phys": torch.stack([item["labels_phys"] for item in batch], dim=0),
        "answer": [item["answer"] for item in batch],
        "satellite_id": [item["satellite_id"] for item in batch],
        "pass_start": [item["pass_start"] for item in batch],
        "pass_end": [item["pass_end"] for item in batch],
        "points": [item["points"] for item in batch],
        "true_rainfall_mm": [item["true_rainfall_mm"] for item in batch],
    }
