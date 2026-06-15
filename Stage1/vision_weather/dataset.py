from __future__ import annotations

import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 常见/推荐的类别顺序（用于自动推断时保持人类可读顺序）
KNOWN_CLASS_ORDER = ["sunny", "cloudy", "rain", "light_rain", "moderate_rain", "heavy_rain"]

# 兼容：当 data_dir 下无法推断到类别目录时，使用该默认列表。
DEFAULT_CLASSES = ["sunny", "cloudy", "rain"]


def _parse_timestamp_from_filename(file_name: str) -> Optional[float]:
    """从文件名中解析时间戳（Unix 秒）。

    支持示例：
    - 20260326_103000_cam01.jpg
    - 2026-03-26_10-30-00_xxx.png
    - 1711423800_cam01.jpg
    """

    stem = Path(file_name).stem

    # 兼容: ch01_20260326162220_timingCap.jpg
    m = re.search(r"(\d{14})", stem)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
            return dt.timestamp()
        except ValueError:
            pass

    m = re.search(r"(\d{8})_(\d{6})", stem)
    if m:
        text = f"{m.group(1)}{m.group(2)}"
        try:
            dt = datetime.strptime(text, "%Y%m%d%H%M%S")
            return dt.timestamp()
        except ValueError:
            pass

    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", stem)
    if m:
        text = f"{m.group(1)} {m.group(2).replace('-', ':')}"
        try:
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            pass

    m = re.search(r"\b(\d{10})\b", stem)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return None


def _preprocess_image(img: Image.Image, image_size: int, is_train: bool) -> torch.Tensor:
    """轻量预处理：Resize + (train时随机翻转) + 标准化。"""

    img = img.convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)

    if is_train and random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, C]
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C, H, W]

    # ImageNet 统计量，作为通用起点
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
    x = (x - mean) / std
    return x


class WeatherImageDataset(Dataset[Dict[str, torch.Tensor]]):
    """目录分类数据集。

    目录结构：
      data_dir/
        sunny/*.jpg
        cloudy/*.jpg
        light_rain/*.jpg
        moderate_rain/*.jpg
        heavy_rain/*.jpg

    输出字段：
      - pixel_values: [3, H, W]
      - labels: [] int64
      - timestamp_unix: [] float32（可选，若文件名可解析）
      - image_path: str（用于导出特征时追踪）
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        *,
        image_size: int,
        is_train: bool,
        class_names: Sequence[str],
        parse_timestamp: bool = True,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.class_names = list(class_names)
        self.parse_timestamp = bool(parse_timestamp)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            x = _preprocess_image(img, image_size=self.image_size, is_train=self.is_train)

        out: Dict[str, torch.Tensor | str] = {
            "pixel_values": x,
            "labels": torch.tensor(label, dtype=torch.long),
            "image_path": str(img_path),
        }

        if self.parse_timestamp:
            ts = _parse_timestamp_from_filename(img_path.name)
            if ts is not None:
                out["timestamp_unix"] = torch.tensor(ts, dtype=torch.float32)
        return out


def _infer_class_names(data_dir: Path) -> List[str]:
    candidates = [p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not candidates:
        return []

    candidate_set = set(candidates)
    if candidate_set.issubset(set(KNOWN_CLASS_ORDER)):
        return [c for c in KNOWN_CLASS_ORDER if c in candidate_set]
    return sorted(candidates)


def _resolve_class_names(data_dir: Path, class_names: Optional[Sequence[str]]) -> List[str]:
    if class_names is not None:
        resolved = [c.strip() for c in class_names if str(c).strip()]
        if resolved:
            return resolved

    inferred = _infer_class_names(data_dir)
    if inferred:
        return inferred
    return list(DEFAULT_CLASSES)


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _gather_paths_by_class(
    data_dir: Path,
    class_names: Sequence[str],
    *,
    check_images: bool,
) -> Tuple[List[List[Path]], List[Path]]:
    paths_by_class: List[List[Path]] = []
    bad_images: List[Path] = []

    for class_name in class_names:
        class_dir = data_dir / class_name
        class_paths: List[Path] = []
        if class_dir.exists():
            for p in sorted(class_dir.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if check_images and not _is_valid_image(p):
                    bad_images.append(p)
                    continue
                class_paths.append(p)
        paths_by_class.append(class_paths)

    return paths_by_class, bad_images


def _compute_split_counts(n: int, *, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    if val_ratio > 0 and n_val == 0 and n >= 2:
        n_val = 1
    if test_ratio > 0 and n_test == 0 and n >= 2:
        n_test = 1

    # 保证至少留 1 个样本给 train（如果 n>=1）
    if n_val + n_test >= n:
        overflow = n_val + n_test - (n - 1)
        while overflow > 0:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break
            overflow -= 1

    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def build_train_val_datasets(
    data_dir: str | Path,
    *,
    image_size: int,
    val_ratio: float,
    seed: int,
    class_names: Optional[Sequence[str]] = None,
    parse_timestamp: bool = True,
    check_images: bool = True,
) -> Tuple[WeatherImageDataset, WeatherImageDataset, List[str]]:
    """按类别目录读取图像，并随机切分 train/val。"""

    train_ds, val_ds, _test_ds, resolved = build_train_val_test_datasets(
        data_dir,
        image_size=image_size,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
        class_names=class_names,
        parse_timestamp=parse_timestamp,
        check_images=check_images,
    )
    return train_ds, val_ds, resolved


def build_train_val_test_datasets(
    data_dir: str | Path,
    *,
    image_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    class_names: Optional[Sequence[str]] = None,
    parse_timestamp: bool = True,
    check_images: bool = True,
) -> Tuple[WeatherImageDataset, WeatherImageDataset, WeatherImageDataset, List[str]]:
    """按类别目录读取图像，并按比例切分 train/val/test（分层按类别）。

    目录结构示例：
      data_dir/
        sunny/*.jpg
        cloudy/*.jpg
        rain/*.jpg

    Args:
        data_dir: 类别目录的根路径。
        image_size: Resize 尺寸。
        val_ratio: 验证集占比（0~1）。
        test_ratio: 测试集占比（0~1）。
        seed: 随机种子（保证复现）。
        class_names: 显式传入类别名列表；为空则从子目录推断。
        parse_timestamp: 是否尝试从文件名解析时间戳。
        check_images: 是否在构建数据集时跳过不可读/损坏图片。
    """

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >=0 and val_ratio+test_ratio < 1.0")

    root = Path(data_dir)
    if not root.exists():
        raise ValueError(f"data_dir not found: {root}")

    resolved = _resolve_class_names(root, class_names)

    paths_by_class, bad_images = _gather_paths_by_class(root, resolved, check_images=check_images)
    total_images = sum(len(x) for x in paths_by_class)
    if total_images == 0:
        raise ValueError(f"no images found under {root} (class_names={resolved})")

    if bad_images:
        preview = "\n".join(str(p) for p in bad_images[:10])
        print(f"[WARN] skipped {len(bad_images)} unreadable images. examples:\n{preview}")

    rnd = random.Random(seed)

    train_paths: List[Path] = []
    train_labels: List[int] = []
    val_paths: List[Path] = []
    val_labels: List[int] = []
    test_paths: List[Path] = []
    test_labels: List[int] = []

    for class_idx, class_paths in enumerate(paths_by_class):
        class_paths = list(class_paths)
        rnd.shuffle(class_paths)

        n = len(class_paths)
        n_train, n_val, n_test = _compute_split_counts(n, val_ratio=val_ratio, test_ratio=test_ratio)

        test_part = class_paths[:n_test]
        val_part = class_paths[n_test : n_test + n_val]
        train_part = class_paths[n_test + n_val :]

        # n_train 由剩余决定，这里做一次断言，便于排查
        if len(train_part) != n_train:
            raise RuntimeError("split count mismatch; please report this bug")

        test_paths.extend(test_part)
        test_labels.extend([class_idx] * len(test_part))
        val_paths.extend(val_part)
        val_labels.extend([class_idx] * len(val_part))
        train_paths.extend(train_part)
        train_labels.extend([class_idx] * len(train_part))

    # 最后再整体 shuffle 一次，避免 loader 看到同类聚集
    def _shuffle_pair(paths: List[Path], labels: List[int]) -> Tuple[List[Path], List[int]]:
        idxs = list(range(len(paths)))
        rnd.shuffle(idxs)
        return [paths[i] for i in idxs], [labels[i] for i in idxs]

    train_paths, train_labels = _shuffle_pair(train_paths, train_labels)
    val_paths, val_labels = _shuffle_pair(val_paths, val_labels)
    test_paths, test_labels = _shuffle_pair(test_paths, test_labels)

    train_ds = WeatherImageDataset(
        train_paths,
        train_labels,
        image_size=image_size,
        is_train=True,
        class_names=resolved,
        parse_timestamp=parse_timestamp,
    )
    val_ds = WeatherImageDataset(
        val_paths,
        val_labels,
        image_size=image_size,
        is_train=False,
        class_names=resolved,
        parse_timestamp=parse_timestamp,
    )
    test_ds = WeatherImageDataset(
        test_paths,
        test_labels,
        image_size=image_size,
        is_train=False,
        class_names=resolved,
        parse_timestamp=parse_timestamp,
    )

    return train_ds, val_ds, test_ds, list(resolved)


def build_eval_dataset(
    data_dir: str | Path,
    *,
    image_size: int,
    class_names: Optional[Sequence[str]] = None,
    parse_timestamp: bool = False,
    check_images: bool = True,
) -> Tuple[WeatherImageDataset, List[str]]:
    """构建用于独立验证/测试的完整数据集（不做随机切分）。"""

    root = Path(data_dir)
    if not root.exists():
        raise ValueError(f"data_dir not found: {root}")

    resolved = _resolve_class_names(root, class_names)
    paths_by_class, bad_images = _gather_paths_by_class(root, resolved, check_images=check_images)
    total_images = sum(len(x) for x in paths_by_class)
    if total_images == 0:
        raise ValueError(f"no images found under {root} (class_names={resolved})")

    if bad_images:
        preview = "\n".join(str(p) for p in bad_images[:10])
        print(f"[WARN] skipped {len(bad_images)} unreadable images. examples:\n{preview}")

    image_paths: List[Path] = []
    labels: List[int] = []
    for class_idx, class_paths in enumerate(paths_by_class):
        image_paths.extend(class_paths)
        labels.extend([class_idx] * len(class_paths))

    ds = WeatherImageDataset(
        image_paths,
        labels,
        image_size=image_size,
        is_train=False,
        class_names=resolved,
        parse_timestamp=parse_timestamp,
    )
    return ds, list(resolved)


def build_dataset_from_class_dir(
    data_dir: str | Path,
    *,
    image_size: int,
    is_train: bool,
    class_names: Optional[Sequence[str]] = None,
    parse_timestamp: bool = False,
    check_images: bool = True,
) -> Tuple[WeatherImageDataset, List[str]]:
    """从单个类别目录根构建数据集。

    目录示例:
      split/train/
        sunny/*.jpg
        cloudy/*.jpg
        rain/*.jpg
    """

    root = Path(data_dir)
    if not root.exists():
        raise ValueError(f"data_dir not found: {root}")

    resolved = _resolve_class_names(root, class_names)
    paths_by_class, bad_images = _gather_paths_by_class(root, resolved, check_images=check_images)
    total_images = sum(len(x) for x in paths_by_class)
    if total_images == 0:
        raise ValueError(f"no images found under {root} (class_names={resolved})")

    if bad_images:
        preview = "\n".join(str(p) for p in bad_images[:10])
        print(f"[WARN] skipped {len(bad_images)} unreadable images. examples:\n{preview}")

    image_paths: List[Path] = []
    labels: List[int] = []
    for class_idx, class_paths in enumerate(paths_by_class):
        image_paths.extend(class_paths)
        labels.extend([class_idx] * len(class_paths))

    ds = WeatherImageDataset(
        image_paths,
        labels,
        image_size=image_size,
        is_train=is_train,
        class_names=resolved,
        parse_timestamp=parse_timestamp,
    )
    return ds, list(resolved)


def build_datasets_from_split_root(
    split_root: str | Path,
    *,
    image_size: int,
    class_names: Optional[Sequence[str]] = None,
    parse_timestamp: bool = False,
    check_images: bool = True,
) -> Tuple[WeatherImageDataset, WeatherImageDataset, WeatherImageDataset, List[str]]:
    """从 prepare_weather_split.py 落盘后的 split 根目录读取 train/val/test。"""

    root = Path(split_root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise ValueError(f"split root must contain train/val/test: {root}")

    train_ds, resolved = build_dataset_from_class_dir(
        train_dir,
        image_size=image_size,
        is_train=True,
        class_names=class_names,
        parse_timestamp=parse_timestamp,
        check_images=check_images,
    )
    val_ds, _ = build_dataset_from_class_dir(
        val_dir,
        image_size=image_size,
        is_train=False,
        class_names=resolved,
        parse_timestamp=parse_timestamp,
        check_images=check_images,
    )
    test_ds, _ = build_dataset_from_class_dir(
        test_dir,
        image_size=image_size,
        is_train=False,
        class_names=resolved,
        parse_timestamp=parse_timestamp,
        check_images=check_images,
    )

    return train_ds, val_ds, test_ds, resolved
