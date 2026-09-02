from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VISION_ROOT = Path(__file__).resolve().parent
STAGE1_ROOT = VISION_ROOT.parent

try:
    from .models import WeatherClassifier
except ImportError:
    from models import WeatherClassifier

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_path(path_text: str, *, must_exist: bool) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_path = candidate
    repo_path = VISION_ROOT / candidate
    top = candidate.parts[0] if candidate.parts else ""

    if top in {"data", "logs", "weights"}:
        return repo_path
    if cwd_path.exists():
        return cwd_path
    if repo_path.exists():
        return repo_path
    if must_exist:
        return repo_path
    if len(candidate.parts) > 1:
        return repo_path
    return cwd_path


def _latest_weight(weights_dir: Path) -> Path:
    candidates = sorted(weights_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise ValueError(f"no *.pt found under {weights_dir}")
    return candidates[0]


def _parse_timestamp_from_filename(file_name: str) -> Optional[datetime]:
    stem = Path(file_name).stem

    m = re.search(r"(\d{14})", stem)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass

    m = re.search(r"(\d{8})_(\d{6})", stem)
    if m:
        text = f"{m.group(1)}{m.group(2)}"
        try:
            return datetime.strptime(text, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", stem)
    if m:
        text = f"{m.group(1)} {m.group(2).replace('-', ':')}"
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    return None


def _preprocess_image(img: Image.Image, image_size: int) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


class InferenceImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], image_size: int) -> None:
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            x = _preprocess_image(img, self.image_size)
        return {
            "pixel_values": x,
            "image_path": str(img_path),
        }


def _collect_images(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            files.append(p)
    return files


def _to_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(STAGE1_ROOT))
    except Exception:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict weather labels for unlabeled images and export CSV.")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder containing images to label.")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to weight file (*.pt). If empty, use latest file under <output-dir>/weights.",
    )
    parser.add_argument("--output-dir", type=str, default=str(VISION_ROOT), help="Vision artifact root.")
    parser.add_argument("--save-csv", type=str, default="", help="Custom output CSV path. Optional.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    input_dir = _resolve_path(args.input_dir, must_exist=True)
    artifact_root = _resolve_path(args.output_dir, must_exist=False)
    weights_dir = artifact_root / "weights"
    logs_dir = artifact_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.weights:
        weight_path = _resolve_path(args.weights, must_exist=True)
    else:
        weight_path = _latest_weight(weights_dir)

    ckpt = torch.load(weight_path, map_location="cpu")
    class_names = ckpt.get("class_names", [])
    if not class_names:
        raise ValueError("checkpoint does not contain class_names; please retrain or pass a compatible checkpoint")

    image_size = int(ckpt.get("image_size", 224))
    image_paths = _collect_images(input_dir)
    if not image_paths:
        raise ValueError(f"no image files found under {input_dir}")

    dataset = InferenceImageDataset(image_paths=image_paths, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherClassifier(
        num_classes=len(class_names),
        dropout=float(ckpt.get("dropout", 0.2)),
        resnet_width=int(ckpt.get("resnet_width", 32)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            for i in range(x.shape[0]):
                image_path = Path(batch["image_path"][i])
                pred_idx = int(pred[i].item())
                pred_label = str(class_names[pred_idx])
                confidence = float(conf[i].item())
                ts = _parse_timestamp_from_filename(image_path.name)

                row = {
                    "image_path": _to_repo_relative(image_path),
                    "file_name": image_path.name,
                    "timestamp": ts.isoformat() if ts else "",
                    "pred_label": pred_label,
                    "pred_idx": pred_idx,
                    "confidence": f"{confidence:.6f}",
                }

                for cls_i, cls_name in enumerate(class_names):
                    row[f"prob_{cls_name}"] = f"{float(probs[i, cls_i].item()):.6f}"

                rows.append(row)

    if args.save_csv:
        out_csv = _resolve_path(args.save_csv, must_exist=False)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = logs_dir / f"{ts}_pred_labels.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_path", "file_name", "timestamp", "pred_label", "pred_idx", "confidence"] + [
        f"prob_{name}" for name in class_names
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] weight_path={weight_path}")
    print(f"[INFO] class_names={class_names}")
    print(f"[DONE] images={len(rows)}")
    print(f"[DONE] csv={out_csv}")


if __name__ == "__main__":
    main()
