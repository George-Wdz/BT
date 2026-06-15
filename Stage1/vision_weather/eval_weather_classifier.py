from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

VISION_ROOT = Path(__file__).resolve().parent

from dataset import build_eval_dataset
from models import WeatherClassifier


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


def _evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    num_classes = len(class_names)

    total_loss = 0.0
    total_n = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)

            bsz = y.shape[0]
            total_loss += loss.item() * bsz
            total_n += bsz

            for yt, yp in zip(y.tolist(), pred.tolist()):
                cm[yt][yp] += 1

    if total_n == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "confusion_matrix": cm.tolist(),
            "per_class": {},
            "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

    per_class: Dict[str, Dict[str, float | int]] = {}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for i, class_name in enumerate(class_names):
        tp = float(cm[i][i].item())
        fn = float(cm[i, :].sum().item() - tp)
        fp = float(cm[:, i].sum().item() - tp)
        support = int(cm[i, :].sum().item())

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "class_accuracy": recall,
            "support": support,
        }

    macro_avg = {
        "precision": precision_sum / num_classes,
        "recall": recall_sum / num_classes,
        "f1": f1_sum / num_classes,
    }

    total_correct = float(cm.diag().sum().item())
    return {
        "loss": total_loss / total_n,
        "acc": total_correct / float(total_n),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_avg": macro_avg,
    }


def _save_eval_csv(path: Path, metrics: Dict[str, object], class_names: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "class", "loss", "acc", "precision", "recall", "f1", "support"])

        macro = metrics["macro_avg"]
        writer.writerow(
            [
                "overall",
                "-",
                f"{metrics['loss']:.6f}",
                f"{metrics['acc']:.6f}",
                f"{macro['precision']:.6f}",
                f"{macro['recall']:.6f}",
                f"{macro['f1']:.6f}",
                "-",
            ]
        )

        for class_name in class_names:
            cls = metrics["per_class"][class_name]
            writer.writerow(
                [
                    "per_class",
                    class_name,
                    "",
                    f"{cls['class_accuracy']:.6f}",
                    f"{cls['precision']:.6f}",
                    f"{cls['recall']:.6f}",
                    f"{cls['f1']:.6f}",
                    int(cls["support"]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a weather image classifier with a chosen weight file.")
    parser.add_argument("--data-dir", type=str, required=True, help="Validation dataset root (class subfolders).")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to weight file (*.pt). If empty, use latest file under <output-dir>/weights.",
    )
    parser.add_argument("--output-dir", type=str, default=str(VISION_ROOT))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Comma-separated class names. If empty, prefer class_names from weight, then infer from folders.",
    )
    parser.add_argument("--save-csv", action="store_true", help="Save evaluation metrics as a single CSV file in logs/.")
    args = parser.parse_args()

    data_dir = _resolve_path(args.data_dir, must_exist=True)
    artifact_root = _resolve_path(args.output_dir, must_exist=False)
    logs_dir = artifact_root / "logs"
    weights_dir = artifact_root / "weights"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.weights:
        weight_path = _resolve_path(args.weights, must_exist=True)
    else:
        weight_path = _latest_weight(weights_dir)

    ckpt = torch.load(weight_path, map_location="cpu")

    user_class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    ckpt_class_names = ckpt.get("class_names", [])
    class_names = user_class_names if user_class_names else ckpt_class_names
    if not class_names:
        class_names = None

    image_size = int(ckpt.get("image_size", args.image_size))
    eval_ds, resolved_class_names = build_eval_dataset(
        data_dir,
        image_size=image_size,
        class_names=class_names,
        parse_timestamp=False,
        check_images=True,
    )

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherClassifier(
        num_classes=len(resolved_class_names),
        dropout=float(ckpt.get("dropout", 0.2)),
        resnet_width=int(ckpt.get("resnet_width", 32)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = _evaluate_detailed(model, loader, device=device, class_names=resolved_class_names)

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] weight_path={weight_path}")
    print(f"[INFO] classes={resolved_class_names}")
    print(f"[EVAL] loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} macro_f1={metrics['macro_avg']['f1']:.4f}")
    for class_name in resolved_class_names:
        cls = metrics["per_class"][class_name]
        print(
            f"[EVAL][{class_name}] "
            f"precision={cls['precision']:.4f} recall={cls['recall']:.4f} "
            f"f1={cls['f1']:.4f} support={cls['support']}"
        )

    if args.save_csv:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = logs_dir / f"{ts}_eval.csv"
        _save_eval_csv(csv_path, metrics, resolved_class_names)
        print(f"[EVAL] csv={csv_path}")


if __name__ == "__main__":
    main()
