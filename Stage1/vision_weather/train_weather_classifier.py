from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

VISION_ROOT = Path(__file__).resolve().parent

from dataset import build_datasets_from_split_root, build_train_val_datasets
from models import WeatherClassifier


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bsz = y.shape[0]
        total_loss += loss.item() * bsz
        total_acc += _accuracy(logits.detach(), y) * bsz
        total_n += bsz

    if total_n == 0:
        return {"loss": 0.0, "acc": 0.0}
    return {"loss": total_loss / total_n, "acc": total_acc / total_n}


def _count_by_class(labels: List[int], class_names: List[str]) -> Dict[str, int]:
    cnt = Counter(labels)
    return {name: int(cnt.get(i, 0)) for i, name in enumerate(class_names)}


def _resolve_path(path_text: str, *, must_exist: bool) -> Path:
    """Resolve user-provided path robustly.

    Priority:
    1) absolute path
    2) relative to current working directory
    3) relative to repository root
    """

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
        # builder 会给出更详细报错，这里返回更可能是用户预期的 repo 路径
        return repo_path

    if len(candidate.parts) > 1:
        return repo_path
    return cwd_path


def _sanitize_name(name: str) -> str:
    """Keep filenames safe and readable."""

    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in name.strip())
    return cleaned or "default"


def _build_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_run_name(run_name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in run_name.strip())
    return cleaned or "default"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight weather image classifier.")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument(
        "--split-root",
        type=str,
        default="",
        help="Path to persisted split root containing train/val/test subdirs. If set, training uses this fixed split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(VISION_ROOT),
        help="Artifact root. Logs and weights will be saved under logs/ and weights/.",
    )
    parser.add_argument("--run-name", type=str, default="train")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--resnet-width", type=int, default=32)
    parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Comma-separated class names (optional). If empty, infer from subfolders.",
    )
    parser.add_argument(
        "--parse-timestamp",
        action="store_true",
        help="Parse timestamp from filename and return timestamp_unix in dataset items.",
    )
    args = parser.parse_args()

    data_dir = _resolve_path(args.data_dir, must_exist=True)
    split_root = _resolve_path(args.split_root, must_exist=True) if args.split_root.strip() else None
    artifact_root = _resolve_path(args.output_dir, must_exist=False)
    run_name = _sanitize_name(_sanitize_run_name(args.run_name))
    ts = _build_timestamp()

    logs_dir = artifact_root / "logs"
    weights_dir = artifact_root / "weights"
    log_csv_path = logs_dir / f"{ts}_{run_name}.csv"
    best_weight_path = weights_dir / f"{ts}_{run_name}_best_model.pt"

    torch.manual_seed(args.seed)

    class_names_arg = [x.strip() for x in args.class_names.split(",") if x.strip()] or None
    if split_root is not None:
        train_ds, val_ds, _test_ds, class_names = build_datasets_from_split_root(
            split_root,
            image_size=args.image_size,
            class_names=class_names_arg,
            parse_timestamp=args.parse_timestamp,
            check_images=True,
        )
    else:
        train_ds, val_ds, class_names = build_train_val_datasets(
            data_dir,
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            class_names=class_names_arg,
            parse_timestamp=args.parse_timestamp,
            check_images=True,
        )
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherClassifier(
        num_classes=num_classes,
        dropout=args.dropout,
        resnet_width=args.resnet_width,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    history: List[Dict[str, float]] = []

    print(f"[INFO] device={device} model=tiny_resnet classes={class_names}")
    print(f"[INFO] data_dir={data_dir}")
    if split_root is not None:
        print(f"[INFO] split_root={split_root}")
    print(f"[INFO] artifact_root={artifact_root}")
    print(f"[INFO] run_name={run_name}")
    print(f"[INFO] logs_dir={logs_dir}")
    print(f"[INFO] weights_dir={weights_dir}")
    print(f"[INFO] split_sizes train={len(train_ds)} val={len(val_ds)}")
    class_counts = {"train": _count_by_class(train_ds.labels, class_names), "val": _count_by_class(val_ds.labels, class_names)}
    print(f"[INFO] class_counts train={class_counts['train']} val={class_counts['val']}")
    print(f"[INFO] train_log_csv={log_csv_path}")
    print(f"[INFO] best_weight_file={best_weight_path}")

    with log_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_metrics = _run_epoch(model, val_loader, optimizer=None, device=device)

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }

        with log_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
            )
            writer.writerow(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f}"
        )

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "model_name": "tiny_resnet",
                "class_names": class_names,
                "image_size": args.image_size,
                "dropout": args.dropout,
                "resnet_width": args.resnet_width,
                "run_name": run_name,
                "timestamp": ts,
                "data_dir": str(data_dir),
            }
            torch.save(ckpt, best_weight_path)
            print(f"[INFO] Saved best model: val_acc={best_val_acc:.4f}")

    print(f"[DONE] best_val_acc={best_val_acc:.4f}")
    print(f"[DONE] train_log_csv={log_csv_path}")
    print(f"[DONE] best_weight_file={best_weight_path}")


if __name__ == "__main__":
    main()
