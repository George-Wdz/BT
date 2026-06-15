from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KNOWN_CLASS_ORDER = ["sunny", "cloudy", "rain", "light_rain", "moderate_rain", "heavy_rain"]
VISION_ROOT = Path(__file__).resolve().parent


def _resolve_class_names(data_dir: Path, class_names: Optional[Sequence[str]]) -> List[str]:
    if class_names is not None:
        resolved = [c.strip() for c in class_names if str(c).strip()]
        if resolved:
            return resolved

    candidates = [p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not candidates:
        return []

    candidate_set = set(candidates)
    if candidate_set.issubset(set(KNOWN_CLASS_ORDER)):
        return [c for c in KNOWN_CLASS_ORDER if c in candidate_set]
    return sorted(candidates)


def _compute_split_counts(n: int, *, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    if val_ratio > 0 and n_val == 0 and n >= 2:
        n_val = 1
    if test_ratio > 0 and n_test == 0 and n >= 2:
        n_test = 1

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


def _collect_images(class_dir: Path) -> List[Path]:
    return [p for p in sorted(class_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def _resolve_target_count(
    counts: Dict[str, int],
    *,
    balance_strategy: str,
    max_per_class: int,
) -> Optional[int]:
    nonzero_counts = [count for count in counts.values() if count > 0]
    if not nonzero_counts:
        return None

    if max_per_class > 0:
        return int(max_per_class)
    if balance_strategy == "undersample":
        return min(nonzero_counts)
    return None


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return

    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split weather dataset into train/val/test and save to disk.")
    parser.add_argument("--source-dir", type=str, default=str(VISION_ROOT / "data" / "raw"))
    parser.add_argument("--output-dir", type=str, default=str(VISION_ROOT / "data" / "split"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-names", type=str, default="")
    parser.add_argument("--mode", type=str, choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument(
        "--balance-strategy",
        type=str,
        choices=["none", "undersample"],
        default="none",
        help="Use undersample to cap every non-empty class to the smallest class count before splitting.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional per-class cap before splitting. Overrides --balance-strategy when > 0.",
    )
    parser.add_argument(
        "--train-max-per-class",
        type=int,
        default=0,
        help="Optional cap for each class in train split. Overflow is moved to test.",
    )
    parser.add_argument(
        "--val-max-per-class",
        type=int,
        default=0,
        help="Optional cap for each class in val split. Overflow is moved to test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("invalid ratios")
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_dir = output_dir / "manifests"

    if not source_dir.exists():
        raise ValueError(f"source_dir not found: {source_dir}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names_arg = [x.strip() for x in args.class_names.split(",") if x.strip()] or None
    class_names = _resolve_class_names(source_dir, class_names_arg)
    if not class_names:
        raise ValueError(f"no class folders found under {source_dir}")

    rnd = random.Random(args.seed)
    images_by_class: Dict[str, List[Path]] = {}
    original_counts: Dict[str, int] = {}
    for class_name in class_names:
        class_dir = source_dir / class_name
        images = _collect_images(class_dir) if class_dir.exists() else []
        images_by_class[class_name] = images
        original_counts[class_name] = len(images)

    target_count = _resolve_target_count(
        original_counts,
        balance_strategy=args.balance_strategy,
        max_per_class=args.max_per_class,
    )

    records: List[Dict[str, str]] = []
    summary: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    selected_counts: Dict[str, int] = {}

    for class_name in class_names:
        class_dir = source_dir / class_name
        images = list(images_by_class[class_name])
        rnd.shuffle(images)
        if target_count is not None:
            images = images[: min(len(images), target_count)]
        selected_counts[class_name] = len(images)

        n = len(images)
        n_train, n_val, n_test = _compute_split_counts(n, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

        test_part = images[:n_test]
        val_part = images[n_test : n_test + n_val]
        train_part = images[n_test + n_val :]
        if len(train_part) != n_train:
            raise RuntimeError("split count mismatch")

        if args.val_max_per_class > 0 and len(val_part) > args.val_max_per_class:
            test_part.extend(val_part[args.val_max_per_class :])
            val_part = val_part[: args.val_max_per_class]

        if args.train_max_per_class > 0 and len(train_part) > args.train_max_per_class:
            test_part.extend(train_part[args.train_max_per_class :])
            train_part = train_part[: args.train_max_per_class]

        buckets = {"train": train_part, "val": val_part, "test": test_part}
        for split, part in buckets.items():
            summary[split][class_name] = len(part)
            for src in part:
                rel = src.relative_to(class_dir)
                dst = output_dir / split / class_name / rel
                _link_or_copy(src, dst, args.mode)
                records.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "source_path": str(src),
                        "split_path": str(dst),
                    }
                )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifest_dir / "split_manifest.csv"
    summary_csv = manifest_dir / "split_summary.csv"

    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "source_path", "split_path"])
        writer.writeheader()
        writer.writerows(records)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_name", "count"])
        for split in ["train", "val", "test"]:
            for class_name in class_names:
                writer.writerow([split, class_name, summary[split].get(class_name, 0)])

    print(f"[DONE] source_dir={source_dir}")
    print(f"[DONE] output_dir={output_dir}")
    print(f"[DONE] mode={args.mode}")
    print(f"[DONE] balance_strategy={args.balance_strategy} max_per_class={args.max_per_class}")
    print(f"[DONE] train_max_per_class={args.train_max_per_class} val_max_per_class={args.val_max_per_class}")
    print(f"[DONE] classes={class_names}")
    print(f"[DONE] original_counts={original_counts}")
    print(f"[DONE] selected_counts={selected_counts}")
    for split in ["train", "val", "test"]:
        split_total = sum(summary[split].values())
        print(f"[DONE] {split} total={split_total} per_class={summary[split]}")
    print(f"[DONE] manifest={manifest_csv}")
    print(f"[DONE] summary={summary_csv}")


if __name__ == "__main__":
    main()
