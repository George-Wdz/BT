#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str], cuda_devices: str | None = None) -> None:
    print(" ".join(command), flush=True)
    environment = os.environ.copy()
    if cuda_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_devices
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def build_run_id(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    terminal = args.terminal_id.rsplit("-", 1)[-1]
    position = "pos" if args.position_mode == "required" else args.position_mode
    snr = "snr_raw" if args.min_snr_db is None else f"snr_ge_{args.min_snr_db:g}"
    return (
        f"{timestamp}_t{terminal}_minphy{args.min_phy_points}_{position}_{snr}_"
        f"{args.split_strategy}_seed{args.split_seed}"
    ).replace("-", "neg").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minute rainfall data and training workflow")
    parser.add_argument("--mode", choices=("build", "train", "all"), default="all")
    parser.add_argument(
        "--rebuild-dataset", type=int, choices=(0, 1), default=1,
        help="1: rebuild the NPZ from the current database; 0: reuse dataset-path",
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument(
        "--dataset-path",
        help="Dataset to archive/reuse. If omitted while rebuilding, a run-specific path is used.",
    )
    parser.add_argument("--output-dir", help="Model output; defaults to outputs/training_runs/RUN_ID")
    parser.add_argument("--run-id", help="Optional stable run name; defaults to timestamp plus data/split config")
    parser.add_argument(
        "--archive-root", default=str(ROOT / "data" / "training_runs"),
        help="Every run receives full/split datasets and a manifest below this directory.",
    )
    parser.add_argument(
        "--archive-quality-views", type=int, choices=(0, 1), default=0,
        help="Also derive SNR quality views for this run; normally only canonical releases need this.",
    )
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    parser.add_argument("--image-csv")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--min-phy-points", type=int, default=10)
    parser.add_argument(
        "--min-snr-db", type=float,
        help="Filter low-quality PHY points before applying min-phy-points.",
    )
    parser.add_argument("--position-mode", choices=("required", "omit"), default="required")
    parser.add_argument("--position-tolerance-seconds", type=float, default=5.0)
    parser.add_argument("--weather-tolerance-seconds", type=float, default=60.0)
    parser.add_argument("--image-tolerance-seconds", type=float, default=600.0)
    parser.add_argument("--split-strategy", choices=("stratified_all", "time", "event_holdout"),
                        default="stratified_all")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--holdout-period", nargs=2, action="append", metavar=("START", "END"),
        default=[], help="Event period reserved for test; repeat for multiple periods.",
    )
    parser.add_argument("--holdout-buffer-minutes", type=float, default=60.0)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--evaluate-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-train-dry-ratio", type=float, default=-1.0)
    parser.add_argument(
        "--selection-metric", choices=("mae", "balanced_mae", "rainy_mae"),
        default="balanced_mae",
    )
    parser.add_argument("--heavy-rain-threshold", type=float, default=0.1)
    parser.add_argument("--heavy-rain-loss-weight", type=float, default=2.0)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument(
        "--snr-quality-mode", choices=("none", "hard_mask", "soft_gate"), default="none",
        help="Apply SNR quality control inside the model without changing the dataset split.",
    )
    parser.add_argument("--snr-threshold-db", type=float, default=-10.0)
    parser.add_argument("--snr-gate-temperature-db", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_id = args.run_id or build_run_id(args)
    archive_dir = Path(args.archive_root).expanduser().resolve() / run_id
    archive_dir.mkdir(parents=True, exist_ok=False)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else ROOT / "outputs" / "training_runs" / run_id
    )
    source_dataset_path = (
        Path(args.dataset_path).expanduser().resolve()
        if args.dataset_path else archive_dir / "processed" / "minute_rainfall_full.npz"
    )

    if args.evaluate_only == 1 and args.rebuild_dataset == 1:
        raise ValueError("--evaluate-only 1 requires --rebuild-dataset 0")
    should_build = args.mode == "build" or (args.mode == "all" and args.rebuild_dataset == 1)
    should_train = args.mode in ("train", "all")
    if should_build:
        print(f"run_id={run_id}", flush=True)
        print(f"rebuild_dataset=1: building current data into {source_dataset_path}", flush=True)
        command = [sys.executable, "build_dataset.py", "--db-path", args.db_path,
                   "--output-path", str(source_dataset_path), "--terminal-id", args.terminal_id,
                   "--min-phy-points", str(args.min_phy_points),
                   "--position-mode", args.position_mode,
                   "--position-tolerance-seconds", str(args.position_tolerance_seconds),
                   "--weather-tolerance-seconds", str(args.weather_tolerance_seconds),
                   "--image-tolerance-seconds", str(args.image_tolerance_seconds),
                   "--split-strategy", args.split_strategy,
                   "--split-seed", str(args.split_seed),
                   "--holdout-buffer-minutes", str(args.holdout_buffer_minutes)]
        if args.min_snr_db is not None:
            command.extend(["--min-snr-db", str(args.min_snr_db)])
        for period in args.holdout_period:
            command.extend(["--holdout-period", *period])
        for flag, value in (("--image-csv", args.image_csv), ("--start-time", args.start_time),
                            ("--end-time", args.end_time)):
            if value:
                command.extend([flag, value])
        run(command)
    elif should_train:
        if not source_dataset_path.is_file():
            raise FileNotFoundError(
                f"rebuild_dataset=0 but dataset does not exist: {source_dataset_path}"
            )
        print(f"run_id={run_id}", flush=True)
        print(f"rebuild_dataset=0: reusing {source_dataset_path}", flush=True)

    archived_dataset_path = archive_dir / "processed" / "minute_rainfall_full.npz"
    if should_build or should_train:
        run([
            sys.executable, "archive_dataset.py",
            "--dataset-path", str(source_dataset_path),
            "--archive-dir", str(archive_dir),
            "--quality-archives", str(args.archive_quality_views),
        ])

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow_args": vars(args),
        "source_dataset": str(source_dataset_path),
        "training_dataset": str(archived_dataset_path),
        "model_output": str(output_dir),
    }
    (archive_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if should_train:
        run([sys.executable, "train.py", "--dataset-path", str(archived_dataset_path),
             "--output-dir", str(output_dir), "--epochs", str(args.epochs),
             "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
             "--patience", str(args.patience),
             "--evaluate-only", str(args.evaluate_only),
             "--max-train-dry-ratio", str(args.max_train_dry_ratio),
             "--selection-metric", args.selection_metric,
             "--heavy-rain-threshold", str(args.heavy_rain_threshold),
             "--heavy-rain-loss-weight", str(args.heavy_rain_loss_weight),
             "--probability-threshold", str(args.probability_threshold),
             "--classification-weight", str(args.classification_weight),
             "--snr-quality-mode", args.snr_quality_mode,
             "--snr-threshold-db", str(args.snr_threshold_db),
             "--snr-gate-temperature-db", str(args.snr_gate_temperature_db),
             "--seed", str(args.seed)],
            args.cuda_visible_devices)
    print(f"archive_dir={archive_dir}", flush=True)
    if should_train:
        print(f"model_output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
