#!/usr/bin/env python3
"""Workflow runner for the GPT2 rainfall retrieval baseline."""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = (
    "/home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/"
    "pass_dataset_rain_retrieval_20260626_1804.npz"
)
DEFAULT_IMAGE_CSV = (
    "/home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/"
    "latest_weather_labels_slim.csv"
)
DEFAULT_GPT2_DIR = "/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/gpt2"
DEFAULT_POSITION_COLUMNS = (
    "[longitude,latitude,satAltitude,posLongitude,posLatitude,altitude,"
    "slant_range_km,elevation_deg,azimuth_sin,azimuth_cos]"
)
DEFAULT_FEATURE_GROUP_DIMS = "[4,10,3,4,4]"


def timestamp_minute() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def count_visible_gpus(visible: str) -> int:
    visible = visible.strip().replace(" ", "")
    if not visible or visible.lower() == "all":
        try:
            import torch

            return int(torch.cuda.device_count())
        except Exception:
            return 0
    return len([item for item in visible.split(",") if item])


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_logged(cmd: list[str], *, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    print(format_cmd(cmd))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def build_train_command(args: argparse.Namespace, checkpoint_base: Path) -> tuple[list[str], int]:
    if args.ddp:
        nproc = count_visible_gpus(args.cuda_visible_devices)
        if nproc < 2:
            raise ValueError(
                f"--ddp requires at least two visible GPUs, got: {args.cuda_visible_devices}"
            )
        launcher = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            "train_gpt2_rain.py",
        ]
    else:
        nproc = 1
        launcher = [args.python, "train_gpt2_rain.py"]

    cmd = [
        *launcher,
        "--config",
        args.config,
        "--set",
        f"checkpoints={checkpoint_base}",
        "--set",
        f"data.pass_dataset_path={args.dataset_npz}",
        "--set",
        f"data.val_strategy={args.val_strategy}",
        "--set",
        f"image_weather.csv_path={args.image_label_csv}",
        "--set",
        f"features.position={args.position_columns}",
        "--set",
        f"model.input_dim={args.input_dim}",
        "--set",
        f"model.feature_group_dims={args.feature_group_dims}",
        "--set",
        f"model.gpt2_model_dir={args.gpt2_model_dir}",
        "--set",
        f"model.gpt2_layers={args.gpt2_layers}",
        "--set",
        f"model.freeze_gpt2={args.freeze_gpt2}",
        "--set",
        f"training.iterations={args.iterations}",
        "--set",
        f"training.batch_size={args.batch_size}",
        "--set",
        f"training.epochs={args.epochs}",
        "--set",
        f"training.patience={args.patience}",
        "--set",
        f"training.lr={args.lr}",
    ]
    for item in args.set:
        cmd.extend(["--set", item])
    return cmd, nproc


def build_eval_command(args: argparse.Namespace, best_checkpoint: str, result_dir: Path, run_ts: str) -> list[str]:
    return [
        args.python,
        "evaluate_gpt2_checkpoint.py",
        "--checkpoint-dir",
        best_checkpoint,
        "--batch-size",
        str(args.eval_batch_size),
        "--out-csv",
        str(result_dir / f"gpt2_rain_{run_ts}_predictions.csv"),
        "--test-csv",
        str(result_dir / f"gpt2_rain_{run_ts}_test_predictions.csv"),
        "--metrics-csv",
        str(result_dir / f"gpt2_rain_{run_ts}_metrics.csv"),
    ]


def write_manifest(path: Path, rows: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for key, value in rows.items():
            writer.writerow([key, value])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-ts", default="")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--eval-cuda-visible-devices", default="0")
    parser.add_argument("--ddp", action="store_true")

    parser.add_argument("--checkpoint-base", default="")
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--log-dir", default=str(ROOT / "logs"))

    parser.add_argument("--dataset-npz", default=DEFAULT_DATASET)
    parser.add_argument("--image-label-csv", default=DEFAULT_IMAGE_CSV)
    parser.add_argument("--gpt2-model-dir", default=DEFAULT_GPT2_DIR)

    parser.add_argument("--val-strategy", default="stratified_before_test")
    parser.add_argument("--position-columns", default=DEFAULT_POSITION_COLUMNS)
    parser.add_argument("--input-dim", type=int, default=25)
    parser.add_argument("--feature-group-dims", default=DEFAULT_FEATURE_GROUP_DIMS)
    parser.add_argument("--gpt2-layers", type=int, default=6)
    parser.add_argument("--freeze-gpt2", default="all", choices=["all", "ln_wpe", "none"])

    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ts = args.run_ts or timestamp_minute()
    checkpoint_base = Path(args.checkpoint_base or ROOT / "checkpoints" / f"gpt2_rain_{run_ts}")
    result_dir = Path(args.result_dir or ROOT / "runs" / f"gpt2_rain_{run_ts}")
    log_dir = Path(args.log_dir)
    train_log = log_dir / f"gpt2_rain_{run_ts}_train.log"
    eval_log = log_dir / f"gpt2_rain_{run_ts}_eval.log"

    train_cmd, nproc = build_train_command(args, checkpoint_base)
    train_env = os.environ.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    eval_env = os.environ.copy()
    eval_env["CUDA_VISIBLE_DEVICES"] = args.eval_cuda_visible_devices

    print(f"[INFO] run_ts={run_ts}")
    print(f"[INFO] ddp={int(args.ddp)} nproc_per_node={nproc} cuda_visible_devices={args.cuda_visible_devices}")
    print(f"[INFO] checkpoint_base={checkpoint_base}")
    print(f"[INFO] result_dir={result_dir}")
    print("[INFO] train command:")
    run_logged(train_cmd, env=train_env, log_path=train_log, dry_run=args.dry_run)
    if args.dry_run:
        print("[INFO] eval command:")
        dry_eval = build_eval_command(args, "<best_checkpoint>", result_dir, run_ts)
        print(format_cmd(dry_eval))
        return

    best_path = checkpoint_base / "best_iteration_checkpoint.txt"
    if not best_path.exists():
        raise FileNotFoundError(f"missing best checkpoint marker: {best_path}")
    best_checkpoint = best_path.read_text().strip()

    eval_cmd = build_eval_command(args, best_checkpoint, result_dir, run_ts)
    print("[INFO] eval command:")
    run_logged(eval_cmd, env=eval_env, log_path=eval_log, dry_run=False)

    write_manifest(
        result_dir / "run_manifest.csv",
        {
            "run_ts": run_ts,
            "ddp": str(int(args.ddp)),
            "nproc_per_node": str(nproc),
            "cuda_visible_devices": args.cuda_visible_devices,
            "checkpoint_base": str(checkpoint_base),
            "best_checkpoint": best_checkpoint,
            "result_dir": str(result_dir),
            "train_log": str(train_log),
            "eval_log": str(eval_log),
        },
    )


if __name__ == "__main__":
    main()
