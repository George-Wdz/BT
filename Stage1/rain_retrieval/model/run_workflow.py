#!/usr/bin/env python3
"""Stage1 rainfall workflow runner.

This moves workflow orchestration out of shell scripts while keeping the same
training/evaluation Python entry points. Shell wrappers can stay small and
delegate here.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
STAGE1_ROOT = ROOT.parent
BT_STAGE1_ROOT = STAGE1_ROOT.parent
DEFAULT_DB = "/home/wdz/satellite_data/satellite_data.db"
DEFAULT_WEIGHTS = (
    BT_STAGE1_ROOT
    / "vision_weather"
    / "weights"
    / "20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt"
)

GROUP_DIMS = {
    "link": 4,
    "position": 6,
    "ground_weather": 3,
    "image_weather": 4,
    "dry_delta": 4,
    "dry_delta_summary": 24,
}

VARIANT_GROUPS = {
    "full_a": "link,position,ground_weather,image_weather,dry_delta",
    "core_e": "link,position,dry_delta",
    "no_position": "link,ground_weather,image_weather,dry_delta",
}


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def env_int(name: str, default: int) -> int:
    return int(env(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = env(name)
    if value is None:
        return default
    return parse_bool(value)


def parse_bool(value: str | bool | int) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected 0/1 or true/false, got: {value}")


def timestamp_minute() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def parse_groups(groups: str) -> list[str]:
    groups = groups.strip().strip("[]").replace(" ", "")
    out = [g for g in groups.split(",") if g]
    unknown = [g for g in out if g not in GROUP_DIMS]
    if unknown:
        raise ValueError(f"unknown feature groups: {unknown}; available={sorted(GROUP_DIMS)}")
    return out


def feature_overrides(groups: str) -> list[str]:
    parsed = parse_groups(groups)
    dims = [GROUP_DIMS[g] for g in parsed]
    return [
        f"features.enabled_groups=[{','.join(parsed)}]",
        f"model.input_dim={sum(dims)}",
        f"model.feature_group_dims=[{','.join(str(d) for d in dims)}]",
    ]


def latest_npz(dataset_dir: Path, exclude: Path | None = None) -> Path | None:
    candidates = []
    for path in dataset_dir.glob("pass_dataset_*.npz"):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_checkpoint(checkpoints: Path) -> Path:
    matches = sorted(
        checkpoints.rglob("checkpoint.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No checkpoint.pth found under {checkpoints}")
    return matches[-1].parent


def write_manifest(path: Path, rows: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for key, value in rows.items():
            writer.writerow([key, value])


class Logger:
    def __init__(self, workflow_log: Path):
        self.workflow_log = workflow_log
        self.workflow_log.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{datetime.now():%F %T}] {message}"
        print(line, flush=True)
        with self.workflow_log.open("a") as f:
            f.write(line + "\n")

    def run(self, cmd: list[str], *, cwd: Path = ROOT, extra_log: Path | None = None) -> None:
        self.log(" ".join(cmd))
        if extra_log is not None:
            extra_log.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        extra_f = extra_log.open("a") if extra_log is not None else None
        try:
            with self.workflow_log.open("a") as wf:
                for line in proc.stdout:
                    print(line, end="")
                    wf.write(line)
                    if extra_f is not None:
                        extra_f.write(line)
        finally:
            if extra_f is not None:
                extra_f.close()
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


@dataclass
class Paths:
    run_ts: str
    dataset_name: str
    label_dir: Path
    dataset_dir: Path
    checkpoint_base: Path
    result_base: Path
    log_dir: Path
    label_csv: Path
    slim_label_csv: Path
    pass_dataset_path: Path
    checkpoints: Path
    result_dir: Path
    workflow_log: Path
    train_log: Path


def make_paths(args: argparse.Namespace, default_experiment: str) -> Paths:
    run_ts = args.run_ts or env("RUN_TS") or timestamp_minute()
    experiment = args.experiment or env("EXPERIMENT", default_experiment)
    dataset_name = args.dataset_name or env("DATASET_NAME", f"pass_dataset_{experiment}_{run_ts}")
    label_dir = Path(args.label_dir or env("LABEL_DIR", str(STAGE1_ROOT / "data" / "camera_labels"))).expanduser()
    dataset_dir = Path(args.dataset_dir or env("DATASET_DIR", str(ROOT / "data" / "datasets"))).expanduser()
    checkpoint_base = Path(args.checkpoint_base or env("CHECKPOINT_BASE", str(ROOT / "checkpoints"))).expanduser()
    result_base = Path(args.result_base or env("RESULT_BASE", str(STAGE1_ROOT / "analysis" / "satellite_weather_diff" / "runs"))).expanduser()
    log_dir = Path(args.log_dir or env("LOG_DIR", str(ROOT / "logs"))).expanduser()
    pass_dataset_path = Path(
        args.pass_dataset_path or env("PASS_DATASET_PATH", str(dataset_dir / f"{dataset_name}.npz"))
    ).expanduser()
    checkpoints = Path(args.checkpoints or env("CHECKPOINTS", str(checkpoint_base / dataset_name))).expanduser()
    result_dir = Path(args.run_result_dir or env("RUN_RESULT_DIR", str(result_base / dataset_name))).expanduser()
    workflow_log = Path(args.workflow_log or env("WORKFLOW_LOG", str(log_dir / f"{dataset_name}_workflow.log"))).expanduser()
    train_log = Path(args.train_log or env("TRAIN_LOG", str(log_dir / f"{dataset_name}_train.log"))).expanduser()
    return Paths(
        run_ts=run_ts,
        dataset_name=dataset_name,
        label_dir=label_dir,
        dataset_dir=dataset_dir,
        checkpoint_base=checkpoint_base,
        result_base=result_base,
        log_dir=log_dir,
        label_csv=label_dir / f"{run_ts}_weather_labels.csv",
        slim_label_csv=label_dir / f"{run_ts}_weather_labels_slim.csv",
        pass_dataset_path=pass_dataset_path,
        checkpoints=checkpoints,
        result_dir=result_dir,
        workflow_log=workflow_log,
        train_log=train_log,
    )


def predict_camera_weather(args: argparse.Namespace, paths: Paths, logger: Logger) -> Path:
    paths.label_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.camera_input_dir or env("INPUT_DIR", str(STAGE1_ROOT / "data" / "camera"))).expanduser()
    vision_dir = Path(args.vision_dir or env("VISION_DIR", str(BT_STAGE1_ROOT / "vision_weather"))).expanduser()
    weights = Path(args.vision_weights or env("WEIGHTS", str(DEFAULT_WEIGHTS))).expanduser()
    cmd = [
        args.python,
        str(vision_dir / "predict_weather_labels.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(vision_dir),
        "--save-csv",
        str(paths.label_csv),
        "--batch-size",
        str(args.vision_batch_size),
        "--num-workers",
        str(args.vision_num_workers),
    ]
    if str(weights):
        cmd.extend(["--weights", str(weights)])
    logger.run(cmd)

    latest_csv = paths.label_dir / "latest_weather_labels.csv"
    latest_slim = paths.label_dir / "latest_weather_labels_slim.csv"
    shutil.copyfile(paths.label_csv, latest_csv)
    cols = [
        "timestamp",
        "pred_label",
        "pred_idx",
        "confidence",
        "prob_sunny",
        "prob_cloudy",
        "prob_rain",
    ]
    df = pd.read_csv(paths.label_csv, usecols=cols)
    df.to_csv(paths.slim_label_csv, index=False)
    df.to_csv(latest_slim, index=False)
    logger.log(f"weather labels exported: {paths.label_csv}")
    logger.log(f"slim labels exported: {paths.slim_label_csv}")
    logger.log(f"latest slim copy: {latest_slim}")
    return paths.slim_label_csv


def build_or_reuse_dataset(args: argparse.Namespace, paths: Paths, image_csv: Path, logger: Logger) -> Path:
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_dataset:
        if not paths.pass_dataset_path.exists():
            latest = latest_npz(paths.dataset_dir)
            if latest is None:
                raise FileNotFoundError(f"REUSE_DATASET requested but no NPZ found under {paths.dataset_dir}")
            paths.pass_dataset_path = latest
        logger.log(f"reuse_dataset=1 dataset_npz={paths.pass_dataset_path}")
        return paths.pass_dataset_path

    raw_source = args.incremental_source_npz or env("INCREMENTAL_SOURCE_NPZ")
    if raw_source:
        source = Path(raw_source).expanduser()
    else:
        source = paths.pass_dataset_path if paths.pass_dataset_path.exists() else latest_npz(paths.dataset_dir, paths.pass_dataset_path)
    if source is None or not source.exists():
        source = paths.pass_dataset_path
        logger.log("no source NPZ found; building full dataset")
    else:
        logger.log(f"source_npz={source} output_npz={paths.pass_dataset_path}")

    cmd = [
        args.python,
        "incremental_build_pass_dataset.py",
        "--db-path",
        args.db_path,
        "--existing-npz",
        str(source),
        "--output-path",
        str(paths.pass_dataset_path),
        "--image-csv",
        str(image_csv),
        "--image-tolerance",
        args.image_tolerance,
    ]
    if args.incremental_npz and source.exists():
        cmd.extend(["--lookback-minutes", str(args.incremental_lookback_minutes)])
    if args.strict_source_filters:
        cmd.append("--strict-source-filters")
    logger.run(cmd)
    if not paths.pass_dataset_path.exists():
        raise FileNotFoundError(f"missing expected dataset NPZ: {paths.pass_dataset_path}")
    return paths.pass_dataset_path


def training_overrides(
    args: argparse.Namespace,
    *,
    pass_dataset_path: Path,
    checkpoints: Path,
    image_csv: Path,
    feature_groups: str,
    use_channel_attention: bool,
) -> list[str]:
    overrides = [
        f"data.pass_dataset_path={pass_dataset_path}",
        f"checkpoints={checkpoints}",
        f"data.val_strategy={args.val_strategy}",
        f"model.use_channel_attention={str(use_channel_attention).lower()}",
        f"training.iterations={args.iterations}",
        f"training.batch_size={args.batch_size}",
        f"training.epochs={args.epochs}",
        f"training.patience={args.patience}",
        f"data.num_workers={args.data_num_workers}",
        "features.link=[phyRssi,rssi,snr,lastCniValue]",
        "image_weather.enabled=true",
        f"image_weather.csv_path={image_csv}",
        f"image_weather.tolerance={args.image_tolerance}",
        "dry_baseline.enabled=true",
        "dry_baseline.method=mean",
        "dry_baseline.exclude_instant_rain=true",
        "dry_baseline.exclude_image_rain=true",
        f"dry_baseline.image_rain_prob_threshold={args.dry_baseline_image_rain_prob_threshold}",
        *feature_overrides(feature_groups),
        "model.use_summary_token=true",
        "targets.auxiliary=[rain_rate_mean,rain_rate_max,rainy_ratio]",
        f"training.auxiliary_loss_weight={args.auxiliary_loss_weight}",
    ]
    if args.lr is not None:
        overrides.append(f"training.lr={args.lr}")
    if args.e_layers is not None:
        overrides.append(f"model.e_layers={args.e_layers}")
    if args.d_layers is not None:
        overrides.append(f"model.d_layers={args.d_layers}")
    if args.d_model is not None:
        overrides.append(f"model.d_model={args.d_model}")
    if args.d_ff is not None:
        overrides.append(f"model.d_ff={args.d_ff}")
    if args.patch_len is not None:
        overrides.append(f"model.patch_len={args.patch_len}")
    if args.stride is not None:
        overrides.append(f"model.stride={args.stride}")
    overrides.extend(args.set or [])
    return overrides


def train_variant(
    args: argparse.Namespace,
    logger: Logger,
    *,
    pass_dataset_path: Path,
    checkpoints: Path,
    image_csv: Path,
    feature_groups: str,
    use_channel_attention: bool,
    train_log: Path,
) -> Path:
    checkpoints.mkdir(parents=True, exist_ok=True)
    train_log.parent.mkdir(parents=True, exist_ok=True)
    overrides = training_overrides(
        args,
        pass_dataset_path=pass_dataset_path,
        checkpoints=checkpoints,
        image_csv=image_csv,
        feature_groups=feature_groups,
        use_channel_attention=use_channel_attention,
    )
    cmd = [args.python, "main.py", "--config", args.config]
    for item in overrides:
        cmd.extend(["--set", item])
    logger.run(cmd, extra_log=train_log)
    ckpt_dir = latest_checkpoint(checkpoints)
    logger.log(f"selected_checkpoint={ckpt_dir}")
    return ckpt_dir


def evaluate_checkpoint(
    args: argparse.Namespace,
    logger: Logger,
    *,
    ckpt_dir: Path,
    result_dir: Path,
    stem: str,
) -> tuple[Path, Path, Path]:
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = result_dir / f"{stem}_predictions.csv"
    test_csv = result_dir / f"{stem}_test_predictions.csv"
    metrics_csv = result_dir / f"{stem}_metrics.csv"
    cmd = [
        args.python,
        "evaluate_checkpoint_splits.py",
        "--checkpoint-dir",
        str(ckpt_dir),
        "--batch-size",
        str(args.eval_batch_size),
        "--out-csv",
        str(pred_csv),
        "--test-csv",
        str(test_csv),
        "--metrics-csv",
        str(metrics_csv),
    ]
    logger.run(cmd)
    return pred_csv, test_csv, metrics_csv


def combine_metrics(run_result_dir: Path, variants: Iterable[str]) -> None:
    frames = []
    for variant in variants:
        matches = sorted((run_result_dir / variant).glob("*_metrics.csv"))
        if not matches:
            continue
        df = pd.read_csv(matches[-1])
        df.insert(0, "variant", variant)
        frames.append(df)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined_path = run_result_dir / "combined_metrics.csv"
    combined.to_csv(combined_path, index=False)
    pivot = combined.pivot_table(
        index=["split", "subset"],
        columns="variant",
        values=["mae", "mse", "rmse", "n"],
        aggfunc="first",
    )
    pivot.to_csv(run_result_dir / "combined_metrics_pivot.csv")


def run_rain(args: argparse.Namespace) -> None:
    paths = make_paths(args, "rain_retrieval")
    for path in (paths.label_dir, paths.dataset_dir, paths.checkpoints, paths.result_dir, paths.log_dir):
        path.mkdir(parents=True, exist_ok=True)
    logger = Logger(paths.workflow_log)
    logger.log("Stage1 workflow started")
    logger.log(f"dataset_name={paths.dataset_name}")
    logger.log(f"run_ts={paths.run_ts}")
    logger.log(f"dataset_npz={paths.pass_dataset_path}")
    logger.log(f"checkpoints={paths.checkpoints}")
    logger.log(f"results={paths.result_dir}")

    if args.image_label_csv:
        image_csv = Path(args.image_label_csv).expanduser()
    elif args.reuse_dataset:
        image_csv = paths.label_dir / "latest_weather_labels_slim.csv"
    else:
        image_csv = predict_camera_weather(args, paths, logger)
    pass_dataset_path = build_or_reuse_dataset(args, paths, image_csv, logger)
    ckpt_dir = train_variant(
        args,
        logger,
        pass_dataset_path=pass_dataset_path,
        checkpoints=paths.checkpoints,
        image_csv=image_csv,
        feature_groups=args.feature_groups,
        use_channel_attention=args.use_channel_attention,
        train_log=paths.train_log,
    )
    pred_csv, test_csv, metrics_csv = evaluate_checkpoint(
        args,
        logger,
        ckpt_dir=ckpt_dir,
        result_dir=paths.result_dir,
        stem=paths.dataset_name,
    )
    write_manifest(
        paths.result_dir / "run_manifest.csv",
        {
            "dataset_name": paths.dataset_name,
            "run_ts": paths.run_ts,
            "label_csv": str(image_csv),
            "dataset_npz": str(pass_dataset_path),
            "checkpoint_dir": str(ckpt_dir),
            "prediction_csv": str(pred_csv),
            "test_prediction_csv": str(test_csv),
            "metrics_csv": str(metrics_csv),
            "workflow_log": str(paths.workflow_log),
            "train_log": str(paths.train_log),
        },
    )
    logger.log(f"manifest={paths.result_dir / 'run_manifest.csv'}")
    logger.log("Stage1 workflow completed")


def run_feature_ablation(args: argparse.Namespace) -> None:
    args.reuse_dataset = True if args.reuse_dataset is None else args.reuse_dataset
    paths = make_paths(args, "feature_ablation")
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(paths.workflow_log)
    pass_dataset_path = paths.pass_dataset_path if paths.pass_dataset_path.exists() else latest_npz(paths.dataset_dir)
    if pass_dataset_path is None or not pass_dataset_path.exists():
        raise FileNotFoundError(f"no pass_dataset_*.npz found under {paths.dataset_dir}")
    image_csv = Path(args.image_label_csv or env("IMAGE_LABEL_CSV", str(paths.label_dir / "latest_weather_labels_slim.csv"))).expanduser()
    variants = [v for v in args.variants.split(",") if v]

    logger.log("Stage1 feature ablation workflow started")
    logger.log(f"dataset_name={paths.dataset_name}")
    logger.log(f"dataset_npz={pass_dataset_path}")
    logger.log(f"variants={','.join(variants)}")

    for variant in variants:
        groups = VARIANT_GROUPS.get(variant)
        if groups is None:
            raise ValueError(f"unknown feature ablation variant: {variant}; available={sorted(VARIANT_GROUPS)}")
        result_dir = paths.result_dir / variant
        checkpoints = paths.checkpoint_base / f"{paths.dataset_name}_{variant}"
        train_log = paths.log_dir / f"{paths.dataset_name}_{variant}_train.log"
        logger.log(f"training_variant={variant} feature_groups={groups}")
        ckpt_dir = train_variant(
            args,
            logger,
            pass_dataset_path=pass_dataset_path,
            checkpoints=checkpoints,
            image_csv=image_csv,
            feature_groups=groups,
            use_channel_attention=False,
            train_log=train_log,
        )
        pred_csv, test_csv, metrics_csv = evaluate_checkpoint(
            args,
            logger,
            ckpt_dir=ckpt_dir,
            result_dir=result_dir,
            stem=f"{paths.dataset_name}_{variant}",
        )
        write_manifest(
            result_dir / "run_manifest.csv",
            {
                "variant": variant,
                "feature_groups": groups,
                "dataset_name": paths.dataset_name,
                "dataset_npz": str(pass_dataset_path),
                "checkpoint_dir": str(ckpt_dir),
                "prediction_csv": str(pred_csv),
                "test_prediction_csv": str(test_csv),
                "metrics_csv": str(metrics_csv),
                "train_log": str(train_log),
            },
        )
    combine_metrics(paths.result_dir, variants)
    write_manifest(
        paths.result_dir / "run_manifest.csv",
        {
            "dataset_name": paths.dataset_name,
            "run_ts": paths.run_ts,
            "dataset_npz": str(pass_dataset_path),
            "variants": ",".join(variants),
            "combined_metrics": str(paths.result_dir / "combined_metrics.csv"),
            "combined_metrics_pivot": str(paths.result_dir / "combined_metrics_pivot.csv"),
            "workflow_log": str(paths.workflow_log),
        },
    )
    logger.log(f"combined_metrics={paths.result_dir / 'combined_metrics.csv'}")
    logger.log("Stage1 feature ablation workflow completed")


def run_compare_channels(args: argparse.Namespace) -> None:
    paths = make_paths(args, "rain_retrieval_compare_channels")
    for path in (paths.label_dir, paths.dataset_dir, paths.result_dir, paths.log_dir):
        path.mkdir(parents=True, exist_ok=True)
    logger = Logger(paths.workflow_log)
    logger.log("Stage1 channel comparison workflow started")
    if args.image_label_csv:
        image_csv = Path(args.image_label_csv).expanduser()
    elif args.reuse_dataset:
        image_csv = paths.label_dir / "latest_weather_labels_slim.csv"
    else:
        image_csv = predict_camera_weather(args, paths, logger)
    pass_dataset_path = build_or_reuse_dataset(args, paths, image_csv, logger)
    variants = [("cm", False), ("cw", True)]
    for variant, use_ca in variants:
        result_dir = paths.result_dir / variant
        checkpoints = paths.checkpoint_base / f"{paths.dataset_name}_{variant}"
        train_log = paths.log_dir / f"{paths.dataset_name}_{variant}_train.log"
        logger.log(f"training_variant={variant} use_channel_attention={use_ca}")
        ckpt_dir = train_variant(
            args,
            logger,
            pass_dataset_path=pass_dataset_path,
            checkpoints=checkpoints,
            image_csv=image_csv,
            feature_groups=args.feature_groups,
            use_channel_attention=use_ca,
            train_log=train_log,
        )
        pred_csv, test_csv, metrics_csv = evaluate_checkpoint(
            args,
            logger,
            ckpt_dir=ckpt_dir,
            result_dir=result_dir,
            stem=f"{paths.dataset_name}_{variant}",
        )
        write_manifest(
            result_dir / "run_manifest.csv",
            {
                "variant": variant,
                "use_channel_attention": str(use_ca).lower(),
                "dataset_name": paths.dataset_name,
                "dataset_npz": str(pass_dataset_path),
                "checkpoint_dir": str(ckpt_dir),
                "prediction_csv": str(pred_csv),
                "test_prediction_csv": str(test_csv),
                "metrics_csv": str(metrics_csv),
                "train_log": str(train_log),
            },
        )
    combine_metrics(paths.result_dir, ["cm", "cw"])
    write_manifest(
        paths.result_dir / "run_manifest.csv",
        {
            "dataset_name": paths.dataset_name,
            "run_ts": paths.run_ts,
            "label_csv": str(image_csv),
            "dataset_npz": str(pass_dataset_path),
            "combined_metrics": str(paths.result_dir / "combined_metrics.csv"),
            "combined_metrics_pivot": str(paths.result_dir / "combined_metrics_pivot.csv"),
            "workflow_log": str(paths.workflow_log),
        },
    )
    logger.log(f"combined_metrics={paths.result_dir / 'combined_metrics.csv'}")
    logger.log("Stage1 channel comparison workflow completed")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", default=env("PYTHON", "python3"))
    parser.add_argument("--config", default=env("CONFIG", "configs/default.yaml"))
    parser.add_argument("--run-ts", default=env("RUN_TS"))
    parser.add_argument("--experiment", default=env("EXPERIMENT"))
    parser.add_argument("--dataset-name", default=env("DATASET_NAME"))
    parser.add_argument("--db-path", default=env("DB_PATH", DEFAULT_DB))
    parser.add_argument("--pass-dataset-path", default=env("PASS_DATASET_PATH"))
    parser.add_argument("--image-label-csv", default=env("IMAGE_LABEL_CSV"))
    parser.add_argument("--image-tolerance", default=env("IMAGE_TOLERANCE", "10min"))
    parser.add_argument("--label-dir", default=env("LABEL_DIR"))
    parser.add_argument("--dataset-dir", default=env("DATASET_DIR"))
    parser.add_argument("--checkpoint-base", default=env("CHECKPOINT_BASE"))
    parser.add_argument("--checkpoints", default=env("CHECKPOINTS"))
    parser.add_argument("--result-base", default=env("RESULT_BASE"))
    parser.add_argument("--run-result-dir", default=env("RUN_RESULT_DIR"))
    parser.add_argument("--log-dir", default=env("LOG_DIR"))
    parser.add_argument("--workflow-log", default=env("WORKFLOW_LOG"))
    parser.add_argument("--train-log", default=env("TRAIN_LOG"))
    parser.add_argument(
        "--reuse-dataset",
        nargs="?",
        const="1",
        type=parse_bool,
        default=env_bool("REUSE_DATASET", False),
        metavar="{0,1}",
        help="0: generate new image labels and a new NPZ; 1: reuse an existing NPZ and latest image labels.",
    )
    parser.add_argument("--no-reuse-dataset", dest="reuse_dataset", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--incremental-npz", action=argparse.BooleanOptionalAction, default=env_bool("INCREMENTAL_NPZ", True))
    parser.add_argument("--incremental-source-npz", default=env("INCREMENTAL_SOURCE_NPZ"))
    parser.add_argument("--incremental-lookback-minutes", type=float, default=env_float("INCREMENTAL_LOOKBACK_MINUTES", 20.0))
    parser.add_argument("--strict-source-filters", action="store_true", default=env_bool("STRICT_SOURCE_FILTERS", False))
    parser.add_argument("--camera-input-dir", default=env("INPUT_DIR"))
    parser.add_argument("--vision-dir", default=env("VISION_DIR"))
    parser.add_argument("--vision-weights", default=env("WEIGHTS"))
    parser.add_argument("--vision-batch-size", type=int, default=env_int("VISION_BATCH_SIZE", env_int("BATCH_SIZE", 64)))
    parser.add_argument("--vision-num-workers", type=int, default=env_int("VISION_NUM_WORKERS", 0))
    parser.add_argument("--feature-groups", default=env("FEATURE_GROUPS", VARIANT_GROUPS["full_a"]))
    parser.add_argument("--use-channel-attention", action=argparse.BooleanOptionalAction, default=env_bool("USE_CHANNEL_ATTENTION", False))
    parser.add_argument("--val-strategy", default=env("VAL_STRATEGY", "stratified_all"))
    parser.add_argument("--iterations", type=int, default=env_int("ITERATIONS", 1))
    parser.add_argument("--epochs", type=int, default=env_int("EPOCHS", 100))
    parser.add_argument("--batch-size", type=int, default=env_int("TRAIN_BATCH_SIZE", env_int("BATCH_SIZE", 32)))
    parser.add_argument("--patience", type=int, default=env_int("PATIENCE", 15))
    parser.add_argument("--data-num-workers", type=int, default=env_int("DATA_NUM_WORKERS", 0))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", 128))
    parser.add_argument("--dry-baseline-image-rain-prob-threshold", type=float, default=env_float("DRY_BASELINE_IMAGE_RAIN_PROB_THRESHOLD", 0.2))
    parser.add_argument("--auxiliary-loss-weight", type=float, default=env_float("AUXILIARY_LOSS_WEIGHT", 0.3))
    parser.add_argument("--lr", default=env("LR"))
    parser.add_argument("--e-layers", type=int, default=env("E_LAYERS"))
    parser.add_argument("--d-layers", type=int, default=env("D_LAYERS"))
    parser.add_argument("--d-model", type=int, default=env("D_MODEL"))
    parser.add_argument("--d-ff", type=int, default=env("D_FF"))
    parser.add_argument("--patch-len", type=int, default=env("PATCH_LEN"))
    parser.add_argument("--stride", type=int, default=env("STRIDE"))
    parser.add_argument("--set", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rain = sub.add_parser("rain", help="Run the end-to-end rainfall workflow.")
    add_common_args(rain)
    rain.set_defaults(func=run_rain)

    ablation = sub.add_parser("feature-ablation", help="Run feature-group ablation variants.")
    add_common_args(ablation)
    ablation.set_defaults(func=run_feature_ablation)
    ablation.add_argument("--variants", default=env("VARIANTS", "core_e,no_position"))

    compare = sub.add_parser("compare-channels", help="Compare channel-mixing and two-stage attention.")
    add_common_args(compare)
    compare.set_defaults(func=run_compare_channels)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.lr = float(args.lr) if args.lr not in (None, "") else None
    args.func(args)


if __name__ == "__main__":
    main()
