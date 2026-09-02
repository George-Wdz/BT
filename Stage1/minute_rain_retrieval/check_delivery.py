#!/usr/bin/env python3
"""Verify that a Stage1 checkout has the dependencies and local artifacts to run."""
from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE1_ROOT = ROOT.parent
BT_ROOT = STAGE1_ROOT.parent

MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "PyYAML": "yaml",
    "Pillow": "PIL",
    "PyTorch": "torch",
    "torchvision": "torchvision",
    "FastAPI": "fastapi",
    "Uvicorn": "uvicorn",
    "pytest": "pytest",
}

REQUIRED_TABLE_COLUMNS = {
    "phy_data": {"localTime", "satelliteId", "phyRssi", "rssi", "snr",
                 "lastCniValue", "terminalId"},
    "position_data": {"localTime", "satId", "posLongitude", "posLatitude",
                      "altitude", "ecefPx", "ecefPy", "ecefPz", "terminalId"},
    "weather_data": {"timestamp", "temperature", "humidity", "pressure",
                     "terminalId"},
    "weather_station": {"datetime", "rainfall", "terminalId"},
    "phy_bb_data": {"localTime", "trackNo", "phaseNo", "validMeasBb", "snr",
                    "terminalId"},
    "phy_rssi_data": {"localTime", "validMeasRssi", "chanRssi", "carrRssi",
                      "terminalId"},
}


def report(ok: bool, label: str, detail: str = "") -> bool:
    state = "OK" if ok else "MISSING"
    print(f"[{state}] {label}{': ' + detail if detail else ''}")
    return ok


def check_database(path: Path) -> bool:
    if not report(path.is_file(), "acquisition database", str(path)):
        return False
    valid = True
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            for table, expected in REQUIRED_TABLE_COLUMNS.items():
                columns = {
                    str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
                }
                missing = sorted(expected - columns)
                valid &= report(not missing, f"database table {table}",
                                "missing=" + ",".join(missing) if missing else "schema ready")
    except sqlite3.Error as exc:
        report(False, "database readable", str(exc))
        return False
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", type=Path,
        default=Path("/home/wdz/satellite_data/satellite_data.db"),
    )
    parser.add_argument(
        "--code-only", action="store_true",
        help="Check Python modules and source files without local data/weights.",
    )
    parser.add_argument(
        "--training-only", action="store_true",
        help="Check the Git-contained training dataset without online runtime artifacts.",
    )
    args = parser.parse_args()

    checks = []
    checks.append(report(sys.version_info[:2] in {(3, 10), (3, 11)}, "Python version",
                         sys.version.split()[0]))
    for label, module in MODULES.items():
        checks.append(report(importlib.util.find_spec(module) is not None,
                             f"Python module {label}"))

    source_files = [
        ROOT / "build_dataset.py",
        ROOT / "train.py",
        ROOT / "service.py",
        STAGE1_ROOT / "rainfall_dashboard" / "app.py",
        BT_ROOT / "MoE" / "lora-moe" / "src" / "lora_moe" / "history"
        / "rain_retrieval.py",
    ]
    for path in source_files:
        checks.append(report(path.is_file(), "source file", str(path)))

    if not args.code_only:
        training_artifacts = [
            ROOT / "data" / "reproducible_v1" / "minute_rainfall_full.npz",
            ROOT / "data" / "reproducible_v1" / "train" / "minute_rainfall_train.npz",
            ROOT / "data" / "reproducible_v1" / "val" / "minute_rainfall_val.npz",
            ROOT / "data" / "reproducible_v1" / "test" / "minute_rainfall_test.npz",
            ROOT / "data" / "reproducible_v1" / "camera_weather_labels.csv",
        ]
        for path in training_artifacts:
            checks.append(report(path.is_file(), "training artifact", str(path)))

    if not args.code_only and not args.training_only:
        checks.append(check_database(args.db_path.expanduser()))
        runtime_artifacts = [
            ROOT / "data" / "archive"
            / "minute_rainfall_v1_20260825_minphy3_stratified_seed42"
            / "processed" / "minute_rainfall_full.npz",
            ROOT / "weights" / "deployed" / "position_model" / "best.pt",
            ROOT / "weights" / "deployed" / "no_position_fallback" / "best.pt",
            ROOT / "weights" / "deployed" / "new_terminal_transfer" / "best.pt",
            STAGE1_ROOT / "vision_weather" / "weights"
            / "20260623_133102_weather_cls_rain_station_balanced_100ep_best_model.pt",
            STAGE1_ROOT / "data" / "camera" / "labels"
            / "latest_weather_labels_slim.csv",
            STAGE1_ROOT / "terminal_002_rain_retrieval" / "config.yaml",
            STAGE1_ROOT / "terminal_002_rain_retrieval" / "adapter.json",
            STAGE1_ROOT / "terminal_003_rain_retrieval" / "config.yaml",
            STAGE1_ROOT / "terminal_003_rain_retrieval" / "adapter.json",
            STAGE1_ROOT / "rainfall_dashboard" / "static" / "echarts.min.js",
        ]
        for path in runtime_artifacts:
            checks.append(report(path.is_file(), "runtime artifact", str(path)))
        camera_dir = STAGE1_ROOT / "data" / "camera" / "images"
        checks.append(report(camera_dir.is_dir(), "camera image directory", str(camera_dir)))

    failed = sum(not value for value in checks)
    print(f"\nchecks={len(checks)} failed={failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
