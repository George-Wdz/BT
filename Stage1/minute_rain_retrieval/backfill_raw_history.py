#!/usr/bin/env python3
"""Rebuild minute-rain history from live and recovered terminal databases."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from service import MinuteThreeTerminalRunner


TERMINALS = (
    "01-31-0005-0001",
    "01-31-0005-0002",
    "01-31-0005-0003",
)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument("--output-db", required=True, type=Path)
    parser.add_argument("--progress-path", required=True, type=Path)
    parser.add_argument("--config-002", required=True, type=Path)
    parser.add_argument("--config-003", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--fallback-checkpoint-path", required=True, type=Path)
    parser.add_argument("--transfer-checkpoint-path", type=Path)
    parser.add_argument("--backup-db-001", required=True, type=Path)
    parser.add_argument("--backup-db-002", required=True, type=Path)
    parser.add_argument("--backup-db-003", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-phy-points", type=int, default=3)
    parser.add_argument("--position-tolerance-s", type=float, default=5.0)
    parser.add_argument("--weather-tolerance-s", type=float, default=5.0)
    parser.add_argument("--image-tolerance-s", type=float, default=600.0)
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Optional per-terminal cap within each chunk; 0 keeps every sample.",
    )
    parser.add_argument(
        "--chunk-days", type=int, default=14,
        help="Number of consecutive days read and inferred in one database scan.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.end_date < args.start_date:
        parser.error("--end-date must not precede --start-date")
    if args.chunk_days < 1:
        parser.error("--chunk-days must be positive")
    output_db = args.output_db.expanduser().resolve()
    if output_db.exists():
        if not args.overwrite:
            parser.error(f"output database already exists: {output_db}")
        output_db.unlink()
        for suffix in ("-wal", "-shm"):
            output_db.with_name(output_db.name + suffix).unlink(missing_ok=True)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    progress = args.progress_path.expanduser().resolve()
    progress.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        progress.unlink(missing_ok=True)

    runner = MinuteThreeTerminalRunner(
        config_002=args.config_002,
        config_003=args.config_003,
        checkpoint_path=args.checkpoint_path,
        fallback_checkpoint_path=args.fallback_checkpoint_path,
        transfer_checkpoint_path=args.transfer_checkpoint_path,
        device_name=args.device,
        history_db_path=output_db,
        poll_interval_s=0,
        worker_lookback_hours=24,
        worker_max_samples=256,
        link_analysis_dir=Path(__file__).resolve().parents[1]
        / "link_reliability_analysis" / "artifacts",
        min_phy_points=args.min_phy_points,
        fallback_min_phy_points=args.min_phy_points,
        position_tolerance_s=args.position_tolerance_s,
        weather_tolerance_s=args.weather_tolerance_s,
        image_tolerance_s=args.image_tolerance_s,
        probability_threshold=None,
        backup_db_001=args.backup_db_001,
        backup_db_002=args.backup_db_002,
        backup_db_003=args.backup_db_003,
        camera_input_dir=None,
        vision_weights=None,
        vision_full_csv=None,
        vision_slim_csv=None,
        vision_refresh_interval_s=60.0,
        vision_max_images_per_refresh=0,
        vision_batch_size=256,
        vision_num_workers=0,
        backup_only=True,
    )

    current = args.start_date
    total_rows = 0
    while current <= args.end_date:
        started = time.monotonic()
        chunk_end = min(
            args.end_date + timedelta(days=1),
            current + timedelta(days=args.chunk_days),
        )
        start = pd.Timestamp(datetime.combine(current, datetime.min.time()))
        end = (
            pd.Timestamp(datetime.combine(chunk_end, datetime.min.time()))
            - pd.Timedelta(microseconds=1)
        )
        counts: dict[str, int] = {}
        for terminal_id in TERMINALS:
            samples = runner._build_minute_samples(
                terminal_id, start, end, args.max_samples
            )
            rows = runner._predict_samples(terminal_id, samples)
            runner.history.upsert_many(rows)
            marker_date = current
            while marker_date < chunk_end:
                if marker_date < date.today():
                    runner.history.mark_day_materialized(
                        marker_date, terminal_id, runner.model_version
                    )
                marker_date += timedelta(days=1)
            counts[terminal_id] = len(rows)
            total_rows += len(rows)
        record = {
            "start_date": current.isoformat(),
            "end_date": (chunk_end - timedelta(days=1)).isoformat(),
            "terminal_counts": counts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        current = chunk_end

    with sqlite3.connect(output_db) as connection:
        connection.execute("ANALYZE")
        connection.execute("PRAGMA optimize")
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": total_rows,
                "database": str(output_db),
                "integrity": integrity,
                "model_version": runner.model_version,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
