#!/usr/bin/env python3
"""Create a self-contained raw-source archive for a processed minute dataset."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_TIMES = {
    "phy_data": "localTime",
    "position_data": "localTime",
    "weather_data": "timestamp",
    "weather_station": "datetime",
}


def dataset_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    with np.load(path, allow_pickle=True) as data:
        samples = data["samples"]
        start = min(int(item["window_start_ns"]) for item in samples)
        end = max(int(item["anchor_time_ns"]) for item in samples)
    return pd.to_datetime(start, unit="ns"), pd.to_datetime(end, unit="ns")


def copy_table(connection: sqlite3.Connection, source: sqlite3.Connection,
               table: str, time_column: str, start: str, end: str,
               terminal_id: str) -> int:
    schema = source.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not schema or not schema[0]:
        raise RuntimeError(f"source table is missing: {table}")
    connection.execute(schema[0])
    columns = [row[1] for row in source.execute(f"PRAGMA table_info({table})")]
    terminal_clause = " AND terminalId = ?" if "terminalId" in columns else ""
    params = [start, end] + ([terminal_id] if terminal_clause else [])
    query = (
        f"SELECT * FROM {table} WHERE datetime({time_column}) >= datetime(?) "
        f"AND datetime({time_column}) <= datetime(?) {terminal_clause} "
        f"ORDER BY {time_column}"
    )
    placeholders = ",".join("?" for _ in columns)
    insert = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    count = 0
    cursor = source.execute(query, params)
    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break
        connection.executemany(insert, rows)
        count += len(rows)
    connection.execute(f"CREATE INDEX idx_{table}_archive_time ON {table}({time_column})")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--output-db", required=True, type=Path)
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    parser.add_argument("--image-csv", type=Path)
    args = parser.parse_args()

    start, end = dataset_range(args.dataset_path)
    start_text = start.isoformat()
    end_text = end.isoformat()
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    if args.output_db.exists():
        args.output_db.unlink()

    source = sqlite3.connect(f"file:{args.source_db.resolve()}?mode=ro", uri=True)
    output = sqlite3.connect(args.output_db)
    output.execute("PRAGMA journal_mode=WAL")
    counts = {}
    try:
        output.execute("BEGIN")
        for table, time_column in TABLE_TIMES.items():
            counts[table] = copy_table(
                output, source, table, time_column,
                start_text, end_text, args.terminal_id,
            )
        output.commit()
        output.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        output.close()
        source.close()

    image_output = None
    image_rows = 0
    if args.image_csv and args.image_csv.is_file():
        images = pd.read_csv(args.image_csv)
        timestamps = pd.to_datetime(images["timestamp"], errors="coerce")
        images = images.loc[(timestamps >= start) & (timestamps <= end)].copy()
        image_output = args.output_db.with_name("camera_weather_labels.csv")
        images.to_csv(image_output, index=False)
        image_rows = len(images)

    manifest = {
        "processed_dataset": str(args.dataset_path.resolve()),
        "source_db": str(args.source_db.resolve()),
        "raw_archive_db": str(args.output_db.resolve()),
        "terminal_id": args.terminal_id,
        "time_range": {"start": start_text, "end": end_text},
        "rows": counts,
        "camera_weather_labels": str(image_output) if image_output else None,
        "camera_weather_label_rows": image_rows,
        "provenance_note": (
            "The factory backup was incomplete for this range and lacked the rain-gauge "
            "table. This archive therefore uses the consolidated acquisition database, "
            "whose tables retain the pre-model records used by the online dataset builder."
        ),
    }
    args.output_db.with_name("raw_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
