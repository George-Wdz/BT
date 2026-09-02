#!/usr/bin/env python3
"""Backfill invalid raw positions for later TLE repair without model exposure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .apply_accepted_mapping import online_backup
except ImportError:
    from apply_accepted_mapping import online_backup


INVALID_POSITION_ONLY_ID = 2147483908
DB_COLUMNS = [
    "localTime", "ueId", "satId", "altitude", "posLatitude", "posLongitude",
    "northSouthDirSpeed", "eastWestDirSpeed", "verticalDirSpeed",
    "ecefPx", "ecefPy", "ecefPz", "longitude", "latitude", "satAltitude",
    "reportTimestamp", "bdtTime", "visibleSatCount", "visibleSatPos", "terminalId",
]


def load_mapping(path: Path) -> dict[tuple[str, int], int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["source_let_version"], int(row["raw_satellite_id"])):
                int(row["canonical_0727_satellite_id"])
            for row in csv.DictReader(handle)
        }


def versions(local_time: pd.Series) -> pd.Series:
    result = pd.Series("0401", index=local_time.index)
    result[local_time >= "2026-04-29T18:21:19.033281"] = "0429"
    result[local_time >= "2026-05-27T10:38:04.238518"] = "0611"
    result[local_time >= "2026-07-08T23:43:54.540569"] = "0727"
    return result


def classify(frame: pd.DataFrame) -> pd.Series:
    numeric = {
        field: pd.to_numeric(frame[field], errors="coerce")
        for field in (
            "satAltitude", "longitude", "latitude", "altitude",
            "posLongitude", "posLatitude", "ecefPx", "ecefPy", "ecefPz",
        )
    }
    reason = pd.Series("", index=frame.index, dtype="string")
    for field in (
        "satAltitude", "longitude", "latitude", "altitude",
        "posLongitude", "posLatitude", "ecefPx", "ecefPy", "ecefPz",
    ):
        reason[(reason == "") & numeric[field].isna()] = f"missing_{field}"
        reason[(reason == "") & numeric[field].eq(0)] = f"zero_{field}"
    radius2 = numeric["ecefPx"] ** 2 + numeric["ecefPy"] ** 2 + numeric["ecefPz"] ** 2
    reason[(reason == "") & ~radius2.between(6.4e6 ** 2, 1.0e7 ** 2)] = "ecef_radius_out_of_range"
    reason[(reason == "") & ~numeric["satAltitude"].between(1.0e5, 3.0e6)] = "satAltitude_out_of_leo_range"
    return reason


def sql_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=Path("/home/wdz/satellite_data/satellite_data.db"))
    parser.add_argument(
        "--mapping-path", type=Path,
        default=module_dir / "analysis/latest/operational_canonical_0727_mapping.csv",
    )
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    parser.add_argument("--backup-dir", type=Path, default=Path("/home/wdz/satellite_data/backups"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"satellite_data_before_raw_position_backfill_{run_id}.db"
    online_backup(args.db_path.resolve(), backup)

    connection = sqlite3.connect(args.db_path, timeout=120)
    connection.execute("PRAGMA busy_timeout=120000")
    placeholders = ",".join("?" for _ in DB_COLUMNS)
    insert_sql = (
        f"INSERT OR IGNORE INTO position_data ({','.join(DB_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    counts = Counter()
    deleted_anomaly = connection.execute(
        "DELETE FROM position_data WHERE satId=?", (INVALID_POSITION_ONLY_ID,)
    ).rowcount
    connection.commit()

    for frame in pd.read_csv(args.csv_path, chunksize=args.chunk_size, low_memory=False):
        counts["rows_scanned"] += len(frame)
        reason = classify(frame)
        satellite_id = pd.to_numeric(frame["satId"], errors="coerce").astype("Int64")
        invalid = reason.ne("") & satellite_id.notna() & satellite_id.ne(INVALID_POSITION_ONLY_ID)
        if not invalid.any():
            continue
        selected = frame.loc[invalid].copy()
        selected_versions = versions(selected["localTime"].astype(str))
        rows = []
        for index, raw in selected.iterrows():
            source_id = int(raw["satId"])
            version = selected_versions.loc[index]
            values = raw.to_dict()
            values.update({
                "satId": mapping.get((version, source_id), source_id),
                "terminalId": args.terminal_id,
            })
            rows.append(tuple(sql_value(values.get(column)) for column in DB_COLUMNS))
        before = connection.total_changes
        connection.executemany(insert_sql, rows)
        connection.commit()
        inserted = connection.total_changes - before
        counts["candidate_invalid_rows"] += len(rows)
        counts["inserted_rows"] += inserted
        counts["duplicates_skipped"] += len(rows) - inserted
        print(
            f"scanned={counts['rows_scanned']} inserted={counts['inserted_rows']}",
            flush=True,
        )
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    position_ids = connection.execute(
        "SELECT COUNT(DISTINCT satId) FROM position_data WHERE terminalId=?",
        (args.terminal_id,),
    ).fetchone()[0]
    connection.close()
    if quick_check != "ok":
        raise RuntimeError(f"post-backfill quick_check failed: {quick_check}")

    report = {
        "run_id": run_id,
        "database": str(args.db_path.resolve()),
        "raw_csv": str(args.csv_path.resolve()),
        "mapping": str(args.mapping_path.resolve()),
        "backup": str(backup),
        "deleted_anomalous_position_rows": deleted_anomaly,
        **counts,
        "post_backfill_position_ids": position_ids,
        "post_backfill_quick_check": quick_check,
    }
    report_path = module_dir / "analysis/latest" / f"raw_position_backfill_{run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
