#!/usr/bin/env python3
"""Back up SQLite and insert reviewed TLE-reconstructed position candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from .apply_accepted_mapping import online_backup
except ImportError:
    from apply_accepted_mapping import online_backup


DB_COLUMNS = [
    "localTime", "ueId", "satId", "altitude", "posLatitude", "posLongitude",
    "northSouthDirSpeed", "eastWestDirSpeed", "verticalDirSpeed",
    "ecefPx", "ecefPy", "ecefPz", "longitude", "latitude", "satAltitude",
    "reportTimestamp", "bdtTime", "visibleSatCount", "visibleSatPos", "terminalId",
]


VALIDITY_COLUMNS = [
    "satAltitude", "longitude", "latitude", "altitude", "posLongitude",
    "posLatitude", "ecefPx", "ecefPy", "ecefPz",
]


def is_valid_position(values: tuple[object, ...]) -> bool:
    try:
        parsed = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) and value != 0.0 for value in parsed):
        return False
    sat_altitude = parsed[0]
    ecef_x, ecef_y, ecef_z = parsed[-3:]
    radius2 = ecef_x * ecef_x + ecef_y * ecef_y + ecef_z * ecef_z
    return 1.0e5 <= sat_altitude <= 3.0e6 and 6.4e6 ** 2 <= radius2 <= 1.0e7 ** 2


def sql_value(value: str | None):
    if value is None or value == "":
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=Path("/home/wdz/satellite_data/satellite_data.db"))
    parser.add_argument("--repaired-csv", type=Path, required=True)
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    parser.add_argument("--backup-dir", type=Path, default=Path("/home/wdz/satellite_data/backups"))
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"satellite_data_before_position_repair_{run_id}.db"
    print(f"Creating online backup: {backup}", flush=True)
    online_backup(args.db_path.resolve(), backup)

    connection = sqlite3.connect(args.db_path, timeout=120)
    connection.execute("PRAGMA busy_timeout=120000")
    placeholders = ",".join("?" for _ in DB_COLUMNS)
    insert_sql = f"INSERT INTO position_data ({','.join(DB_COLUMNS)}) VALUES ({placeholders})"
    update_columns = [column for column in DB_COLUMNS if column != "localTime"]
    update_sql = (
        f"UPDATE position_data SET {','.join(f'{column}=?' for column in update_columns)} "
        "WHERE id=?"
    )
    inserted = updated_invalid = skipped_valid = skipped_duplicate = 0
    seen_times: set[str] = set()
    batch = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        with args.repaired_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                terminal_id = row.get("terminalId") or args.terminal_id
                local_time = row["localTime"]
                if local_time in seen_times:
                    skipped_duplicate += 1
                    continue
                seen_times.add(local_time)
                existing = connection.execute(
                    f"SELECT id, {','.join(VALIDITY_COLUMNS)} "
                    "FROM position_data WHERE localTime=? LIMIT 1",
                    (local_time,),
                ).fetchone()
                row["terminalId"] = terminal_id
                if existing:
                    if is_valid_position(existing[1:]):
                        skipped_valid += 1
                        continue
                    values = [sql_value(row.get(column)) for column in update_columns]
                    connection.execute(update_sql, (*values, existing[0]))
                    updated_invalid += 1
                    continue
                batch.append(tuple(sql_value(row.get(column)) for column in DB_COLUMNS))
                if len(batch) >= args.batch_size:
                    connection.executemany(insert_sql, batch)
                    inserted += len(batch)
                    batch.clear()
            if batch:
                connection.executemany(insert_sql, batch)
                inserted += len(batch)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    verification = sqlite3.connect(f"file:{args.db_path.resolve()}?mode=ro", uri=True)
    try:
        quick_check = verification.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        verification.close()
    if quick_check != "ok":
        raise RuntimeError(f"post-insert quick_check failed: {quick_check}")

    report = {
        "run_id": run_id,
        "database": str(args.db_path.resolve()),
        "repaired_csv": str(args.repaired_csv.resolve()),
        "repaired_csv_sha256": hashlib.sha256(args.repaired_csv.read_bytes()).hexdigest(),
        "backup": str(backup),
        "inserted_rows": inserted,
        "updated_invalid_rows": updated_invalid,
        "skipped_existing_valid_rows": skipped_valid,
        "skipped_duplicate_candidate_times": skipped_duplicate,
        "applied_repaired_rows": inserted + updated_invalid,
        "post_insert_quick_check": quick_check,
    }
    report_path = args.repaired_csv.with_name(f"position_repair_apply_{run_id}.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
