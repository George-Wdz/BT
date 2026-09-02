#!/usr/bin/env python3
"""Apply accepted historical IDs to the latest 0727 namespace safely."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path


VERSION_RANGES = {
    "0401": ("0000-01-01", "2026-04-29T18:21:19.033281"),
    "0429": ("2026-04-29T18:21:19.033281", "2026-05-27T10:38:04.238518"),
    "0611": ("2026-05-27T10:38:04.238518", "2026-07-08T23:43:54.540569"),
    "0727": ("2026-07-08T23:43:54.540569", "9999-12-31"),
}
TABLES = (("phy_data", "satelliteId"), ("position_data", "satId"))


def load_mapping(path: Path) -> dict[str, dict[int, int]]:
    mappings: dict[str, dict[int, int]] = {version: {} for version in VERSION_RANGES}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            version = row["source_let_version"]
            source = int(row["raw_satellite_id"])
            target = int(row["canonical_0727_satellite_id"])
            existing = mappings[version].get(source)
            if existing is not None and existing != target:
                raise ValueError(f"conflicting mapping for {version}:{source}")
            mappings[version][source] = target
    return mappings


def online_backup(source_path: Path, backup_path: Path) -> None:
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True, timeout=60)
    destination = sqlite3.connect(backup_path)
    last_percent = -10

    def progress(status: int, remaining: int, total: int) -> None:
        nonlocal last_percent
        percent = round((total - remaining) * 100 / max(total, 1))
        if percent >= last_percent + 10 or remaining == 0:
            print(f"Database backup: {percent}%", flush=True)
            last_percent = percent

    try:
        # Pin one WAL read snapshot so continuous collector writes cannot make
        # sqlite3_backup restart from earlier pages indefinitely.
        source.execute("BEGIN")
        source.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        source.backup(destination, pages=4096, progress=progress, sleep=0.05)
        result = destination.execute("PRAGMA quick_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"backup quick_check failed: {result}")
    finally:
        destination.close()
        if source.in_transaction:
            source.rollback()
        source.close()


def case_expression(column: str, mapping: dict[int, int]) -> tuple[str, list[int]]:
    clauses = []
    parameters = []
    for source, target in sorted(mapping.items()):
        clauses.append("WHEN ? THEN ?")
        parameters.extend((source, target))
    return f"CASE {column} {' '.join(clauses)} ELSE {column} END", parameters


def create_rollback_sidecar(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE changed_rows (
            table_name TEXT NOT NULL,
            row_id INTEGER NOT NULL,
            raw_satellite_id INTEGER NOT NULL,
            canonical_satellite_id INTEGER NOT NULL,
            PRIMARY KEY (table_name, row_id)
        )
        """
    )
    return connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", type=Path,
        default=Path("/home/wdz/satellite_data/satellite_data.db"),
    )
    parser.add_argument(
        "--mapping-path", type=Path,
        default=Path(__file__).resolve().parent / "analysis" / "latest" / "accepted_canonical_0727_mapping.csv",
    )
    parser.add_argument(
        "--backup-dir", type=Path,
        default=Path("/home/wdz/satellite_data/backups"),
    )
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    args = parser.parse_args()

    db_path = args.db_path.resolve()
    mapping_path = args.mapping_path.resolve()
    mappings = load_mapping(mapping_path)
    changed_mappings = {
        version: {source: target for source, target in mapping.items() if source != target}
        for version, mapping in mappings.items()
    }
    changed_mappings = {key: value for key, value in changed_mappings.items() if value}
    if not changed_mappings:
        raise ValueError("mapping contains no ID changes")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = args.backup_dir / f"satellite_data_before_0727_mapping_{run_id}.db"
    rollback_path = args.backup_dir / f"satellite_id_mapping_rows_{run_id}.sqlite3"
    report_path = Path(__file__).resolve().parent / "analysis" / "latest" / f"mapping_apply_{run_id}.json"

    print(f"Creating online backup: {backup_path}", flush=True)
    online_backup(db_path, backup_path)
    print(f"Backup validated: {backup_path}", flush=True)

    rollback = create_rollback_sidecar(rollback_path)
    database = sqlite3.connect(db_path, timeout=120)
    database.execute("PRAGMA busy_timeout=120000")
    summary_rows = []
    try:
        before_quick_check = database.execute("PRAGMA quick_check").fetchone()[0]
        if before_quick_check != "ok":
            raise RuntimeError(f"source quick_check failed: {before_quick_check}")
        database.execute("BEGIN IMMEDIATE")
        for version, mapping in changed_mappings.items():
            start, stop = VERSION_RANGES[version]
            source_ids = sorted(mapping)
            placeholders = ",".join("?" for _ in source_ids)
            for table, column in TABLES:
                selected = database.execute(
                    f"""
                    SELECT id, {column}
                    FROM {table}
                    WHERE terminalId = ? AND localTime >= ? AND localTime < ?
                      AND {column} IN ({placeholders})
                    """,
                    (args.terminal_id, start, stop, *source_ids),
                ).fetchall()
                rollback.executemany(
                    "INSERT INTO changed_rows VALUES (?, ?, ?, ?)",
                    (
                        (table, row_id, raw_id, mapping[raw_id])
                        for row_id, raw_id in selected
                    ),
                )
                expression, case_parameters = case_expression(column, mapping)
                cursor = database.execute(
                    f"""
                    UPDATE {table}
                    SET {column} = {expression}
                    WHERE terminalId = ? AND localTime >= ? AND localTime < ?
                      AND {column} IN ({placeholders})
                    """,
                    (*case_parameters, args.terminal_id, start, stop, *source_ids),
                )
                if cursor.rowcount != len(selected):
                    raise RuntimeError(
                        f"row count mismatch for {table}/{version}: "
                        f"selected={len(selected)} updated={cursor.rowcount}"
                    )
                counts = Counter(raw_id for _, raw_id in selected)
                summary_rows.extend(
                    {
                        "version": version,
                        "table": table,
                        "source_id": source,
                        "canonical_id": target,
                        "affected_rows": counts.get(source, 0),
                    }
                    for source, target in sorted(mapping.items())
                )
                print(
                    f"Updated {table}/{version}: {cursor.rowcount} rows",
                    flush=True,
                )
        rollback.commit()
        database.commit()
    except Exception:
        database.rollback()
        rollback.rollback()
        raise
    finally:
        database.close()
        rollback.close()

    verification = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        quick_check = verification.execute("PRAGMA quick_check").fetchone()[0]
        distinct_ids = {
            table: verification.execute(
                f"SELECT COUNT(DISTINCT {column}) FROM {table} WHERE terminalId = ?",
                (args.terminal_id,),
            ).fetchone()[0]
            for table, column in TABLES
        }
    finally:
        verification.close()
    if quick_check != "ok":
        raise RuntimeError(f"post-update quick_check failed: {quick_check}")

    report = {
        "run_id": run_id,
        "database": str(db_path),
        "mapping": str(mapping_path),
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "terminal_id": args.terminal_id,
        "version_ranges": VERSION_RANGES,
        "backup": str(backup_path),
        "rollback_rows": str(rollback_path),
        "updated_rows": sum(row["affected_rows"] for row in summary_rows),
        "updates": summary_rows,
        "post_update_quick_check": quick_check,
        "post_update_distinct_ids": distinct_ids,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "updates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
