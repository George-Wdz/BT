#!/usr/bin/env python3
"""Audit raw terminal positions and prepare per-ID TLE acquisition windows."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    from .position_quality import version_for_local_time
except ImportError:
    from position_quality import version_for_local_time


FIELDS = [
    "localTime", "satId", "altitude", "posLatitude", "posLongitude",
    "ecefPx", "ecefPy", "ecefPz", "longitude", "latitude", "satAltitude",
]


def load_mapping(path: Path) -> dict[tuple[str, int], dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["let_version"], int(row["raw_satellite_id"])): row
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv-path", type=Path)
    source.add_argument("--db-path", type=Path)
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument(
        "--mapping-path", type=Path,
        default=Path(__file__).resolve().parent / "analysis/latest/historical_physical_mapping.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "analysis/position_quality",
    )
    parser.add_argument("--chunk-size", type=int, default=400_000)
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_path)
    reasons = Counter()
    per_key = defaultdict(lambda: {"rows": 0, "invalid_rows": 0, "first": None, "last": None})
    all_ids, valid_ids = set(), set()
    total = 0

    connection = None
    if args.csv_path:
        chunks = pd.read_csv(
            args.csv_path, usecols=FIELDS, chunksize=args.chunk_size, low_memory=False
        )
        source_description = str(args.csv_path.resolve())
    else:
        connection = sqlite3.connect(f"file:{args.db_path.resolve()}?mode=ro", uri=True)
        predicates = ["terminalId = ?"]
        params: list[object] = [args.terminal_id]
        if args.start_time:
            predicates.append("datetime(localTime) >= datetime(?)")
            params.append(args.start_time)
        if args.end_time:
            predicates.append("datetime(localTime) <= datetime(?)")
            params.append(args.end_time)
        query = (
            f"SELECT {', '.join(FIELDS)} FROM position_data "
            f"WHERE {' AND '.join(predicates)} ORDER BY id"
        )
        chunks = pd.read_sql_query(query, connection, params=params, chunksize=args.chunk_size)
        source_description = str(args.db_path.resolve())

    for frame in chunks:
        total += len(frame)
        numeric = {
            field: pd.to_numeric(frame[field], errors="coerce")
            for field in FIELDS if field not in ("localTime", "satId")
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
        valid = reason.eq("")
        reasons.update(reason.where(~valid, "valid").tolist())

        satellite_ids = pd.to_numeric(frame["satId"], errors="coerce").astype("Int64")
        all_ids.update(satellite_ids.dropna().astype(int))
        valid_ids.update(satellite_ids[valid & satellite_ids.notna()].astype(int))
        versions = frame["localTime"].astype(str).map(version_for_local_time)
        for (version, satellite_id), indices in frame[satellite_ids.notna()].groupby(
            [versions[satellite_ids.notna()], satellite_ids[satellite_ids.notna()]], sort=False
        ).groups.items():
            times = frame.loc[indices, "localTime"].astype(str)
            item = per_key[(str(version), int(satellite_id))]
            item["rows"] += len(indices)
            item["invalid_rows"] += int((~valid.loc[indices]).sum())
            first, last = times.min(), times.max()
            item["first"] = first if item["first"] is None else min(item["first"], first)
            item["last"] = last if item["last"] is None else max(item["last"], last)

    if connection is not None:
        connection.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (version, satellite_id), item in sorted(per_key.items()):
        identity = mapping.get((version, satellite_id), {})
        rows.append({
            "let_version": version,
            "raw_satellite_id": satellite_id,
            **item,
            "identity_status": identity.get("status", "not_mapped"),
            "norad_id": identity.get("norad_id", ""),
            "physical_name": identity.get("physical_name", ""),
            "recommended_tle_start": item["first"][:10] if item["invalid_rows"] else "",
            "recommended_tle_end": item["last"][:10] if item["invalid_rows"] else "",
        })
    per_id_path = args.output_dir / "position_quality_by_id.csv"
    with per_id_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_type": "csv" if args.csv_path else "sqlite",
        "source_path": source_description,
        "terminal_id": args.terminal_id if args.db_path else None,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "total_rows": total,
        "quality_counts": dict(reasons),
        "raw_satellite_ids": len(all_ids),
        "valid_satellite_ids": len(valid_ids),
        "invalid_only_satellite_ids": len(all_ids - valid_ids),
        "invalid_only_ids": sorted(all_ids - valid_ids),
        "per_id_csv": str(per_id_path.resolve()),
    }
    summary_path = args.output_dir / "position_quality_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
