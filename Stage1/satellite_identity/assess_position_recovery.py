#!/usr/bin/env python3
"""Audit how much historical PHY telemetry has trustworthy position support."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sqlite3
from array import array
from collections import Counter, defaultdict
from pathlib import Path


def load_mapping(mapping_path: Path) -> dict[tuple[str, int], str]:
    statuses: dict[tuple[str, int], str] = {}
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_version"], int(row["raw_satellite_id"]))
            statuses[key] = row["status"]
    return statuses


def load_daily_catalogs(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [row["date"] for row in rows],
        [row["selected_catalog"] for row in rows],
    )


def catalog_for_date(date: str, dates: list[str], catalogs: list[str]) -> str | None:
    if not dates:
        return None
    index = bisect.bisect_right(dates, date) - 1
    # Before position collection started, use the first evidenced catalog.
    index = max(index, 0)
    return catalogs[index]


def is_clean_position(row: sqlite3.Row) -> bool:
    nonzero_fields = (
        "longitude", "latitude", "altitude", "posLongitude", "posLatitude",
    )
    if not all(
        row[name] is not None and math.isfinite(float(row[name]))
        and float(row[name]) != 0.0
        for name in nonzero_fields
    ):
        return False
    orbit_values = [row[name] for name in ("ecefPx", "ecefPy", "ecefPz", "satAltitude")]
    if not all(value is not None and math.isfinite(float(value)) for value in orbit_values):
        return False
    radius = math.sqrt(sum(float(value) ** 2 for value in orbit_values[:3]))
    satellite_altitude = float(orbit_values[3])
    return 6.4e6 <= radius <= 1.0e7 and 1.0e5 <= satellite_altitude <= 3.0e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", type=Path,
        default=Path("/home/wdz/satellite_data/satellite_data.db"),
    )
    parser.add_argument(
        "--mapping-path", type=Path,
        default=Path(__file__).resolve().parent / "analysis" / "latest" / "satellite_id_mapping.csv",
    )
    parser.add_argument(
        "--daily-catalog-path", type=Path,
        default=Path(__file__).resolve().parent / "analysis" / "latest" / "daily_position_catalog_scores.csv",
    )
    parser.add_argument("--tolerance-seconds", type=float, default=5.0)
    parser.add_argument(
        "--output-path", type=Path,
        default=Path(__file__).resolve().parent / "analysis" / "latest" / "position_recovery_audit.json",
    )
    args = parser.parse_args()

    mapping_status = load_mapping(args.mapping_path)
    catalog_dates, daily_catalogs = load_daily_catalogs(args.daily_catalog_path)
    connection = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    position_times: dict[int, array] = defaultdict(lambda: array("q"))
    try:
        for row in connection.execute(
            """
            SELECT satId, CAST(bdtTime AS INTEGER) AS bdt_ms,
                   ecefPx, ecefPy, ecefPz, longitude, latitude,
                   altitude, posLongitude, posLatitude, satAltitude
            FROM position_data
            WHERE satId IS NOT NULL AND bdtTime IS NOT NULL
            ORDER BY satId, CAST(bdtTime AS INTEGER)
            """
        ):
            if is_clean_position(row):
                position_times[int(row["satId"])].append(int(row["bdt_ms"]))

        tolerance_ms = round(args.tolerance_seconds * 1000)
        counts = Counter()
        per_satellite: dict[int, Counter] = defaultdict(Counter)
        for row in connection.execute(
            """
            SELECT satelliteId, CAST(bdtTime AS INTEGER) AS bdt_ms, localTime
            FROM phy_data
            WHERE satelliteId IS NOT NULL AND bdtTime IS NOT NULL
              AND satelliteId != 4294967295
            ORDER BY id
            """
        ):
            satellite_id = int(row["satelliteId"])
            bdt_ms = int(row["bdt_ms"])
            date = str(row["localTime"])[:10]
            source_version = catalog_for_date(date, catalog_dates, daily_catalogs)
            identity_status = mapping_status.get((source_version, satellite_id))
            times = position_times.get(satellite_id)
            matched = False
            if times:
                index = bisect.bisect_left(times, bdt_ms)
                for candidate_index in (index - 1, index):
                    if 0 <= candidate_index < len(times):
                        if abs(times[candidate_index] - bdt_ms) <= tolerance_ms:
                            matched = True
                            break

            if matched:
                category = "same_id_ecef_within_tolerance"
            elif identity_status == "accepted":
                category = "canonical_identity_known_but_historical_ecef_missing"
            elif identity_status == "provisional":
                category = "numeric_id_continuity_provisional_ecef_missing"
            elif identity_status == "unresolved":
                category = "cross_version_identity_unresolved"
            elif source_version is None:
                category = "catalog_version_unavailable_for_date"
            else:
                category = "satellite_id_outside_supplied_let_catalogs"
            counts[category] += 1
            per_satellite[satellite_id][category] += 1

    finally:
        connection.close()

    total = sum(counts.values())
    report = {
        "db_path": str(args.db_path.resolve()),
        "mapping_path": str(args.mapping_path.resolve()),
        "daily_catalog_path": str(args.daily_catalog_path.resolve()),
        "position_match_rule": (
            "same satellite ID and nearest clean position bdtTime within "
            f"{args.tolerance_seconds:g} seconds"
        ),
        "clean_position_rule": (
            "finite ECEF with LEO radius 6400-10000 km and altitude 100-3000 km; "
            "non-zero satellite and receiver geodetic fields used by current ingestion"
        ),
        "catalog_date_rule": (
            "select the earliest catalog on equal daily ID overlap, switch when a "
            "later catalog has better overlap, and carry the latest evidenced catalog "
            "through dates without position rows"
        ),
        "total_phy_rows_with_bdt": total,
        "counts": dict(counts),
        "ratios": {
            key: value / total if total else 0.0
            for key, value in counts.items()
        },
        "important_limitation": (
            "Canonical ID mapping alone cannot reconstruct ECEF coordinates at a "
            "historical timestamp. Rows without same-time position remain missing "
            "until the opaque LET orbit payload is authoritatively decoded and propagated."
        ),
        "largest_unmatched_satellites": [
            {"satellite_id": satellite_id, **dict(category_counts)}
            for satellite_id, category_counts in sorted(
                per_satellite.items(),
                key=lambda item: sum(
                    value for key, value in item[1].items()
                    if key != "same_id_ecef_within_tolerance"
                ),
                reverse=True,
            )[:50]
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
