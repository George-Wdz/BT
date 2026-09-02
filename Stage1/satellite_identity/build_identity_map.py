#!/usr/bin/env python3
"""Build a conservative historical LET ID map to the latest catalog."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .let_table import LetTable, parse_let_table
except ImportError:  # Direct script execution.
    from let_table import LetTable, parse_let_table


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LETS = [
    ("0401", ROOT / "let_table0401.bin"),
    ("0429", ROOT / "let_0429.bin"),
    ("0611", ROOT / "let0611.bin"),
    ("0727", ROOT / "let0727(1)(1).bin"),
]


def parse_let_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("LET must use VERSION=/path/to/file.bin")
    version, raw_path = value.split("=", 1)
    if not version or not raw_path:
        raise argparse.ArgumentTypeError("LET must use VERSION=/path/to/file.bin")
    return version, Path(raw_path).expanduser().resolve()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_position_evidence(db_path: Path) -> tuple[dict[int, dict], list[dict]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        summary_rows = connection.execute(
            """
            SELECT satId AS satellite_id,
                   COUNT(*) AS position_rows,
                   MIN(localTime) AS first_position_time,
                   MAX(localTime) AS last_position_time,
                   SUM(CASE WHEN ecefPx IS NOT NULL AND ecefPy IS NOT NULL
                                  AND ecefPz IS NOT NULL
                                  AND ecefPx != 0 AND ecefPy != 0 AND ecefPz != 0
                                  AND longitude != 0 AND latitude != 0
                                  AND altitude != 0
                                  AND posLongitude != 0 AND posLatitude != 0
                            THEN 1 ELSE 0 END) AS clean_ecef_rows
            FROM position_data
            WHERE satId IS NOT NULL
            GROUP BY satId
            ORDER BY satId
            """
        ).fetchall()
        evidence = {
            int(row["satellite_id"]): dict(row)
            for row in summary_rows
        }
        daily_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT substr(localTime, 1, 10) AS date,
                       satId AS satellite_id,
                       COUNT(*) AS position_rows
                FROM position_data
                WHERE satId IS NOT NULL
                GROUP BY substr(localTime, 1, 10), satId
                ORDER BY date, satId
                """
            )
        ]
    finally:
        connection.close()
    return evidence, daily_rows


def build_daily_catalog_scores(
    daily_rows: list[dict], tables: list[tuple[str, LetTable]]
) -> list[dict]:
    by_date: dict[str, set[int]] = defaultdict(set)
    for row in daily_rows:
        by_date[str(row["date"])].add(int(row["satellite_id"]))

    catalog_ids = {
        version: set(table.by_id)
        for version, table in tables
    }
    rows: list[dict] = []
    for date, observed_ids in sorted(by_date.items()):
        scores = {
            version: len(observed_ids & ids) / max(len(observed_ids), 1)
            for version, ids in catalog_ids.items()
        }
        best_score = max(scores.values())
        dominant = [version for version, score in scores.items() if score == best_score]
        row = {
            "date": date,
            "observed_position_ids": len(observed_ids),
            "dominant_catalog": "|".join(dominant),
            # On a tie, retain the earlier supplied catalog until a later
            # catalog has a uniquely better overlap. The ambiguity remains
            # visible in dominant_catalog.
            "selected_catalog": dominant[0],
            "dominant_overlap_ratio": round(best_score, 6),
        }
        row.update({f"overlap_{version}": round(score, 6) for version, score in scores.items()})
        rows.append(row)
    return rows


def build_mapping_rows(
    tables: list[tuple[str, LetTable]], position_evidence: dict[int, dict]
) -> list[dict]:
    versions = [version for version, _ in tables]
    version_ids = [set(table.by_id) for _, table in tables]
    latest_version = versions[-1]
    rows: list[dict] = []

    for source_index, (source_version, table) in enumerate(tables):
        for source_id in sorted(table.by_id):
            continuity = [
                source_id in version_ids[index]
                for index in range(source_index, len(tables))
            ]
            continuous_to_latest = all(continuity)
            reappears_after_gap = (
                source_id in version_ids[-1] and not continuous_to_latest
            )
            evidence = position_evidence.get(source_id, {})

            if source_version == latest_version:
                canonical_id: int | str = source_id
                candidate_id: int | str = source_id
                status = "accepted"
                method = "latest_catalog_identity"
                confidence = "definition"
            elif continuous_to_latest:
                canonical_id = ""
                candidate_id = source_id
                status = "provisional"
                method = "numeric_identifier_continuity_without_physical_orbit_validation"
                confidence = "unverified"
            else:
                canonical_id = ""
                candidate_id = source_id if reappears_after_gap else ""
                status = "unresolved"
                method = (
                    "numeric_id_reappears_after_catalog_gap_not_accepted"
                    if reappears_after_gap
                    else "no_verified_cross_version_identity"
                )
                confidence = "none"

            rows.append(
                {
                    "source_version": source_version,
                    "raw_satellite_id": source_id,
                    "canonical_version": latest_version,
                    "canonical_satellite_id": canonical_id,
                    "candidate_canonical_satellite_id": candidate_id,
                    "status": status,
                    "method": method,
                    "confidence": confidence,
                    "confidence_scope": (
                        "LET catalog identifier continuity; not a calibrated "
                        "physical-object probability"
                    ),
                    "version_presence_path": ">".join(
                        f"{versions[index]}:{int(continuity[index - source_index])}"
                        for index in range(source_index, len(tables))
                    ),
                    "position_rows": evidence.get("position_rows", 0),
                    "clean_ecef_rows": evidence.get("clean_ecef_rows", 0),
                    "first_position_time": evidence.get("first_position_time", ""),
                    "last_position_time": evidence.get("last_position_time", ""),
                }
            )
    return rows


def build_let_point_rows(tables: list[tuple[str, LetTable]]) -> list[dict]:
    rows: list[dict] = []
    for version, table in tables:
        for record in table.records:
            for point_index, point in enumerate(record.points):
                rows.append(
                    {
                        "let_version": version,
                        "satellite_id": record.satellite_id,
                        "point_index": point_index,
                        "epoch_bdt_seconds": point.epoch_bdt_seconds,
                        "opaque_payload_hex": point.orbit_payload.hex(),
                        "opaque_payload_sha256": point.payload_sha256,
                    }
                )
    return rows


def build_latest_provenance_rows(tables: list[tuple[str, LetTable]]) -> list[dict]:
    versions = [version for version, _ in tables]
    version_ids = [set(table.by_id) for _, table in tables]
    latest_ids = sorted(version_ids[-1])
    rows: list[dict] = []
    for satellite_id in latest_ids:
        presence = [satellite_id in ids for ids in version_ids]
        first_index = presence.index(True)
        has_gap = not all(presence[first_index:])
        if has_gap:
            status = "numeric_id_reappeared_after_gap_identity_unresolved"
        elif first_index == 0:
            status = "continuous_catalog_entry_from_earliest_supplied_version"
        else:
            status = "new_catalog_entry_or_renumbered_identity_unresolved"
        rows.append(
            {
                "latest_satellite_id": satellite_id,
                "first_seen_version": versions[first_index],
                "catalog_status": status,
                "version_presence_path": ">".join(
                    f"{version}:{int(is_present)}"
                    for version, is_present in zip(versions, presence)
                ),
                "interpretation": (
                    "Catalog appearance alone cannot distinguish a newly launched "
                    "satellite from a renumbered historical satellite."
                ),
            }
        )
    return rows


def build_ecef_anchor_rows(
    db_path: Path, tables: list[tuple[str, LetTable]], tolerance_ms: int = 3000
) -> list[dict]:
    """Find observed ECEF rows close to LET epochs without decoding payloads."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            """
            CREATE TEMP TABLE let_epochs (
                let_version TEXT,
                satellite_id INTEGER,
                point_index INTEGER,
                epoch_bdt_ms INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO let_epochs VALUES (?, ?, ?, ?)",
            (
                (version, record.satellite_id, point_index, point.epoch_bdt_seconds * 1000)
                for version, table in tables
                for record in table.records
                for point_index, point in enumerate(record.points)
            ),
        )
        connection.execute(
            "CREATE INDEX temp.idx_let_epochs_satellite ON let_epochs(satellite_id)"
        )
        rows = connection.execute(
            """
            WITH candidates AS (
                SELECT e.let_version, e.satellite_id, e.point_index,
                       e.epoch_bdt_ms, p.localTime, p.bdtTime,
                       p.ecefPx, p.ecefPy, p.ecefPz,
                       p.longitude, p.latitude, p.satAltitude,
                       abs(CAST(p.bdtTime AS INTEGER) - e.epoch_bdt_ms) AS lag_ms,
                       ROW_NUMBER() OVER (
                           PARTITION BY e.let_version, e.satellite_id, e.point_index
                           ORDER BY abs(CAST(p.bdtTime AS INTEGER) - e.epoch_bdt_ms), p.id
                       ) AS candidate_rank
                FROM position_data AS p
                JOIN let_epochs AS e ON e.satellite_id = p.satId
                WHERE p.bdtTime IS NOT NULL
                  AND CAST(p.bdtTime AS INTEGER)
                      BETWEEN e.epoch_bdt_ms - ? AND e.epoch_bdt_ms + ?
                  AND p.ecefPx IS NOT NULL AND p.ecefPy IS NOT NULL
                  AND p.ecefPz IS NOT NULL
            )
            SELECT * FROM candidates WHERE candidate_rank = 1
            ORDER BY let_version, satellite_id, point_index
            """,
            (tolerance_ms, tolerance_ms),
        ).fetchall()
    finally:
        connection.close()
    return [
        {key: row[key] for key in row.keys() if key != "candidate_rank"}
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--let", action="append", type=parse_let_argument,
        help="ordered LET catalog as VERSION=/path/file.bin; repeat as needed",
    )
    parser.add_argument(
        "--db-path", type=Path,
        default=Path("/home/wdz/satellite_data/satellite_data.db"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "analysis" / "latest",
    )
    args = parser.parse_args()

    let_specs = args.let or DEFAULT_LETS
    tables = [(version, parse_let_table(path)) for version, path in let_specs]
    if len({version for version, _ in tables}) != len(tables):
        raise ValueError("LET version labels must be unique")

    position_evidence, daily_rows = load_position_evidence(args.db_path)
    mapping_rows = build_mapping_rows(tables, position_evidence)
    point_rows = build_let_point_rows(tables)
    latest_provenance_rows = build_latest_provenance_rows(tables)
    daily_scores = build_daily_catalog_scores(daily_rows, tables)
    ecef_anchor_rows = build_ecef_anchor_rows(args.db_path, tables)

    output_dir = args.output_dir.resolve()
    write_csv(
        output_dir / "satellite_id_mapping.csv", mapping_rows,
        list(mapping_rows[0]),
    )
    write_csv(
        output_dir / "let_points_opaque.csv", point_rows,
        list(point_rows[0]),
    )
    write_csv(
        output_dir / "latest_id_provenance.csv", latest_provenance_rows,
        list(latest_provenance_rows[0]),
    )
    write_csv(
        output_dir / "daily_position_catalog_scores.csv", daily_scores,
        list(daily_scores[0]),
    )
    if ecef_anchor_rows:
        write_csv(
            output_dir / "let_ecef_validation_anchors.csv", ecef_anchor_rows,
            list(ecef_anchor_rows[0]),
        )

    transitions = []
    for (left_version, left_table), (right_version, right_table) in zip(tables, tables[1:]):
        left_ids, right_ids = set(left_table.by_id), set(right_table.by_id)
        transitions.append(
            {
                "from": left_version,
                "to": right_version,
                "common_ids": len(left_ids & right_ids),
                "removed_ids": len(left_ids - right_ids),
                "added_ids": len(right_ids - left_ids),
                "net_catalog_growth": len(right_ids) - len(left_ids),
                "added_id_interpretation": (
                    "new launch or renumbered identity unresolved; not automatically "
                    "paired with a removed ID"
                ),
            }
        )

    summary = {
        "policy": (
            "Only identities in the latest catalog are accepted by definition. "
            "Numeric continuity across versions is provisional until physical-orbit "
            "validation succeeds. An ID that disappears and later reappears is unresolved."
        ),
        "orbit_payload_status": (
            "opaque: 4-byte little-endian BDT epoch is decoded; the remaining "
            "17 bytes are preserved but not interpreted"
        ),
        "let_versions": [
            {
                "version": version,
                "path": str(table.path),
                "records": len(table.records),
                "point_count_distribution": dict(
                    sorted(Counter(record.declared_point_count for record in table.records).items())
                ),
            }
            for version, table in tables
        ],
        "transitions": transitions,
        "mapping_rows": len(mapping_rows),
        "accepted_rows": sum(row["status"] == "accepted" for row in mapping_rows),
        "provisional_rows": sum(row["status"] == "provisional" for row in mapping_rows),
        "unresolved_rows": sum(row["status"] == "unresolved" for row in mapping_rows),
        "database_position_ids": len(position_evidence),
        "let_ecef_validation_anchors_within_3s": len(ecef_anchor_rows),
        "outputs": {
            "mapping": str(output_dir / "satellite_id_mapping.csv"),
            "let_points": str(output_dir / "let_points_opaque.csv"),
            "latest_id_provenance": str(output_dir / "latest_id_provenance.csv"),
            "daily_catalog_scores": str(output_dir / "daily_position_catalog_scores.csv"),
            "ecef_validation_anchors": str(output_dir / "let_ecef_validation_anchors.csv"),
        },
    }
    (output_dir / "mapping_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
