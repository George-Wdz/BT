#!/usr/bin/env python3
"""Match terminal LET IDs to physical Qianfan NORAD identities.

The matcher combines three independent observations at common epochs:
historical GP/TLE orbit propagation, terminal-reported ECEF positions, and PHY
pass visibility from the fixed Shanghai receiver.  Results are written as a
sidecar mapping; the source database is never modified.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sqlite3
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, jday
from sgp4.propagation import gstime

try:
    from .let_table import LetRecord, parse_let_table
except ImportError:
    from let_table import LetRecord, parse_let_table


ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
BDT_ORIGIN = datetime(2006, 1, 1, tzinfo=timezone.utc)
BDT_MINUS_UTC_SECONDS = 4.0
CHINA_STANDARD_TIME = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class VersionSpec:
    version: str
    let_path: Path
    history_path: Path
    start_local_time: str
    stop_local_time: str


@dataclass(frozen=True)
class ElementSet:
    epoch: datetime
    satellite: Satrec


@dataclass
class PublicSatellite:
    norad_id: int
    name: str
    object_id: str
    elements: list[ElementSet]

    def nearest(self, epoch: datetime) -> Satrec:
        epochs = [item.epoch for item in self.elements]
        index = bisect.bisect_left(epochs, epoch)
        choices = {max(0, index - 1), min(len(epochs) - 1, index)}
        best = min(choices, key=lambda idx: abs((epochs[idx] - epoch).total_seconds()))
        return self.elements[best].satellite


def default_specs(history_dir: Path) -> list[VersionSpec]:
    return [
        VersionSpec("0401", ROOT / "let_table0401.bin", history_dir / "qianfan_gp_history_0401.csv", "0000-01-01", "2026-04-29T18:21:19.033281"),
        VersionSpec("0429", ROOT / "let_0429.bin", history_dir / "qianfan_gp_history_0429.csv", "2026-04-29T18:21:19.033281", "2026-05-27T10:38:04.238518"),
        VersionSpec("0611", ROOT / "let0611.bin", history_dir / "qianfan_gp_history_0611.csv", "2026-05-27T10:38:04.238518", "2026-07-08T23:43:54.540569"),
        VersionSpec("0727", ROOT / "let0727(1)(1).bin", history_dir / "qianfan_gp_history_0727.csv", "2026-07-08T23:43:54.540569", "9999-12-31"),
    ]


def circular_difference(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def bdt_to_utc(value_ms: int) -> datetime:
    return BDT_ORIGIN + timedelta(
        seconds=value_ms / 1000.0 - BDT_MINUS_UTC_SECONDS
    )


def local_time_to_utc(value: str | datetime) -> datetime:
    """Interpret receiver ``localTime`` as China Standard Time and return UTC."""
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=CHINA_STANDARD_TIME)
    return timestamp.astimezone(timezone.utc)


def load_history(path: Path) -> dict[int, PublicSatellite]:
    grouped: dict[int, list[tuple[datetime, Satrec]]] = defaultdict(list)
    metadata: dict[int, tuple[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            norad_id = int(row["NORAD_CAT_ID"])
            epoch = datetime.fromisoformat(row["EPOCH"]).replace(tzinfo=timezone.utc)
            satellite = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
            grouped[norad_id].append((epoch, satellite))
            metadata[norad_id] = (row["OBJECT_NAME"], row["OBJECT_ID"])
    result = {}
    for norad_id, values in grouped.items():
        values.sort(key=lambda item: item[0])
        name, object_id = metadata[norad_id]
        result[norad_id] = PublicSatellite(
            norad_id,
            name,
            object_id,
            [ElementSet(epoch, satellite) for epoch, satellite in values],
        )
    return result


def propagate_ecef(satellite: Satrec, epoch: datetime) -> tuple[np.ndarray, Satrec]:
    second = epoch.second + epoch.microsecond / 1e6
    jd, fraction = jday(
        epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, second
    )
    error, position, _ = satellite.sgp4(jd, fraction)
    if error:
        raise ValueError(f"SGP4 propagation error {error}")
    theta = gstime(jd + fraction)
    cosine, sine = math.cos(theta), math.sin(theta)
    x, y, z = position
    return np.array(
        [cosine * x + sine * y, -sine * x + cosine * y, z], dtype=float
    ), satellite


def sample_evenly(rows: list[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
    if len(rows) <= limit:
        return rows
    indices = np.linspace(0, len(rows) - 1, limit, dtype=int)
    return [rows[index] for index in indices]


def table_has_column(
    connection: sqlite3.Connection, table: str, column: str
) -> bool:
    """Return whether a SQLite table contains a column.

    Early single-terminal backups predate the ``terminalId`` column.  They are
    still useful for identity recovery because every row belongs to terminal
    001; newer multi-terminal databases must retain the explicit filter.
    """
    return any(
        str(row["name"]) == column
        for row in connection.execute(f"PRAGMA table_info({table})")
    )


def load_position_samples(
    connection: sqlite3.Connection, spec: VersionSpec, limit: int
) -> dict[int, list[sqlite3.Row]]:
    rows = connection.execute(
        """
        SELECT satId, localTime, bdtTime, ecefPx, ecefPy, ecefPz
        FROM position_data
        WHERE localTime >= ? AND localTime < ?
          AND satId IS NOT NULL AND bdtTime IS NOT NULL
          AND ecefPx IS NOT NULL AND ecefPy IS NOT NULL AND ecefPz IS NOT NULL
          AND ecefPx != 0 AND ecefPy != 0 AND ecefPz != 0
        ORDER BY satId, CAST(bdtTime AS INTEGER)
        """,
        (spec.start_local_time, spec.stop_local_time),
    ).fetchall()
    grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[int(row["satId"])].append(row)
    return {key: sample_evenly(values, limit) for key, values in grouped.items()}


def load_phy_pass_centers(
    connection: sqlite3.Connection,
    spec: VersionSpec,
    terminal_id: str,
    max_passes: int,
) -> dict[int, list[datetime]]:
    terminal_clause = (
        "terminalId = ? AND "
        if table_has_column(connection, "phy_data", "terminalId")
        else ""
    )
    parameters: tuple[str, ...]
    if terminal_clause:
        parameters = (terminal_id, spec.start_local_time, spec.stop_local_time)
    else:
        parameters = (spec.start_local_time, spec.stop_local_time)
    rows = connection.execute(
        f"""
        SELECT satelliteId, localTime, bdtTime
        FROM phy_data
        WHERE {terminal_clause}localTime >= ? AND localTime < ?
          AND satelliteId IS NOT NULL AND localTime IS NOT NULL
        ORDER BY satelliteId, localTime
        """,
        parameters,
    ).fetchall()
    grouped: dict[int, list[datetime]] = defaultdict(list)
    for row in rows:
        grouped[int(row["satelliteId"])].append(
            local_time_to_utc(row["localTime"])
        )

    result: dict[int, list[datetime]] = {}
    for satellite_id, times in grouped.items():
        passes: list[list[datetime]] = []
        current = [times[0]]
        for timestamp in times[1:]:
            if timestamp - current[-1] > timedelta(seconds=60):
                passes.append(current)
                current = []
            current.append(timestamp)
        passes.append(current)
        centers = [item[len(item) // 2] for item in passes if len(item) >= 10]
        if centers:
            if len(centers) > max_passes:
                indices = np.linspace(0, len(centers) - 1, max_passes, dtype=int)
                centers = [centers[index] for index in indices]
            result[satellite_id] = centers
    return result


def let_orbit_residual(
    record: LetRecord, public: PublicSatellite
) -> tuple[float, float] | None:
    if not public.elements:
        return None
    first_epoch = public.elements[0].epoch - timedelta(days=2)
    last_epoch = public.elements[-1].epoch + timedelta(days=2)
    eligible_points = [
        point for point in record.points
        if first_epoch <= bdt_to_utc(point.epoch_bdt_seconds * 1000) <= last_epoch
    ]
    if not eligible_points:
        return None
    point = min(
        eligible_points,
        key=lambda item: abs(
            (bdt_to_utc(item.epoch_bdt_seconds * 1000) - public.elements[-1].epoch)
            .total_seconds()
        ),
    )
    epoch = bdt_to_utc(point.epoch_bdt_seconds * 1000)
    if epoch.year != 2026:
        return None
    _, _, raw_inclination, raw_node, _, _ = struct.unpack(
        "<IBHHiI", point.orbit_payload
    )
    if raw_inclination < 10_000:
        return None
    satellite = public.nearest(epoch)
    try:
        _, propagated = propagate_ecef(satellite, epoch)
    except ValueError:
        return None
    second = epoch.second + epoch.microsecond / 1e6
    jd, fraction = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, second)
    inclination = raw_inclination * 360.0 / 65536.0
    earth_fixed_node = raw_node * 360.0 / 65536.0
    public_inclination = math.degrees(propagated.im)
    public_node = (math.degrees(propagated.Om) - math.degrees(gstime(jd + fraction))) % 360.0
    return (
        circular_difference(earth_fixed_node, public_node),
        abs(inclination - public_inclination),
    )


def position_scores(
    samples: list[sqlite3.Row], history: dict[int, PublicSatellite]
) -> list[tuple[float, int]]:
    scores = []
    for norad_id, public in history.items():
        distances = []
        for row in samples:
            # ECEF is the orbital state reported for bdtTime; receiver
            # localTime only records when that state reached the collector.
            epoch = bdt_to_utc(int(row["bdtTime"]))
            try:
                predicted, _ = propagate_ecef(public.nearest(epoch), epoch)
            except ValueError:
                distances = []
                break
            observed = np.array(
                [row["ecefPx"], row["ecefPy"], row["ecefPz"]], dtype=float
            ) / 1000.0
            distances.append(float(np.linalg.norm(predicted - observed)))
        if distances:
            scores.append((float(np.median(distances)), norad_id))
    return sorted(scores)


def load_receiver_location(
    connection: sqlite3.Connection, terminal_id: str
) -> tuple[float, float, float, int]:
    terminal_clause = (
        "terminalId = ? AND "
        if table_has_column(connection, "position_data", "terminalId")
        else ""
    )
    parameters = (terminal_id,) if terminal_clause else ()
    row = connection.execute(
        f"""
        SELECT AVG(posLatitude) AS latitude,
               AVG(posLongitude) AS longitude,
               AVG(altitude) AS altitude_m,
               COUNT(*) AS samples,
               MAX(posLatitude) - MIN(posLatitude) AS latitude_span,
               MAX(posLongitude) - MIN(posLongitude) AS longitude_span
        FROM position_data
        WHERE {terminal_clause}posLatitude IS NOT NULL AND posLongitude IS NOT NULL
          AND posLatitude != 0 AND posLongitude != 0
        """,
        parameters,
    ).fetchone()
    if row is None or int(row["samples"] or 0) == 0:
        raise ValueError(f"receiver coordinates not found for terminal {terminal_id}")
    if float(row["latitude_span"] or 0) > 0.01 or float(row["longitude_span"] or 0) > 0.01:
        raise ValueError("receiver coordinates are not spatially stable")
    return (
        float(row["latitude"]),
        float(row["longitude"]),
        float(row["altitude_m"] or 0.0),
        int(row["samples"]),
    )


def receiver_geometry(
    latitude_deg: float, longitude_deg: float, altitude_m: float
) -> tuple[np.ndarray, np.ndarray]:
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    altitude_km = altitude_m / 1000.0
    semi_major_km = 6378.137
    eccentricity_squared = 6.69437999014e-3
    prime_vertical = semi_major_km / math.sqrt(
        1.0 - eccentricity_squared * math.sin(latitude) ** 2
    )
    receiver = np.array(
        [
            (prime_vertical + altitude_km) * math.cos(latitude) * math.cos(longitude),
            (prime_vertical + altitude_km) * math.cos(latitude) * math.sin(longitude),
            (
                prime_vertical * (1.0 - eccentricity_squared) + altitude_km
            ) * math.sin(latitude),
        ]
    )
    up = np.array(
        [
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        ]
    )
    return receiver, up


def elevation(
    public: PublicSatellite,
    epoch: datetime,
    receiver: np.ndarray,
    up: np.ndarray,
) -> float:
    satellite, _ = propagate_ecef(public.nearest(epoch), epoch)
    relative = satellite - receiver
    return math.degrees(math.asin(float(np.dot(relative, up) / np.linalg.norm(relative))))


def visibility_scores(
    centers: list[datetime],
    candidates: list[int],
    history: dict[int, PublicSatellite],
    receiver: np.ndarray,
    up: np.ndarray,
) -> list[dict]:
    scores = []
    for norad_id in candidates:
        try:
            elevations = [
                elevation(history[norad_id], timestamp, receiver, up)
                for timestamp in centers
            ]
        except ValueError:
            continue
        positive = sum(value >= 0 for value in elevations)
        score = positive * 100.0 + float(np.median(elevations)) + 0.1 * float(np.mean(elevations))
        scores.append(
            {
                "score": score,
                "norad_id": norad_id,
                "positive": positive,
                "median": float(np.median(elevations)),
                "minimum": min(elevations),
            }
        )
    return sorted(scores, key=lambda row: row["score"], reverse=True)


def analyze_version(
    spec: VersionSpec,
    connection: sqlite3.Connection,
    terminal_id: str,
    receiver: np.ndarray,
    up: np.ndarray,
) -> list[dict]:
    table = parse_let_table(spec.let_path)
    history = load_history(spec.history_path)
    positions = load_position_samples(connection, spec, limit=7)
    passes = load_phy_pass_centers(connection, spec, terminal_id, max_passes=12)
    rows = []

    for record in table.records:
        orbit_residuals = {
            norad_id: residual
            for norad_id, public in history.items()
            if (residual := let_orbit_residual(record, public)) is not None
        }
        compatible = [
            norad_id
            for norad_id, (node_error, inclination_error) in orbit_residuals.items()
            if node_error <= 0.30 and inclination_error <= 0.08
        ]

        position_result = None
        if record.satellite_id in positions:
            scores = position_scores(positions[record.satellite_id], history)
            if scores:
                position_result = {
                    "norad_id": scores[0][1],
                    "distance": scores[0][0],
                    "margin": scores[1][0] - scores[0][0] if len(scores) > 1 else float("inf"),
                    "samples": len(positions[record.satellite_id]),
                }

        visibility_result = None
        if record.satellite_id in passes and compatible:
            scores = visibility_scores(
                passes[record.satellite_id], compatible, history, receiver, up
            )
            if scores:
                visibility_result = {
                    **scores[0],
                    "margin": scores[0]["score"] - scores[1]["score"] if len(scores) > 1 else float("inf"),
                    "passes": len(passes[record.satellite_id]),
                }

        position_high = bool(
            position_result
            and position_result["distance"] <= 750.0
            and position_result["margin"] >= 400.0
            and position_result["norad_id"] in compatible
        )
        visibility_high = bool(
            visibility_result
            and visibility_result["passes"] >= 2
            and visibility_result["positive"] == visibility_result["passes"]
            and visibility_result["median"] >= 5.0
            and visibility_result["margin"] >= 5.0
        )

        selected = None
        evidence = ""
        status = "unresolved"
        if position_high and visibility_high:
            if position_result["norad_id"] == visibility_result["norad_id"]:
                selected = position_result["norad_id"]
                evidence = "historical_tle+ecef+let_orbit+phy_visibility"
                status = "accepted"
            else:
                evidence = "ecef_visibility_conflict"
        elif position_high:
            selected = position_result["norad_id"]
            evidence = "historical_tle+ecef+let_orbit"
            status = "accepted"
        elif visibility_high:
            selected = visibility_result["norad_id"]
            evidence = "historical_tle+let_orbit+phy_visibility"
            status = "accepted"
        else:
            candidates = []
            if position_result and position_result["norad_id"] in compatible:
                candidates.append(position_result["norad_id"])
            if visibility_result:
                candidates.append(visibility_result["norad_id"])
            if candidates and len(set(candidates)) == 1:
                selected = candidates[0]
                evidence = "insufficient_margin_or_pass_count"
                status = "provisional"

        public = history.get(selected) if selected is not None else None
        node_error, inclination_error = orbit_residuals.get(selected, (None, None))
        rows.append(
            {
                "let_version": spec.version,
                "raw_satellite_id": record.satellite_id,
                "norad_id": selected if selected is not None else "",
                "physical_name": public.name if public else "",
                "object_id": public.object_id if public else "",
                "status": status,
                "evidence": evidence,
                "let_compatible_candidates": len(compatible),
                "let_node_error_deg": node_error,
                "let_inclination_error_deg": inclination_error,
                "ecef_samples": position_result["samples"] if position_result else 0,
                "ecef_median_error_km": position_result["distance"] if position_result else "",
                "ecef_margin_km": position_result["margin"] if position_result else "",
                "phy_passes": visibility_result["passes"] if visibility_result else 0,
                "visibility_min_elevation_deg": visibility_result["minimum"] if visibility_result else "",
                "visibility_median_elevation_deg": visibility_result["median"] if visibility_result else "",
                "visibility_score_margin": visibility_result["margin"] if visibility_result else "",
            }
        )
    return rows


def propagate_numeric_continuity(rows: list[dict], specs: list[VersionSpec]) -> None:
    """Use an accepted later identity only when the older LET orbit corroborates it."""
    accepted_by_raw: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        if row["status"] == "accepted":
            accepted_by_raw[int(row["raw_satellite_id"])][int(row["norad_id"])] += 1

    spec_by_version = {spec.version: spec for spec in specs}
    history_cache = {spec.version: load_history(spec.history_path) for spec in specs}
    table_cache = {spec.version: parse_let_table(spec.let_path) for spec in specs}
    for row in rows:
        if row["status"] != "unresolved":
            continue
        raw_id = int(row["raw_satellite_id"])
        candidates = accepted_by_raw.get(raw_id)
        if not candidates or len(candidates) != 1:
            continue
        norad_id = next(iter(candidates))
        history = history_cache[row["let_version"]]
        if norad_id not in history:
            continue
        record = table_cache[row["let_version"]].by_id[raw_id]
        residual = let_orbit_residual(record, history[norad_id])
        if residual is None or residual[0] > 0.30 or residual[1] > 0.08:
            continue
        public = history[norad_id]
        row.update(
            {
                "norad_id": norad_id,
                "physical_name": public.name,
                "object_id": public.object_id,
                "status": "provisional",
                "evidence": "numeric_continuity+historical_tle+let_orbit",
                "let_node_error_deg": residual[0],
                "let_inclination_error_deg": residual[1],
            }
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("/home/wdz/satellite_data/satellite_data.db"),
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=MODULE_DIR / "analysis" / "history_gp",
    )
    parser.add_argument(
        "--latest-let-path",
        type=Path,
        help=(
            "Use a newer LET file as the current 0727 catalog while retaining "
            "the 0727 version label for canonical mapping."
        ),
    )
    parser.add_argument(
        "--latest-let-version",
        help=(
            "Version label for a newly installed LET catalog. Must be used with "
            "--latest-let-start-local-time; the previous catalog is retained."
        ),
    )
    parser.add_argument(
        "--latest-let-start-local-time",
        help=(
            "Local installation time of a newly versioned LET catalog, for "
            "example 2026-08-20T09:40:00."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_DIR / "analysis" / "latest",
    )
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    args = parser.parse_args()

    specs = default_specs(args.history_dir.resolve())
    versioning_new_catalog = bool(
        args.latest_let_version or args.latest_let_start_local_time
    )
    if versioning_new_catalog and not (
        args.latest_let_path
        and args.latest_let_version
        and args.latest_let_start_local_time
    ):
        parser.error(
            "--latest-let-path, --latest-let-version, and "
            "--latest-let-start-local-time must be provided together"
        )
    if args.latest_let_path is not None and versioning_new_catalog:
        previous = specs[-1]
        specs[-1] = VersionSpec(
            previous.version,
            previous.let_path,
            previous.history_path,
            previous.start_local_time,
            args.latest_let_start_local_time,
        )
        specs.append(
            VersionSpec(
                args.latest_let_version,
                args.latest_let_path.resolve(),
                args.history_dir.resolve()
                / f"qianfan_gp_history_{args.latest_let_version}.csv",
                args.latest_let_start_local_time,
                "9999-12-31",
            )
        )
    elif args.latest_let_path is not None:
        latest = specs[-1]
        specs[-1] = VersionSpec(
            latest.version,
            args.latest_let_path.resolve(),
            latest.history_path,
            latest.start_local_time,
            latest.stop_local_time,
        )
    connection = sqlite3.connect(f"file:{args.db_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        receiver_location = load_receiver_location(connection, args.terminal_id)
        receiver, up = receiver_geometry(*receiver_location[:3])
        print(
            "Receiver from database: "
            f"lat={receiver_location[0]:.6f}, lon={receiver_location[1]:.6f}, "
            f"altitude_m={receiver_location[2]:.2f}, samples={receiver_location[3]}",
            flush=True,
        )
        rows = []
        for spec in specs:
            print(f"Matching LET {spec.version}...", flush=True)
            rows.extend(
                analyze_version(
                    spec, connection, args.terminal_id, receiver, up
                )
            )
    finally:
        connection.close()

    propagate_numeric_continuity(rows, specs)
    collision_counts = Counter(
        (row["let_version"], row["norad_id"])
        for row in rows
        if row["status"] == "accepted" and row["norad_id"] != ""
    )
    for row in rows:
        key = (row["let_version"], row["norad_id"])
        row["accepted_same_version_aliases"] = collision_counts.get(key, 0)
        if row["status"] == "accepted" and collision_counts.get(key, 0) > 1:
            row["status"] = "provisional"
            row["evidence"] = f"same_version_identity_collision;{row['evidence']}"

    canonical_version = specs[-1].version
    canonical_id_column = f"canonical_{canonical_version}_satellite_id"
    latest_by_norad: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["let_version"] == canonical_version and row["norad_id"] != "":
            latest_by_norad[int(row["norad_id"])].append(row)
    for row in rows:
        row[canonical_id_column] = ""
        row["canonical_status"] = "unresolved"
        if row["norad_id"] == "":
            continue
        candidates = latest_by_norad.get(int(row["norad_id"]), [])
        accepted_candidates = [item for item in candidates if item["status"] == "accepted"]
        if len(accepted_candidates) == 1:
            row[canonical_id_column] = accepted_candidates[0]["raw_satellite_id"]
            row["canonical_status"] = (
                "accepted" if row["status"] == "accepted" else "provisional"
            )
        elif len(candidates) == 1:
            row[canonical_id_column] = candidates[0]["raw_satellite_id"]
            row["canonical_status"] = "provisional"

    output_dir = args.output_dir.resolve()
    mapping_path = output_dir / "historical_physical_mapping.csv"
    write_csv(mapping_path, rows)
    canonical_rows = [
        {
            "source_let_version": row["let_version"],
            "raw_satellite_id": row["raw_satellite_id"],
            "norad_id": row["norad_id"],
            "physical_name": row["physical_name"],
            "object_id": row["object_id"],
            canonical_id_column: row[canonical_id_column],
            "evidence": row["evidence"],
        }
        for row in rows
        if row["canonical_status"] == "accepted"
    ]
    canonical_path = output_dir / f"accepted_canonical_{canonical_version}_mapping.csv"
    if canonical_rows:
        write_csv(canonical_path, canonical_rows)
    summary = {
        "policy": "No source ID is overwritten; only multi-evidence rows are accepted.",
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "version_status_counts": {
            spec.version: dict(
                Counter(row["status"] for row in rows if row["let_version"] == spec.version)
            )
            for spec in specs
        },
        "accepted_unique_norad_ids": len(
            {int(row["norad_id"]) for row in rows if row["status"] == "accepted"}
        ),
        "same_version_collision_groups_downgraded": sum(
            value > 1 for value in collision_counts.values()
        ),
        "accepted_canonical_rows": sum(
            row["canonical_status"] == "accepted" for row in rows
        ),
        "receiver": {
            "terminal_id": args.terminal_id,
            "latitude_deg": receiver_location[0],
            "longitude_deg": receiver_location[1],
            "altitude_m": receiver_location[2],
            "database_samples": receiver_location[3],
        },
        "mapping_path": str(mapping_path),
        "accepted_canonical_mapping_path": str(canonical_path),
        "canonical_let_version": canonical_version,
    }
    summary_path = output_dir / "historical_physical_mapping_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
