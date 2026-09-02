#!/usr/bin/env python3
"""Reconstruct invalid terminal positions from validated historical GP/TLE."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter
from datetime import timezone
from pathlib import Path

import pandas as pd

try:
    from .match_historical_gp import bdt_to_utc, load_history, propagate_ecef
    from .position_quality import ecef_to_geodetic, position_quality_reason, version_for_local_time
except ImportError:
    from match_historical_gp import bdt_to_utc, load_history, propagate_ecef
    from position_quality import ecef_to_geodetic, position_quality_reason, version_for_local_time


PROVENANCE_FIELDS = [
    "positionSource", "positionQuality", "positionQualityReason",
    "sourceSatId", "noradId", "tleEpoch", "tleAgeDays",
]


def load_visibility_validations(path: Path | None) -> dict[tuple[str, int], int]:
    if path is None or not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("recommendation") != "candidate_needs_independent_let_orbit_validation":
                continue
            version = str(row["let_version"]).zfill(4)
            result[(version, int(row["raw_satellite_id"]))] = int(row["norad_id"])
    return result


def load_validated_identities(
    path: Path,
    max_anchor_error_km: float,
    visibility_validations: dict[tuple[str, int], int],
) -> tuple[dict[tuple[str, int], dict], Counter]:
    result = {}
    evidence_counts = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "accepted" or not row["norad_id"]:
                continue
            key = (row["let_version"], int(row["raw_satellite_id"]))
            norad_id = int(row["norad_id"])
            try:
                error = float(row["ecef_median_error_km"])
            except (TypeError, ValueError):
                error = math.inf
            if error <= max_anchor_error_km:
                result[key] = row
                evidence_counts["measured_ecef_anchor"] += 1
            elif visibility_validations.get(key) == norad_id:
                result[key] = row
                evidence_counts["let_orbit_and_repeated_phy_visibility"] += 1
    return result, evidence_counts


def load_canonical_ids(path: Path) -> dict[tuple[str, int], int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["source_let_version"], int(row["raw_satellite_id"])):
                int(row["canonical_0727_satellite_id"])
            for row in csv.DictReader(handle)
        }


def nearest_element(public, epoch):
    epochs = [item.epoch for item in public.elements]
    index = bisect.bisect_left(epochs, epoch)
    choices = {max(0, index - 1), min(len(epochs) - 1, index)}
    return min(choices, key=lambda idx: abs((epochs[idx] - epoch).total_seconds()))


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--mapping-path", type=Path,
        default=module_dir / "analysis/latest/historical_physical_mapping.csv",
    )
    parser.add_argument(
        "--canonical-map-path", type=Path,
        default=module_dir / "analysis/latest/operational_canonical_0727_mapping.csv",
    )
    parser.add_argument(
        "--history-dir", type=Path,
        default=module_dir / "analysis/history_gp",
    )
    parser.add_argument(
        "--visibility-validation-path", type=Path,
        default=module_dir / "analysis/phy_visibility_identity/strong_visibility_candidates.csv",
    )
    parser.add_argument("--max-tle-age-days", type=float, default=7.0)
    parser.add_argument("--max-anchor-error-km", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    visibility_validations = load_visibility_validations(
        args.visibility_validation_path
    )
    identities, identity_evidence_counts = load_validated_identities(
        args.mapping_path, args.max_anchor_error_km, visibility_validations
    )
    canonical_ids = load_canonical_ids(args.canonical_map_path)
    histories = {
        version: load_history(args.history_dir / f"qianfan_gp_history_{version}.csv")
        for version in ("0401", "0429", "0611", "0727")
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    report = Counter()
    writer = None
    with args.output_path.open("w", encoding="utf-8", newline="") as output:
        for frame in pd.read_csv(args.csv_path, chunksize=args.chunk_size, low_memory=False):
            if writer is None:
                fields = [field for field in frame.columns if field != "id"] + PROVENANCE_FIELDS
                writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
            for raw in frame.to_dict("records"):
                report["rows_scanned"] += 1
                reason = position_quality_reason(raw)
                if reason is None:
                    report["already_valid"] += 1
                    continue
                try:
                    source_satellite_id = int(raw["satId"])
                    version = version_for_local_time(raw["localTime"])
                except (TypeError, ValueError):
                    report["invalid_identity_fields"] += 1
                    continue
                identity = identities.get((version, source_satellite_id))
                if identity is None:
                    report["identity_not_validated"] += 1
                    continue
                norad_id = int(identity["norad_id"])
                public = histories[version].get(norad_id)
                if public is None:
                    report["tle_not_downloaded"] += 1
                    continue
                try:
                    epoch = bdt_to_utc(int(raw["bdtTime"])).astimezone(timezone.utc)
                except (TypeError, ValueError):
                    report["invalid_bdtTime"] += 1
                    continue
                element_index = nearest_element(public, epoch)
                element = public.elements[element_index]
                tle_age_days = abs((element.epoch - epoch).total_seconds()) / 86400.0
                if tle_age_days > args.max_tle_age_days:
                    report["tle_too_far_from_measurement"] += 1
                    continue
                try:
                    ecef_km, _ = propagate_ecef(element.satellite, epoch)
                except ValueError:
                    report["sgp4_error"] += 1
                    continue
                ecef_m = tuple(float(value * 1000.0) for value in ecef_km)
                longitude, latitude, altitude = ecef_to_geodetic(ecef_m)
                repaired = dict(raw)
                repaired.update({
                    "satId": canonical_ids.get((version, source_satellite_id), source_satellite_id),
                    "ecefPx": ecef_m[0], "ecefPy": ecef_m[1], "ecefPz": ecef_m[2],
                    "longitude": longitude, "latitude": latitude, "satAltitude": altitude,
                    "positionSource": "tle_reconstructed",
                    "positionQuality": "valid",
                    "positionQualityReason": "",
                    "sourceSatId": source_satellite_id,
                    "noradId": norad_id,
                    "tleEpoch": element.epoch.isoformat(),
                    "tleAgeDays": round(tle_age_days, 6),
                })
                if position_quality_reason(repaired) is not None:
                    report["reconstructed_position_failed_validation"] += 1
                    continue
                writer.writerow(repaired)
                report["repaired_rows"] += 1

    report_data = {
        **report,
        "input_csv": str(args.csv_path.resolve()),
        "output_csv": str(args.output_path.resolve()),
        "max_tle_age_days": args.max_tle_age_days,
        "max_anchor_error_km": args.max_anchor_error_km,
        "validated_identity_counts": dict(identity_evidence_counts),
        "visibility_validation_path": str(args.visibility_validation_path.resolve()),
        "policy": (
            "accepted identity with either a measured ECEF anchor or independently "
            "consistent LET orbit plus repeated PHY visibility, and nearby historical TLE"
        ),
    }
    report_path = args.output_path.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
