#!/usr/bin/env python3
"""Calibrate terminal-ID matching from repeated PHY pass visibility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .match_historical_gp import (
        bdt_to_utc,
        default_specs,
        elevation,
        load_history,
        load_phy_pass_centers,
        load_receiver_location,
        receiver_geometry,
    )
except ImportError:
    from match_historical_gp import (
        bdt_to_utc,
        default_specs,
        elevation,
        load_history,
        load_phy_pass_centers,
        load_receiver_location,
        receiver_geometry,
    )

import sqlite3


def load_mapping(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["let_version"], int(row["raw_satellite_id"])): row
            for row in csv.DictReader(handle)
        }


def score_candidates(centers, history, receiver, up):
    rows = []
    for norad_id, public in history.items():
        elevations = []
        for timestamp in centers:
            try:
                elevations.append(
                    elevation(public, bdt_to_utc(timestamp), receiver, up)
                )
            except ValueError:
                elevations = []
                break
        if not elevations:
            continue
        positive = sum(value >= 0.0 for value in elevations)
        rows.append(
            {
                "norad_id": norad_id,
                "physical_name": public.name,
                "object_id": public.object_id,
                "visible_passes": positive,
                "visibility_fraction": positive / len(elevations),
                "minimum_elevation_deg": min(elevations),
                "median_elevation_deg": float(np.median(elevations)),
                "mean_elevation_deg": float(np.mean(elevations)),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["visible_passes"],
            row["minimum_elevation_deg"],
            row["median_elevation_deg"],
        ),
        reverse=True,
    )


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument(
        "--mapping-path", type=Path,
        default=module_dir / "analysis/latest/historical_physical_mapping.csv",
    )
    parser.add_argument(
        "--history-dir", type=Path,
        default=module_dir / "analysis/history_gp",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=module_dir / "analysis/phy_visibility_identity",
    )
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    parser.add_argument("--max-passes", type=int, default=20)
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_path)
    specs = default_specs(args.history_dir)
    connection = sqlite3.connect(f"file:{args.db_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        latitude, longitude, altitude, receiver_samples = load_receiver_location(
            connection, args.terminal_id
        )
        receiver, up = receiver_geometry(latitude, longitude, altitude)
        rows = []
        for spec in specs:
            history = load_history(spec.history_path)
            passes = load_phy_pass_centers(
                connection, spec, args.terminal_id, args.max_passes
            )
            for raw_id, centers in sorted(passes.items()):
                scores = score_candidates(centers, history, receiver, up)
                if not scores:
                    continue
                source = mapping.get((spec.version, raw_id), {})
                expected = (
                    int(source["norad_id"])
                    if source.get("status") == "accepted" and source.get("norad_id")
                    else None
                )
                expected_rank = next(
                    (i + 1 for i, row in enumerate(scores) if row["norad_id"] == expected),
                    None,
                )
                expected_score = next(
                    (row for row in scores if row["norad_id"] == expected), None
                )
                best = scores[0]
                second = scores[1] if len(scores) > 1 else None
                rows.append(
                    {
                        "let_version": spec.version,
                        "raw_satellite_id": raw_id,
                        "source_status": source.get("status", "outside_let_mapping"),
                        "source_evidence": source.get("evidence", ""),
                        "source_norad_id": expected if expected is not None else "",
                        "pass_count": len(centers),
                        **best,
                        "second_norad_id": second["norad_id"] if second else "",
                        "second_visible_passes": second["visible_passes"] if second else "",
                        "second_minimum_elevation_deg": (
                            second["minimum_elevation_deg"] if second else ""
                        ),
                        "expected_rank": expected_rank if expected_rank is not None else "",
                        "expected_visible_passes": (
                            expected_score["visible_passes"] if expected_score else ""
                        ),
                        "expected_minimum_elevation_deg": (
                            expected_score["minimum_elevation_deg"] if expected_score else ""
                        ),
                        "top1_matches_accepted": (
                            expected is not None and best["norad_id"] == expected
                        ),
                    }
                )
    finally:
        connection.close()

    calibration = [
        row for row in rows
        if row["source_status"] == "accepted" and "ecef" in row["source_evidence"]
    ]
    correct = [row for row in calibration if row["top1_matches_accepted"]]
    all_visible = [
        row for row in calibration
        if row["expected_visible_passes"] == row["pass_count"]
    ]
    accepted_assignments = {
        (version, int(raw["norad_id"])): int(raw["raw_satellite_id"])
        for (version, _), raw in mapping.items()
        if raw.get("status") == "accepted" and raw.get("norad_id")
    }
    strong_candidates = []
    for row in rows:
        margin = (
            row["minimum_elevation_deg"] - row["second_minimum_elevation_deg"]
        )
        if not (
            row["source_status"] != "accepted"
            and row["pass_count"] >= 5
            and row["visible_passes"] == row["pass_count"]
            and margin >= 5.0
        ):
            continue
        conflict = accepted_assignments.get(
            (str(row["let_version"]), int(row["norad_id"]))
        )
        strong_candidates.append(
            {
                **row,
                "minimum_elevation_margin_deg": margin,
                "same_version_accepted_raw_id": conflict or "",
                "recommendation": (
                    "reject_same_version_collision"
                    if conflict and conflict != row["raw_satellite_id"]
                    else "candidate_needs_independent_let_orbit_validation"
                ),
            }
        )
    summary = {
        "receiver": {
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "altitude_m": altitude,
            "samples": receiver_samples,
        },
        "scored_identities": len(rows),
        "ecef_calibration_identities": len(calibration),
        "expected_visible_at_all_pass_centers": len(all_visible),
        "expected_visible_fraction": (
            len(all_visible) / len(calibration) if calibration else None
        ),
        "top1_correct": len(correct),
        "top1_accuracy": len(correct) / len(calibration) if calibration else None,
        "strong_nonaccepted_candidates": len(strong_candidates),
        "strong_candidates_without_same_version_collision": sum(
            row["recommendation"]
            == "candidate_needs_independent_let_orbit_validation"
            for row in strong_candidates
        ),
        "policy": "Calibration only; no database or mapping is modified.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "phy_visibility_identity_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "strong_visibility_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(strong_candidates[0]))
        writer.writeheader()
        writer.writerows(strong_candidates)
    (args.output_dir / "phy_visibility_identity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
