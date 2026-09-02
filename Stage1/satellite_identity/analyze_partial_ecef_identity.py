#!/usr/bin/env python3
"""Evaluate whether partial terminal ECEF rows can identify physical satellites."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .match_historical_gp import bdt_to_utc, load_history, propagate_ecef
    from .position_quality import version_for_local_time
except ImportError:
    from match_historical_gp import bdt_to_utc, load_history, propagate_ecef
    from position_quality import version_for_local_time


@dataclass(frozen=True)
class PartialSample:
    epoch_ms: int
    observed_km: np.ndarray
    observed_mask: np.ndarray


def finite_nonzero(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number != 0.0


def load_mapping(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["let_version"], int(row["raw_satellite_id"])): row
            for row in csv.DictReader(handle)
        }


def sample_partial_rows(
    csv_path: Path,
    samples_per_identity: int,
    chunk_size: int,
    seed: int,
) -> tuple[dict[tuple[str, int], list[PartialSample]], dict[str, int]]:
    rng = random.Random(seed)
    reservoirs: dict[tuple[str, int], list[PartialSample]] = defaultdict(list)
    seen: dict[tuple[str, int], int] = defaultdict(int)
    counters = defaultdict(int)
    columns = ["localTime", "satId", "bdtTime", "ecefPx", "ecefPy", "ecefPz"]
    for frame in pd.read_csv(
        csv_path, usecols=columns, chunksize=chunk_size, low_memory=False
    ):
        for row in frame.itertuples(index=False):
            counters["rows_scanned"] += 1
            try:
                satellite_id = int(row.satId)
                epoch_ms = int(float(row.bdtTime))
                version = version_for_local_time(str(row.localTime))
            except (TypeError, ValueError, OverflowError):
                counters["invalid_identity_or_time"] += 1
                continue
            values = np.array([row.ecefPx, row.ecefPy, row.ecefPz], dtype=float)
            mask = np.array([finite_nonzero(value) for value in values], dtype=bool)
            if int(mask.sum()) < 2 or bool(mask.all()):
                counters["not_partial_two_axis_ecef"] += 1
                continue
            sample = PartialSample(epoch_ms, values / 1000.0, mask)
            key = (version, satellite_id)
            seen[key] += 1
            reservoir = reservoirs[key]
            if len(reservoir) < samples_per_identity:
                reservoir.append(sample)
            else:
                index = rng.randrange(seen[key])
                if index < samples_per_identity:
                    reservoir[index] = sample
            counters["partial_rows"] += 1
    counters["partial_identities"] = len(reservoirs)
    return reservoirs, dict(counters)


def nearest_element(public, epoch):
    return min(
        public.elements,
        key=lambda element: abs((element.epoch - epoch).total_seconds()),
    )


def score_identity(
    samples: list[PartialSample],
    history,
    max_tle_age_days: float,
    min_valid_fraction: float,
    score_mode: str,
) -> list[dict]:
    required = max(3, math.ceil(len(samples) * min_valid_fraction))
    scores = []
    for norad_id, public in history.items():
        residuals = []
        ages = []
        for sample in samples:
            epoch = bdt_to_utc(sample.epoch_ms).astimezone(timezone.utc)
            element = nearest_element(public, epoch)
            age_days = abs((element.epoch - epoch).total_seconds()) / 86400.0
            if age_days > max_tle_age_days:
                continue
            try:
                predicted_km, _ = propagate_ecef(element.satellite, epoch)
            except ValueError:
                continue
            if score_mode == "longitude":
                observed_longitude = math.degrees(
                    math.atan2(sample.observed_km[1], sample.observed_km[0])
                )
                predicted_longitude = math.degrees(
                    math.atan2(predicted_km[1], predicted_km[0])
                )
                residual = abs(
                    (predicted_longitude - observed_longitude + 180.0) % 360.0
                    - 180.0
                )
            else:
                difference = (
                    predicted_km[sample.observed_mask]
                    - sample.observed_km[sample.observed_mask]
                )
                residual = float(np.sqrt(np.mean(difference * difference)))
            residuals.append(residual)
            ages.append(age_days)
        if len(residuals) >= required:
            scores.append(
                {
                    "norad_id": norad_id,
                    "physical_name": public.name,
                    "object_id": public.object_id,
                    "median_residual": float(np.median(residuals)),
                    "p90_residual": float(np.percentile(residuals, 90)),
                    "valid_samples": len(residuals),
                    "median_tle_age_days": float(np.median(ages)),
                }
            )
    return sorted(scores, key=lambda item: item["median_residual"])


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
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
        default=module_dir / "analysis/partial_ecef_identity",
    )
    parser.add_argument("--samples-per-identity", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-tle-age-days", type=float, default=7.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.7)
    parser.add_argument(
        "--score-mode", choices=("partial_ecef", "longitude"),
        default="partial_ecef",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_path)
    samples, source_stats = sample_partial_rows(
        args.csv_path, args.samples_per_identity, args.chunk_size, args.seed
    )
    histories = {
        version: load_history(args.history_dir / f"qianfan_gp_history_{version}.csv")
        for version in ("0401", "0429", "0611", "0727")
    }

    rows = []
    for (version, raw_id), identity_samples in sorted(samples.items()):
        scores = score_identity(
            identity_samples,
            histories[version],
            args.max_tle_age_days,
            args.min_valid_fraction,
            args.score_mode,
        )
        if not scores:
            continue
        best = scores[0]
        second_error = (
            scores[1]["median_residual"] if len(scores) > 1 else math.inf
        )
        source = mapping.get((version, raw_id), {})
        expected = int(source["norad_id"]) if source.get("status") == "accepted" and source.get("norad_id") else None
        expected_rank = next(
            (index + 1 for index, item in enumerate(scores) if item["norad_id"] == expected),
            None,
        )
        expected_error = next(
            (
                item["median_residual"]
                for item in scores if item["norad_id"] == expected
            ),
            None,
        )
        rows.append(
            {
                "let_version": version,
                "raw_satellite_id": raw_id,
                "source_status": source.get("status", "outside_let_mapping"),
                "source_norad_id": expected if expected is not None else "",
                "sample_count": len(identity_samples),
                **best,
                "second_residual": second_error,
                "margin": second_error - best["median_residual"],
                "error_ratio": (
                    second_error / max(best["median_residual"], 1e-9)
                ),
                "expected_rank": expected_rank if expected_rank is not None else "",
                "expected_residual": expected_error if expected_error is not None else "",
                "top1_matches_accepted": expected is not None and best["norad_id"] == expected,
            }
        )

    accepted = [row for row in rows if row["source_status"] == "accepted"]
    correct = [row for row in accepted if row["top1_matches_accepted"]]
    unresolved = [row for row in rows if row["source_status"] != "accepted"]
    summary = {
        "source": source_stats,
        "scored_identities": len(rows),
        "accepted_calibration_identities": len(accepted),
        "accepted_top1_correct": len(correct),
        "accepted_top1_accuracy": len(correct) / len(accepted) if accepted else None,
        "correct_best_residual": {
            "median": percentile([row["median_residual"] for row in correct], 50),
            "p90": percentile([row["median_residual"] for row in correct], 90),
            "max": max((row["median_residual"] for row in correct), default=None),
        },
        "correct_margin": {
            "median": percentile([row["margin"] for row in correct], 50),
            "p10": percentile([row["margin"] for row in correct], 10),
        },
        "nonaccepted_candidates": len(unresolved),
        "score_mode": args.score_mode,
        "residual_unit": "degree" if args.score_mode == "longitude" else "km",
        "policy": (
            "Calibration only. No database or canonical mapping is modified by this script."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / f"{args.score_mode}_identity_scores.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / f"{args.score_mode}_identity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
