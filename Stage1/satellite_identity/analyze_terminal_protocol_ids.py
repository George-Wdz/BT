#!/usr/bin/env python3
"""Audit protocol satellite IDs shared by terminals 001, 002, and 003."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from apply_accepted_mapping import VERSION_RANGES, load_mapping


TERMINALS = {
    "001": "01-31-0005-0001",
    "002": "01-31-0005-0002",
    "003": "01-31-0005-0003",
}


def read_legacy(path: Path) -> pd.DataFrame:
    query = """
        SELECT localTime, satelliteId
        FROM phy_data
        WHERE satelliteId BETWEEN 1 AND 10000
          AND phyRssi IS NOT NULL AND rssi IS NOT NULL
          AND snr IS NOT NULL AND snr != 255
          AND lastCniValue IS NOT NULL
        ORDER BY localTime
    """
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(query, connection)
    frame["time"] = pd.to_datetime(frame.pop("localTime"), errors="coerce")
    frame = frame.dropna(subset=["time", "satelliteId"])
    frame["protocol_satellite_id"] = frame.pop("satelliteId").astype("int64")
    return frame


def read_new(path: Path, terminal_id: str) -> pd.DataFrame:
    query = """
        SELECT localTime, trackNo, phaseNo
        FROM phy_bb_data
        WHERE terminalId = ? AND validMeasBb = 1
          AND trackNo IS NOT NULL AND phaseNo IS NOT NULL
          AND snr IS NOT NULL AND snr != 0
        ORDER BY localTime
    """
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(query, connection, params=[terminal_id])
    frame["time"] = pd.to_datetime(frame.pop("localTime"), errors="coerce")
    frame = frame.dropna(subset=["time", "trackNo", "phaseNo"])
    frame["protocol_satellite_id"] = (
        frame.pop("trackNo").astype("int64") * 256
        + frame.pop("phaseNo").astype("int64")
    )
    return frame.loc[frame["protocol_satellite_id"].between(1, 10000)].copy()


def version_for_time(values: pd.Series) -> np.ndarray:
    result = np.full(len(values), "0401", dtype=object)
    for version, (start, _) in VERSION_RANGES.items():
        if version != "0401":
            result[values.ge(pd.Timestamp(start)).to_numpy()] = version
    return result


def canonicalize(frame: pd.DataFrame, mappings: dict[str, dict[int, int]]) -> pd.DataFrame:
    output = frame.copy()
    output["let_version"] = version_for_time(output["time"])
    output["canonical_protocol_id"] = [
        mappings.get(version, {}).get(int(satellite_id), int(satellite_id))
        for version, satellite_id in zip(
            output["let_version"], output["protocol_satellite_id"]
        )
    ]
    output["minute"] = output["time"].dt.floor("min")
    return output.drop_duplicates(["minute", "canonical_protocol_id"])


def compare_minutes(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    left_sets = left.groupby("minute")["canonical_protocol_id"].agg(set)
    right_sets = right.groupby("minute")["canonical_protocol_id"].agg(set)
    minutes = left_sets.index.intersection(right_sets.index)
    intersections = np.asarray(
        [len(left_sets[minute] & right_sets[minute]) for minute in minutes],
        dtype=np.int64,
    )
    unions = np.asarray(
        [len(left_sets[minute] | right_sets[minute]) for minute in minutes],
        dtype=np.int64,
    )
    return {
        "overlapping_minutes": int(len(minutes)),
        "minutes_with_at_least_one_shared_id": int((intersections > 0).sum()),
        "minutes_with_at_least_one_shared_id_ratio": (
            float((intersections > 0).mean()) if len(minutes) else None
        ),
        "shared_minute_id_pairs": int(intersections.sum()),
        "mean_minute_id_jaccard": (
            float(np.mean(intersections / unions)) if len(minutes) else None
        ),
        "left_mean_ids_per_minute": float(left_sets.loc[minutes].map(len).mean()),
        "right_mean_ids_per_minute": float(right_sets.loc[minutes].map(len).mean()),
    }


def temporal_candidates(source: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    pairs = source[["minute", "canonical_protocol_id"]].merge(
        reference[["minute", "canonical_protocol_id"]],
        on="minute",
        suffixes=("_source", "_reference"),
    )
    pairs = pairs.loc[
        pairs["canonical_protocol_id_source"].ne(
            pairs["canonical_protocol_id_reference"]
        )
    ]
    counts = (
        pairs.groupby(
            ["canonical_protocol_id_source", "canonical_protocol_id_reference"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "cooccurring_minutes"})
    )
    if counts.empty:
        return counts
    source_total = counts.groupby("canonical_protocol_id_source")[
        "cooccurring_minutes"
    ].sum()
    reference_total = counts.groupby("canonical_protocol_id_reference")[
        "cooccurring_minutes"
    ].sum()
    counts["source_share"] = counts["cooccurring_minutes"] / counts[
        "canonical_protocol_id_source"
    ].map(source_total)
    counts["reverse_share"] = counts["cooccurring_minutes"] / counts[
        "canonical_protocol_id_reference"
    ].map(reference_total)
    counts = counts.sort_values(
        ["canonical_protocol_id_source", "cooccurring_minutes"],
        ascending=[True, False],
    ).groupby("canonical_protocol_id_source", as_index=False).head(1)
    counts["decision"] = "not_mapped_temporal_cooccurrence_only"
    return counts.sort_values("cooccurring_minutes", ascending=False)


def load_physical_mapping(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["let_version"].astype(str).eq("727")].copy()
    return frame.sort_values("raw_satellite_id").drop_duplicates("raw_satellite_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-001-db", type=Path, required=True)
    parser.add_argument("--terminal-002-db", type=Path, required=True)
    parser.add_argument("--terminal-003-db", type=Path, required=True)
    parser.add_argument("--operational-mapping", type=Path, required=True)
    parser.add_argument("--physical-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mappings = load_mapping(args.operational_mapping)
    frames = {
        "001": canonicalize(read_legacy(args.terminal_001_db), mappings),
        "002": canonicalize(
            read_new(args.terminal_002_db, TERMINALS["002"]), mappings
        ),
        "003": canonicalize(
            read_new(args.terminal_003_db, TERMINALS["003"]), mappings
        ),
    }

    pair_summary = {}
    candidate_frames = []
    for left, right in (("001", "002"), ("001", "003"), ("002", "003")):
        pair_summary[f"{left}_{right}"] = compare_minutes(frames[left], frames[right])
        candidates = temporal_candidates(frames[right], frames[left])
        candidates.insert(0, "source_terminal", right)
        candidates.insert(1, "reference_terminal", left)
        candidate_frames.append(candidates)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates.to_csv(args.output_dir / "temporal_cross_id_candidates.csv", index=False)

    physical = load_physical_mapping(args.physical_mapping).rename(
        columns={"raw_satellite_id": "canonical_protocol_id"}
    )
    all_ids = sorted(
        set().union(
            *(set(frame["canonical_protocol_id"].unique()) for frame in frames.values())
        )
    )
    inventory = pd.DataFrame({"canonical_protocol_id": all_ids})
    for terminal, frame in frames.items():
        counts = frame["canonical_protocol_id"].value_counts()
        inventory[f"observed_{terminal}"] = inventory["canonical_protocol_id"].isin(
            counts.index
        )
        inventory[f"minute_id_rows_{terminal}"] = (
            inventory["canonical_protocol_id"].map(counts).fillna(0).astype(int)
        )
    inventory = inventory.merge(
        physical[
            [
                "canonical_protocol_id", "norad_id", "physical_name", "object_id",
                "status", "evidence", "ecef_median_error_km", "ecef_margin_km",
            ]
        ],
        on="canonical_protocol_id",
        how="left",
    )
    inventory["status"] = inventory["status"].fillna("outside_latest_let_or_unresolved")
    inventory.to_csv(args.output_dir / "three_terminal_physical_id_inventory.csv", index=False)

    summary = {
        "protocol_formula_new_terminals": "trackNo * 256 + phaseNo",
        "canonicalization": "same LET-version mapping as terminal 001",
        "source": {
            terminal: {
                "minute_id_rows": int(len(frame)),
                "protocol_ids": int(frame["canonical_protocol_id"].nunique()),
                "start": frame["time"].min().isoformat(),
                "end": frame["time"].max().isoformat(),
            }
            for terminal, frame in frames.items()
        },
        "pairs": pair_summary,
        "physical_status_counts": inventory["status"].value_counts().to_dict(),
        "accepted_physical_ids_observed": int(
            inventory.loc[inventory["status"].eq("accepted"), "norad_id"].nunique()
        ),
        "accepted_cross_id_remaps": 0,
        "cross_id_policy": (
            "Temporal co-occurrence is diagnostic only; no unequal protocol IDs are "
            "merged without orbit or ECEF identity evidence."
        ),
    }
    (args.output_dir / "terminal_protocol_id_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
