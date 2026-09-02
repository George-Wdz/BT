#!/usr/bin/env python3
"""Summarize per-terminal accuracy and cross-terminal inference consistency."""

from __future__ import annotations

import argparse
import json
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


TERMINALS = (
    "01-31-0005-0001",
    "01-31-0005-0002",
    "01-31-0005-0003",
)


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    value = left.corr(right)
    return None if pd.isna(value) else float(value)


def _classification(observed: pd.Series, probability: pd.Series) -> dict[str, float]:
    truth = observed > 0
    predicted = probability >= 0.5
    tp = int((truth & predicted).sum())
    tn = int((~truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(truth) if len(truth) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _regression(frame: pd.DataFrame) -> dict[str, float | int | None]:
    error = frame["reported_rainfall_mm"] - frame["observed_rainfall_mm"]
    return {
        "n": int(len(frame)),
        "mae_mm": float(error.abs().mean()) if len(frame) else None,
        "rmse_mm": float(np.sqrt(np.mean(np.square(error)))) if len(frame) else None,
        "pearson": _correlation(
            frame["reported_rainfall_mm"], frame["observed_rainfall_mm"]
        ),
        "observed_sum_mm": float(frame["observed_rainfall_mm"].sum()),
        "predicted_sum_mm": float(frame["reported_rainfall_mm"].sum()),
    }


def _pair_metrics(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    merged = left.merge(right, on="anchor", suffixes=("_left", "_right"))
    amount_gap = (
        merged["reported_rainfall_mm_left"]
        - merged["reported_rainfall_mm_right"]
    ).abs()
    probability_gap = (
        merged["rain_probability_left"] - merged["rain_probability_right"]
    ).abs()
    same_satellite = merged["satellite_id_left"] == merged["satellite_id_right"]
    observed_rain = merged[
        ["observed_rainfall_mm_left", "observed_rainfall_mm_right"]
    ].max(axis=1) > 0

    def subset(mask: pd.Series) -> dict:
        selected = merged.loc[mask]
        if selected.empty:
            return {"n": 0}
        return {
            "n": int(len(selected)),
            "amount_mae_mm": float(
                (
                    selected["reported_rainfall_mm_left"]
                    - selected["reported_rainfall_mm_right"]
                ).abs().mean()
            ),
            "amount_pearson": _correlation(
                selected["reported_rainfall_mm_left"],
                selected["reported_rainfall_mm_right"],
            ),
            "probability_mae": float(
                (
                    selected["rain_probability_left"]
                    - selected["rain_probability_right"]
                ).abs().mean()
            ),
            "rain_decision_agreement": float(
                (
                    (selected["rain_probability_left"] >= 0.5)
                    == (selected["rain_probability_right"] >= 0.5)
                ).mean()
            ),
        }

    return {
        "all_overlap": subset(pd.Series(True, index=merged.index)),
        "rainy_overlap": subset(observed_rain),
        "same_satellite_overlap": subset(same_satellite),
        "same_satellite_rainy_overlap": subset(same_satellite & observed_rain),
        "mean_amount_gap_mm": float(amount_gap.mean()) if len(merged) else None,
        "mean_probability_gap": float(probability_gap.mean()) if len(merged) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "terminal_id", "satellite_id", "pass_end", "reported_rainfall_mm",
        "rain_probability", "observed_rainfall_mm", "observed_available",
        "image_available", "position_available_ratio", "transfer_mode",
    ]
    with sqlite3.connect(args.history_db) as connection:
        frame = pd.read_sql_query(
            f"SELECT {','.join(columns)} FROM rain_retrieval_passes", connection
        )
    frame["anchor"] = pd.to_datetime(frame["pass_end"], errors="coerce")
    frame = frame.dropna(subset=["anchor"])
    for column in (
        "reported_rainfall_mm", "rain_probability", "observed_rainfall_mm",
        "image_available", "position_available_ratio",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("anchor").drop_duplicates(
        ["terminal_id", "anchor"], keep="last"
    )

    summary: dict = {"history_db": str(args.history_db), "terminals": {}, "pairs": {}}
    terminal_frames: dict[str, pd.DataFrame] = {}
    for terminal in TERMINALS:
        item = frame.loc[
            (frame["terminal_id"] == terminal) & (frame["observed_available"] == 1)
        ].copy()
        terminal_frames[terminal] = item
        rainy = item.loc[item["observed_rainfall_mm"] > 0]
        summary["terminals"][terminal] = {
            "all": _regression(item),
            "rainy": _regression(rainy),
            "classification": _classification(
                item["observed_rainfall_mm"], item["rain_probability"]
            ),
            "image_available": int(item["image_available"].fillna(0).sum()),
            "image_coverage": float(item["image_available"].fillna(0).mean())
            if len(item) else 0.0,
            "first_anchor": item["anchor"].min().isoformat() if len(item) else None,
            "last_anchor": item["anchor"].max().isoformat() if len(item) else None,
        }
    for left, right in combinations(TERMINALS, 2):
        summary["pairs"][f"{left[-3:]}_{right[-3:]}"] = _pair_metrics(
            terminal_frames[left], terminal_frames[right]
        )

    triple = terminal_frames[TERMINALS[0]][
        ["anchor", "reported_rainfall_mm", "observed_rainfall_mm"]
    ].rename(columns={"reported_rainfall_mm": "prediction_001"})
    for terminal in TERMINALS[1:]:
        triple = triple.merge(
            terminal_frames[terminal][["anchor", "reported_rainfall_mm"]].rename(
                columns={"reported_rainfall_mm": f"prediction_{terminal[-3:]}"}
            ),
            on="anchor",
        )
    rainy_triple = triple.loc[triple["observed_rainfall_mm"] > 0].copy()
    prediction_columns = [f"prediction_{terminal[-3:]}" for terminal in TERMINALS]
    for label, item in (("all", triple), ("rainy", rainy_triple)):
        ranges = item[prediction_columns].max(axis=1) - item[prediction_columns].min(axis=1)
        summary.setdefault("triple_overlap", {})[label] = {
            "n": int(len(item)),
            "mean_range_mm": float(ranges.mean()) if len(item) else None,
            "median_range_mm": float(ranges.median()) if len(item) else None,
        }

    (args.output_dir / "consistency_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for terminal, metrics in summary["terminals"].items():
        rows.append({"terminal": terminal, **metrics["all"], "scope": "all"})
        rows.append({"terminal": terminal, **metrics["rainy"], "scope": "rainy"})
    pd.DataFrame(rows).to_csv(args.output_dir / "terminal_accuracy.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
