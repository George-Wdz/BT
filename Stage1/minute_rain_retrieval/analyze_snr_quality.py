#!/usr/bin/env python3
"""Audit legacy-terminal SNR quality and minute-window retention."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


INVALID_SATELLITE_ID = 4294967295
THRESHOLDS = (-15.0, -12.0, -10.0, -8.5, -5.0, 0.0)


def load_data(db_path: Path, terminal_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        phy = pd.read_sql_query(
            """
            SELECT localTime, satelliteId, phyRssi, rssi, snr, lastCniValue
            FROM phy_data
            WHERE terminalId = ? AND satelliteId != ?
              AND snr IS NOT NULL AND snr != 255
            ORDER BY localTime
            """,
            connection,
            params=[terminal_id, INVALID_SATELLITE_ID],
        )
        gauge = pd.read_sql_query(
            """
            SELECT datetime AS timestamp, rainfall
            FROM weather_station
            WHERE terminalId = ? AND rainfall IS NOT NULL AND rainfall >= 0
            ORDER BY datetime
            """,
            connection,
            params=[terminal_id],
        )
    phy["localTime"] = pd.to_datetime(phy["localTime"], errors="coerce")
    gauge["timestamp"] = pd.to_datetime(gauge["timestamp"], errors="coerce")
    return (
        phy.dropna(subset=["localTime", "snr"]).sort_values("localTime"),
        gauge.dropna(subset=["timestamp", "rainfall"]).sort_values("timestamp"),
    )


def fit_intersection(snr: np.ndarray) -> tuple[float, dict]:
    rng = np.random.default_rng(42)
    sample = rng.choice(snr, min(len(snr), 300_000), replace=False).reshape(-1, 1)
    mixture = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(sample)
    order = np.argsort(mixture.means_.ravel())
    means = mixture.means_.ravel()[order]
    stds = np.sqrt(mixture.covariances_.ravel()[order])
    weights = mixture.weights_.ravel()[order]
    a = 1 / (2 * stds[0] ** 2) - 1 / (2 * stds[1] ** 2)
    b = -means[0] / stds[0] ** 2 + means[1] / stds[1] ** 2
    c = (
        means[0] ** 2 / (2 * stds[0] ** 2)
        - means[1] ** 2 / (2 * stds[1] ** 2)
        - np.log((weights[0] / stds[0]) / (weights[1] / stds[1]))
    )
    roots = np.roots([a, b, c])
    candidates = [
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-8 and means[0] < root.real < means[1]
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"unable to identify the two-component intersection: {roots}")
    return candidates[0], {
        "means_db": means.tolist(),
        "std_db": stds.tolist(),
        "weights": weights.tolist(),
    }


def preceding_window_counts(
    link_ns: np.ndarray, mask: np.ndarray, anchor_ns: np.ndarray
) -> np.ndarray:
    cumulative = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    left = np.searchsorted(link_ns, anchor_ns - 60_000_000_000, side="right")
    right = np.searchsorted(link_ns, anchor_ns, side="right")
    return cumulative[right] - cumulative[left]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    parser.add_argument("--recommended-threshold", type=float, default=-10.0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phy, gauge = load_data(args.db_path, args.terminal_id)
    snr = phy["snr"].to_numpy(np.float64)
    intersection, mixture = fit_intersection(snr)

    lower = int(np.floor(snr.min()))
    upper = int(np.ceil(snr.max())) + 1
    counts, edges = np.histogram(snr, bins=np.arange(lower, upper + 1))
    histogram = pd.DataFrame({
        "snr_lower_db": edges[:-1],
        "snr_upper_db": edges[1:],
        "point_count": counts,
        "point_fraction": counts / len(snr),
    })
    histogram.to_csv(output_dir / "snr_histogram_1db.csv", index=False)

    minute = phy.assign(minute=phy["localTime"].dt.floor("min")).groupby("minute").agg(
        raw_point_count=("snr", "size"),
        snr_min_db=("snr", "min"),
        snr_median_db=("snr", "median"),
        snr_max_db=("snr", "max"),
        satellite_count=("satelliteId", "nunique"),
    )
    for threshold in THRESHOLDS:
        column = f"point_count_snr_ge_{threshold:g}"
        values = (
            phy.assign(
                minute=phy["localTime"].dt.floor("min"),
                accepted=phy["snr"].ge(threshold),
            )
            .groupby("minute")["accepted"]
            .sum()
        )
        minute[column] = values
    minute.reset_index().to_csv(output_dir / "minute_snr_quality.csv", index=False)

    link_ns = phy["localTime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    anchor_ns = gauge["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    rainy = gauge["rainfall"].to_numpy(np.float64) * 0.1 > 0.005
    rows = []
    for threshold in (float("-inf"), *THRESHOLDS):
        accepted = np.ones(len(snr), dtype=bool) if np.isneginf(threshold) else snr >= threshold
        window_counts = preceding_window_counts(link_ns, accepted, anchor_ns)
        for minimum_points in (3, 5, 10):
            valid = window_counts >= minimum_points
            rows.append({
                "min_snr_db": None if np.isneginf(threshold) else threshold,
                "min_phy_points": minimum_points,
                "retained_phy_points": int(accepted.sum()),
                "retained_phy_fraction": float(accepted.mean()),
                "valid_gauge_minutes": int(valid.sum()),
                "valid_rainy_gauge_minutes": int((valid & rainy).sum()),
                "rainy_minute_recall_before_position_weather": float(
                    (valid & rainy).sum() / max(rainy.sum(), 1)
                ),
            })
    retention = pd.DataFrame(rows)
    retention.to_csv(output_dir / "snr_threshold_retention.csv", index=False)

    categories = pd.cut(
        snr,
        bins=[-np.inf, -15.0, args.recommended_threshold, 0.0, np.inf],
        labels=[
            "noise_or_unlocked",
            "transition_or_very_weak",
            "weak_but_usable",
            "normal_arc",
        ],
        right=False,
    )
    category_counts = pd.Series(categories).value_counts(sort=False)
    summary = {
        "database": str(args.db_path.resolve()),
        "terminal_id": args.terminal_id,
        "phy_points": len(phy),
        "gauge_minutes": len(gauge),
        "rainy_gauge_minutes": int(rainy.sum()),
        "gmm": {**mixture, "intersection_db": intersection},
        "recommended_min_snr_db": args.recommended_threshold,
        "minute_point_count_snr_median_correlation": float(
            minute["raw_point_count"].corr(minute["snr_median_db"])
        ),
        "quality_categories": {
            str(name): {
                "points": int(value),
                "fraction": float(value / len(phy)),
            }
            for name, value in category_counts.items()
        },
    }
    (output_dir / "snr_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# SNR 数据质量审计

- 终端：`{args.terminal_id}`
- PHY 点数：{len(phy):,}
- 双高斯中心：{mixture['means_db'][0]:.2f} dB、{mixture['means_db'][1]:.2f} dB
- 双高斯交点：{intersection:.2f} dB
- 建议工程阈值：SNR $\\geq$ {args.recommended_threshold:g} dB
- 每分钟原始点数与 SNR 中位数相关系数：{summary['minute_point_count_snr_median_correlation']:.3f}

建议阈值略低于统计交点，以保留弧段捕获和退出阶段的少量弱信号点；
`min_phy_points` 必须在 SNR 过滤后计算。低于 -15 dB 的密集簇视为噪声底或
未锁定状态，[-15, {args.recommended_threshold:g}) dB 保留为不参与反演的过渡区。
"""
    (output_dir / "snr_quality_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
