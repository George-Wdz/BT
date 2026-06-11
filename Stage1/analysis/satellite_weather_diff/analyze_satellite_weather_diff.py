#!/usr/bin/env python3
"""Compare same-satellite link metrics under rainy vs dry passes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LINK_COLS = ["phyRssi", "rssi", "snr", "lastCniValue"]
POSITION_COLS = [
    "sat_longitude",
    "sat_latitude",
    "sat_altitude",
    "terminal_longitude",
    "terminal_latitude",
    "terminal_altitude",
]
STAT_NAMES = ["mean", "min", "max", "std", "range"]


def _feature_stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "std": float(np.nanstd(arr)),
        "range": float(np.nanmax(arr) - np.nanmin(arr)),
    }


def build_pass_feature_frame(npz_path: Path, rain_threshold: float) -> pd.DataFrame:
    npz = np.load(npz_path, allow_pickle=True)
    passes = list(npz["passes"])
    rows = []
    for pass_id, p in enumerate(passes):
        link = np.asarray(p["link_features"], dtype=np.float64)
        position = np.asarray(p["position_features"], dtype=np.float64)
        rain = float(p["labels"][0])
        timestamps = pd.DatetimeIndex(p["timestamps"])
        row = {
            "pass_id": pass_id,
            "satellite_id": int(p["satellite_id"]),
            "pass_start": timestamps[0],
            "pass_end": timestamps[-1],
            "points": int(len(timestamps)),
            "pass_rainfall_mm": rain,
            "is_rainy": bool(rain > rain_threshold),
            "image_available": int(p.get("label_meta", {}).get("image_available", 0) or 0),
        }
        for i, name in enumerate(LINK_COLS):
            stats = _feature_stats(link[:, i])
            for stat_name, value in stats.items():
                row[f"{name}_{stat_name}"] = value
        for i, name in enumerate(POSITION_COLS):
            stats = _feature_stats(position[:, i])
            for stat_name, value in stats.items():
                row[f"{name}_{stat_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _format_pass_times(g: pd.DataFrame, limit: int = 5) -> str:
    parts = []
    for _, row in g.sort_values("pass_start").head(limit).iterrows():
        start = pd.Timestamp(row["pass_start"]).strftime("%Y-%m-%d %H:%M")
        end = pd.Timestamp(row["pass_end"]).strftime("%H:%M")
        parts.append(f"{start}-{end}, rain={row['pass_rainfall_mm']:.3g}")
    if len(g) > limit:
        parts.append(f"...共{len(g)}次")
    return "; ".join(parts)


def summarize_by_satellite(pass_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [f"{name}_{stat}" for name in LINK_COLS for stat in STAT_NAMES]
    rows = []
    for sat_id, g in pass_df.groupby("satellite_id"):
        rainy = g[g["is_rainy"]]
        dry = g[~g["is_rainy"]]
        base = {
            "satellite_id": int(sat_id),
            "total_passes": int(len(g)),
            "rainy_passes": int(len(rainy)),
            "dry_passes": int(len(dry)),
            "first_pass_start": pd.Timestamp(g["pass_start"].min()),
            "last_pass_end": pd.Timestamp(g["pass_end"].max()),
            "rainy_pass_time_examples": _format_pass_times(rainy) if len(rainy) else "",
            "dry_pass_time_examples": _format_pass_times(dry, limit=3) if len(dry) else "",
            "max_rainfall_mm": float(g["pass_rainfall_mm"].max()),
            "mean_rainfall_mm": float(g["pass_rainfall_mm"].mean()),
            "image_available_passes": int(g["image_available"].sum()),
        }
        if len(rainy) == 0 or len(dry) == 0:
            rows.append(base)
            continue
        for col in feature_cols:
            rainy_mean = float(rainy[col].mean())
            dry_mean = float(dry[col].mean())
            dry_std = float(dry[col].std(ddof=0))
            delta = rainy_mean - dry_mean
            base[f"{col}_rainy_mean"] = rainy_mean
            base[f"{col}_dry_mean"] = dry_mean
            base[f"{col}_delta"] = delta
            base[f"{col}_abs_delta"] = abs(delta)
            base[f"{col}_z_delta"] = delta / (dry_std + 1e-6)
        rows.append(base)
    return pd.DataFrame(rows)


def summarize_features(sat_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in LINK_COLS:
        for stat in STAT_NAMES:
            prefix = f"{name}_{stat}"
            delta_col = f"{prefix}_delta"
            abs_delta_col = f"{prefix}_abs_delta"
            z_col = f"{prefix}_z_delta"
            if delta_col not in sat_df.columns:
                continue
            valid = sat_df.dropna(subset=[delta_col])
            if valid.empty:
                continue
            rows.append({
                "feature": prefix,
                "satellites_with_rain_and_dry": int(len(valid)),
                "mean_delta": float(valid[delta_col].mean()),
                "median_delta": float(valid[delta_col].median()),
                "mean_abs_delta": float(valid[abs_delta_col].mean()),
                "median_abs_delta": float(valid[abs_delta_col].median()),
                "mean_abs_z_delta": float(valid[z_col].abs().mean()),
                "median_abs_z_delta": float(valid[z_col].abs().median()),
            })
    return pd.DataFrame(rows).sort_values(
        ["mean_abs_z_delta", "mean_abs_delta"],
        ascending=False,
    )


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    npz_path: Path,
    rain_threshold: float,
    pass_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    comparable = sat_df[(sat_df["rainy_passes"] > 0) & (sat_df["dry_passes"] > 0)]
    enough = comparable[(comparable["rainy_passes"] >= 2) & (comparable["dry_passes"] >= 5)]
    top_total = sat_df.sort_values(["total_passes", "rainy_passes"], ascending=False).head(15)
    top_rainy = sat_df.sort_values(["rainy_passes", "total_passes"], ascending=False).head(20)

    lines = [
        "# 同卫星雨/非雨链路差异分析",
        "",
        f"数据集：`{npz_path}`",
        f"雨天阈值：`pass_rainfall_mm > {rain_threshold}` mm",
        "",
        "## 总览",
        "",
        f"- 总过境数：{len(pass_df)}",
        f"- 雨天过境数：{int(pass_df['is_rainy'].sum())}",
        f"- 非雨过境数：{int((~pass_df['is_rainy']).sum())}",
        f"- 卫星数量：{int(pass_df['satellite_id'].nunique())}",
        f"- 同时包含雨天和非雨过境的卫星数量：{len(comparable)}",
        f"- 至少 2 次雨天且至少 5 次非雨过境的卫星数量：{len(enough)}",
        "",
        "## 链路指标敏感性排序",
        "",
        "排序依据：对同时有雨/非雨样本的卫星，计算同卫星雨天均值与非雨均值的差异，按平均绝对 z-delta 排序。"
        "z-delta 可理解为“相对该卫星非雨波动幅度的偏移量”。",
        "",
        _markdown_table(feature_df.head(12)),
        "",
        "## 过境次数最多的卫星",
        "",
        _markdown_table(top_total[[
            "satellite_id", "total_passes", "rainy_passes", "dry_passes",
            "first_pass_start", "last_pass_end", "max_rainfall_mm", "image_available_passes",
        ]]),
        "",
        "## 雨天过境次数最多的卫星",
        "",
        _markdown_table(top_rainy[[
            "satellite_id", "total_passes", "rainy_passes", "dry_passes",
            "first_pass_start", "last_pass_end", "max_rainfall_mm",
            "rainy_pass_time_examples", "image_available_passes",
        ]]),
        "",
        "## 结论",
        "",
        "当前数据还不适合直接按单颗卫星单独训练：单颗卫星过境次数有限，且多数卫星的雨天样本只有 1-2 次。"
        "更稳妥的用法是把同卫星非雨状态作为 clear-sky baseline，构造链路差分特征，再交给统一模型学习。",
        "",
        "轨道几何参数目前无法直接获取，但 `position_features` 中包含卫星经纬度/高度和终端经纬度/高度。"
        "因此后续 baseline 匹配不只用同一颗卫星，还应同时考虑过境中心时间和 position 相似度，减少把完全不同过境条件混在一起的风险。",
        "",
        "本目录输出文件说明：`pass_link_stats.csv` 是 pass 级链路和 position 统计；"
        "`satellite_rain_dry_deltas.csv` 是卫星级雨/非雨差异；"
        "`feature_delta_ranking.csv` 是链路统计量敏感性排序。",
        "",
    ]
    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="/home/wdz/BT/Stage1/model/data/pass_dataset_link4_img_cls.npz",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/wdz/BT/Stage1/analysis/satellite_weather_diff",
    )
    parser.add_argument("--rain-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pass_df = build_pass_feature_frame(npz_path, args.rain_threshold)
    sat_df = summarize_by_satellite(pass_df)
    feature_df = summarize_features(sat_df)

    pass_df.to_csv(out_dir / "pass_link_stats.csv", index=False)
    sat_df.to_csv(out_dir / "satellite_rain_dry_deltas.csv", index=False)
    feature_df.to_csv(out_dir / "feature_delta_ranking.csv", index=False)
    summary = {
        "source_dataset": str(npz_path),
        "rain_threshold": args.rain_threshold,
        "total_passes": int(len(pass_df)),
        "rainy_passes": int(pass_df["is_rainy"].sum()),
        "dry_passes": int((~pass_df["is_rainy"]).sum()),
        "unique_satellites": int(pass_df["satellite_id"].nunique()),
        "satellites_with_both_rainy_and_dry": int(((sat_df["rainy_passes"] > 0) & (sat_df["dry_passes"] > 0)).sum()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir, npz_path, args.rain_threshold, pass_df, sat_df, feature_df)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()
