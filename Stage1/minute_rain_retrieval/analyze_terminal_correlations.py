#!/usr/bin/env python3
"""Compare legacy and new-terminal link measurements on shared minute/satellite keys."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LEGACY_COLUMNS = ["phyRssi", "rssi", "snr", "lastCniValue"]
NEW_COLUMNS = ["carrRssi", "chanRssi", "snr"]


def read_legacy(path: Path) -> pd.DataFrame:
    query = """
        SELECT localTime, satelliteId, phyRssi, rssi, snr, lastCniValue
        FROM phy_data
        WHERE satelliteId IS NOT NULL AND satelliteId != 4294967295
          AND phyRssi IS NOT NULL AND rssi IS NOT NULL
          AND snr IS NOT NULL AND snr != 255 AND lastCniValue IS NOT NULL
        ORDER BY localTime
    """
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(query, connection)
    frame["localTime"] = pd.to_datetime(frame["localTime"], errors="coerce")
    return frame.dropna(subset=["localTime"])


def read_new(path: Path, terminal_id: str) -> pd.DataFrame:
    query = """
        SELECT b.localTime, b.trackNo, b.phaseNo, b.snr,
               r.carrRssi, r.chanRssi
        FROM phy_bb_data AS b
        JOIN phy_rssi_data AS r
          ON r.terminalId = b.terminalId AND r.localTime = b.localTime
        WHERE b.terminalId = ?
          AND b.validMeasBb = 1 AND r.validMeasRssi = 1
          AND b.trackNo IS NOT NULL AND b.phaseNo IS NOT NULL
          AND b.snr IS NOT NULL AND b.snr != 0
          AND r.carrRssi IS NOT NULL AND r.carrRssi != 0
          AND r.chanRssi IS NOT NULL AND r.chanRssi != 0
        ORDER BY b.localTime
    """
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(query, connection, params=[terminal_id])
    frame["localTime"] = pd.to_datetime(frame["localTime"], errors="coerce")
    frame = frame.dropna(subset=["localTime"])
    frame["satelliteId"] = (
        frame["trackNo"].astype("int64") * 256
        + frame["phaseNo"].astype("int64")
    )
    return frame


def mapped_new(
    frame: pd.DataFrame, adapter_path: Path, reference_checkpoint: Path
) -> pd.DataFrame:
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        reference_checkpoint, map_location="cpu", weights_only=False
    )
    target_mean = checkpoint["transforms"]["feature_mean"].numpy()[:4]
    target_scale = checkpoint["transforms"]["feature_std"].numpy()[:4]
    source = np.column_stack([
        pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        for column in adapter["source_columns"]
    ])
    z_score = np.clip(
        (source - np.asarray(adapter["source_mean"]))
        / np.asarray(adapter["source_scale"]),
        -float(adapter.get("clip_z", 5.0)),
        float(adapter.get("clip_z", 5.0)),
    )
    mapped = target_mean.reshape(1, -1) + z_score * target_scale.reshape(1, -1)
    output = frame.copy()
    for index, column in enumerate(LEGACY_COLUMNS):
        output[f"mapped_{column}"] = mapped[:, index]
    return output


def aggregate(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    item = frame[["localTime", "satelliteId", *columns]].copy()
    item["minute"] = item["localTime"].dt.floor("min")
    grouped = item.groupby(["minute", "satelliteId"], as_index=False)[columns].mean()
    return grouped.rename(columns={column: f"{prefix}_{column}" for column in columns})


def paired_correlations(
    left: pd.DataFrame, right: pd.DataFrame, left_prefix: str, right_prefix: str
) -> tuple[pd.DataFrame, dict]:
    paired = left.merge(right, on=["minute", "satelliteId"], how="inner")
    value_columns = [
        column for column in paired.columns
        if column.startswith(f"{left_prefix}_") or column.startswith(f"{right_prefix}_")
    ]
    pearson = paired[value_columns].corr(method="pearson")
    spearman = paired[value_columns].corr(method="spearman")
    rows = []
    for left_column in [c for c in value_columns if c.startswith(f"{left_prefix}_")]:
        for right_column in [c for c in value_columns if c.startswith(f"{right_prefix}_")]:
            rows.append({
                "left": left_column,
                "right": right_column,
                "pearson": float(pearson.loc[left_column, right_column]),
                "spearman": float(spearman.loc[left_column, right_column]),
            })
    summary = {
        "paired_minute_satellite_rows": int(len(paired)),
        "paired_minutes": int(paired["minute"].nunique()),
        "paired_satellites": int(paired["satelliteId"].nunique()),
        "start": paired["minute"].min().isoformat() if len(paired) else None,
        "end": paired["minute"].max().isoformat() if len(paired) else None,
    }
    return pd.DataFrame(rows), summary


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| left | right | Pearson | Spearman |",
        "|---|---|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.left} | {row.right} | {row.pearson:.4f} | {row.spearman:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-db", type=Path, required=True)
    parser.add_argument("--terminal-002-db", type=Path, required=True)
    parser.add_argument("--terminal-003-db", type=Path, required=True)
    parser.add_argument("--adapter-002", type=Path, required=True)
    parser.add_argument("--adapter-003", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    legacy = read_legacy(args.legacy_db)
    terminal_002 = mapped_new(
        read_new(args.terminal_002_db, "01-31-0005-0002"),
        args.adapter_002, args.reference_checkpoint,
    )
    terminal_003 = mapped_new(
        read_new(args.terminal_003_db, "01-31-0005-0003"),
        args.adapter_003, args.reference_checkpoint,
    )
    legacy_agg = aggregate(legacy, LEGACY_COLUMNS, "001")
    mapped_columns = [f"mapped_{column}" for column in LEGACY_COLUMNS]
    terminal_002_mapped = aggregate(terminal_002, mapped_columns, "002")
    terminal_003_mapped = aggregate(terminal_003, mapped_columns, "003")
    terminal_002_native = aggregate(terminal_002, NEW_COLUMNS, "002native")
    terminal_003_native = aggregate(terminal_003, NEW_COLUMNS, "003native")

    pairs = {
        "001_002_mapped": (legacy_agg, terminal_002_mapped, "001", "002"),
        "001_003_mapped": (legacy_agg, terminal_003_mapped, "001", "003"),
        "002_003_native": (
            terminal_002_native, terminal_003_native, "002native", "003native"
        ),
    }
    summary = {
        "source_rows": {
            "001": len(legacy), "002": len(terminal_002), "003": len(terminal_003)
        },
        "method": "mean by identical minute and protocol satellite ID, then inner join",
        "shared_modalities": [
            "position from terminal 001", "temperature", "humidity", "pressure",
            "sky-image probabilities", "rain-gauge target",
        ],
        "pairs": {},
    }
    report_lines = [
        "# 三终端链路相关性分析", "",
        "配对键为相同分钟和相同协议卫星 ID；位置、地面气象、图像和雨量标签为共享数据源。",
        "",
    ]
    for name, arguments in pairs.items():
        correlations, pair_summary = paired_correlations(*arguments)
        correlations.to_csv(args.output_dir / f"{name}_correlations.csv", index=False)
        summary["pairs"][name] = pair_summary
        strongest = correlations.reindex(
            correlations.pearson.abs().sort_values(ascending=False).index
        ).head(8)
        report_lines.extend([
            f"## {name}", "",
            f"配对分钟-卫星样本：{pair_summary['paired_minute_satellite_rows']}；"
            f"分钟数：{pair_summary['paired_minutes']}；卫星数：{pair_summary['paired_satellites']}。",
            "", markdown_table(strongest), "",
        ])
    (args.output_dir / "correlation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "correlation_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
