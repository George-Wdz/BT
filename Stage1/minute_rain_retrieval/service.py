#!/usr/bin/env python3
"""Three-terminal dashboard backed by a balanced minute-rain model."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import sqlite3
import sys
from datetime import date, datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import uvicorn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
STAGE1_ROOT = ROOT.parent
DEMO_ROOT = STAGE1_ROOT / "rainfall_dashboard"
sys.path.insert(0, str(DEMO_ROOT))

import app as dashboard_app  # noqa: E402

sys.path.insert(0, str(ROOT))
from data_flow import (  # noqa: E402
    GEO_COLUMNS,
    IMAGE_COLUMNS,
    LINK_COLUMNS,
    WEATHER_COLUMNS,
    _clean_weather,
)
from dataset import MinuteRainDataset, TrainTransforms, collate_minutes  # noqa: E402
from model import MinuteRainTransformer  # noqa: E402
from terminal_io import (  # noqa: E402
    LegacyTerminalReader,
    NewTerminalReader,
    RuntimeConfig,
    protocol_satellite_id,
)

sys.path.insert(0, str(STAGE1_ROOT))
from minute_rain_retrieval.vision_label_worker import (  # noqa: E402
    IncrementalVisionLabeler,
)


TERMINAL_001 = "01-31-0005-0001"


def _configure_minute_dashboard_text(sampling_ratio_label: str) -> None:
    """Keep the 8040 layout while making minute-service labels unambiguous."""
    replacements = {
        "<title>三终端降雨反演</title>": "<title>三终端分钟降雨反演</title>",
        "<h1>三终端降雨反演</h1><p>Stage1 小模型 · 过境级对比</p>": (
            f"<h1>三终端分钟降雨反演</h1><p>{sampling_ratio_label} 模型 · 分钟级对比</p>"
        ),
        "<h2>重叠过境反演一致性</h2><p>每列为时间重叠的一组过境，比较各终端过境平均反演雨强；竖线表示终端间极差。</p>": (
            "<h2>同分钟反演一致性</h2><p>每列对应相同雨量计锚点的一分钟窗口；竖线表示终端反演雨强极差。</p>"
        ),
        "过境累计雨量保留在悬浮详情中。不同终端过境起止时间不完全一致，因此以平均雨强作为主要一致性指标。": (
            "分钟累计雨量保留在悬浮详情中；一致性仅比较至少两台终端均具有有效PHY输入的同一分钟。"
        ),
        "模型值与雨量计值均对应单次卫星过境。雨量计值按过境起止时刻的累计雨量差计算；“—”表示缺少有效边界。不同卫星可能观测同一天气过程，因此不跨过境累加。002/003 为冻结 001 模型的零样本迁移结果。": (
            "模型值与雨量计值均对应一个前一分钟窗口；模型累计量为有效分钟预测值之和。“—”表示该分钟有效PHY点不足。002/003 为冻结 001 分钟模型的直接迁移结果。"
        ),
        "重叠过境组": "同分钟配对组",
        "所选日期没有可配对的重叠过境": "所选日期没有至少两台终端共同有效的分钟",
        "过境累计反演": "分钟累计反演",
        "过境平均雨强": "分钟等效雨强",
        "重叠过境开始时间": "分钟窗口结束时间",
        "过境平均雨强 / mm/h": "分钟等效雨强 / mm/h",
        "有效过境": "有效分钟",
        "过境 MAE / mm": "分钟 MAE / mm",
        "当日无过境": "当日无有效分钟",
        "当日末次过境": "当日末次有效分钟",
        "所选日期没有满足最少采样点要求的链路过境。": (
            "所选日期没有满足最少采样点要求的分钟窗口。"
        ),
        "Stage1 三终端小模型 · ${d.model_version}": (
            f"三终端 {sampling_ratio_label} 分钟模型 · " + "${d.model_version}"
        ),
    }
    html = dashboard_app.INDEX_HTML
    for source, target in replacements.items():
        html = html.replace(source, target)
    dashboard_app.INDEX_HTML = html


def _load_checkpoint(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    args = checkpoint["args"]
    state = checkpoint["transforms"]
    satellite_to_index = {
        int(satellite): int(index)
        for satellite, index in zip(
            state["satellite_ids"].cpu().tolist(),
            state["satellite_indices"].cpu().tolist(),
        )
    }
    dry_by_satellite = {
        int(satellite): value.cpu().numpy().astype(np.float32)
        for satellite, value in zip(
            state["dry_satellite_ids"].cpu().tolist(), state["dry_values"]
        )
    }
    transforms = TrainTransforms(
        feature_mean=state["feature_mean"].cpu().numpy().astype(np.float32),
        feature_std=state["feature_std"].cpu().numpy().astype(np.float32),
        satellite_to_index=satellite_to_index,
        dry_by_satellite=dry_by_satellite,
        global_dry=state["global_dry"].cpu().numpy().astype(np.float32),
    )
    model = MinuteRainTransformer(
        input_dim=transforms.input_dim,
        num_satellites=max(satellite_to_index.values(), default=0),
        d_model=int(args["d_model"]),
        num_heads=int(args["num_heads"]),
        num_layers=int(args["num_layers"]),
        d_ff=int(args["d_ff"]),
        max_points=int(args["max_points"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, transforms, args


class MinuteThreeTerminalRunner(dashboard_app.ThreeTerminalRunner):
    """Reuse the 8040 dashboard while replacing pass inference with minute inference."""

    def __init__(
        self,
        *,
        config_002: Path,
        config_003: Path,
        checkpoint_path: Path,
        fallback_checkpoint_path: Path | None,
        transfer_checkpoint_path: Path | None,
        device_name: str,
        history_db_path: Path,
        poll_interval_s: float,
        worker_lookback_hours: float,
        worker_max_samples: int,
        link_analysis_dir: Path,
        min_phy_points: int,
        fallback_min_phy_points: int,
        position_tolerance_s: float,
        weather_tolerance_s: float,
        image_tolerance_s: float,
        probability_threshold: float | None,
        backup_db_001: Path | None,
        backup_db_002: Path | None,
        backup_db_003: Path | None,
        camera_input_dir: Path | None,
        vision_weights: Path | None,
        vision_full_csv: Path | None,
        vision_slim_csv: Path | None,
        vision_refresh_interval_s: float,
        vision_max_images_per_refresh: int,
        vision_batch_size: int,
        vision_num_workers: int,
        backup_only: bool = False,
    ) -> None:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        self.minute_model, self.minute_transforms, self.minute_args = _load_checkpoint(
            checkpoint_path, device
        )
        if fallback_checkpoint_path is not None:
            self.fallback_model, self.fallback_transforms, self.fallback_args = (
                _load_checkpoint(fallback_checkpoint_path, device)
            )
        else:
            self.fallback_model = None
            self.fallback_transforms = None
            self.fallback_args = None
        if transfer_checkpoint_path is not None:
            self.transfer_model, self.transfer_transforms, self.transfer_args = (
                _load_checkpoint(transfer_checkpoint_path, device)
            )
        else:
            self.transfer_model = None
            self.transfer_transforms = None
            self.transfer_args = None
        runtime_002 = RuntimeConfig.load(config_002)
        runtime_003 = RuntimeConfig.load(config_003)
        runtime_001 = replace(
            runtime_002,
            terminal_id=TERMINAL_001,
            adapter_path=None,
        )
        target_transforms = self.transfer_transforms or self.minute_transforms
        runtimes = {
            runtime_001.terminal_id: runtime_001,
            runtime_002.terminal_id: runtime_002,
            runtime_003.terminal_id: runtime_003,
        }
        builders = {
            runtime_001.terminal_id: LegacyTerminalReader(runtime_001, vision_slim_csv),
            runtime_002.terminal_id: NewTerminalReader(
                runtime_002,
                target_transforms.feature_mean[:4],
                target_transforms.feature_std[:4],
                vision_slim_csv,
            ),
            runtime_003.terminal_id: NewTerminalReader(
                runtime_003,
                target_transforms.feature_mean[:4],
                target_transforms.feature_std[:4],
                vision_slim_csv,
            ),
        }
        super().__init__(
            config_002,
            config_003,
            device_name,
            history_db_path,
            poll_interval_s,
            worker_lookback_hours,
            worker_max_samples,
            link_analysis_dir,
            runtimes=runtimes,
            builders=builders,
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.minute_checkpoint_path = checkpoint_path
        checkpoint_tag = f"{checkpoint_path.parent.parent.name}/{checkpoint_path.parent.name}"
        fallback_version = (
            f"+fallback/{fallback_checkpoint_path.parent.parent.name}/"
            f"{fallback_checkpoint_path.parent.name}"
            if fallback_checkpoint_path is not None else ""
        )
        transfer_version = (
            f"+transfer/{transfer_checkpoint_path.parent.parent.name}/"
            f"{transfer_checkpoint_path.parent.name}"
            if transfer_checkpoint_path is not None else ""
        )
        dry_ratio = self.minute_args.get("max_train_dry_ratio")
        sampling_version = (
            f"{float(dry_ratio):g}to1" if dry_ratio is not None else "native"
        )
        self.sampling_ratio_label = (
            f"{float(dry_ratio):g}:1" if dry_ratio is not None else "原始分布"
        )
        self.model_version = (
            f"minute-{sampling_version}/{checkpoint_tag}/{checkpoint_path.name}"
            f"{fallback_version}{transfer_version}"
        )
        cache_tag = checkpoint_tag.replace("/", "__") + "__dashboard_v2"
        self.timeline_cache_dir = (
            history_db_path.parent / "timeline_cache" / cache_tag
        )
        self.timeline_cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_points = int(self.minute_args["max_points"])
        self.min_phy_points = max(int(min_phy_points), 1)
        self.fallback_min_phy_points = max(int(fallback_min_phy_points), 1)
        self.position_tolerance = pd.Timedelta(seconds=float(position_tolerance_s))
        self.weather_tolerance = pd.Timedelta(seconds=float(weather_tolerance_s))
        self.image_tolerance = pd.Timedelta(seconds=float(image_tolerance_s))
        self.minute_probability_threshold = float(
            self.minute_args.get("probability_threshold", 0.5)
            if probability_threshold is None
            else probability_threshold
        )
        self.backup_db_paths = {
            TERMINAL_001: backup_db_001,
            "01-31-0005-0002": backup_db_002,
            "01-31-0005-0003": backup_db_003,
        }
        self.backup_only = bool(backup_only)
        self._backup_link_bounds = {
            terminal_id: self._database_link_bounds(terminal_id, path)
            for terminal_id, path in self.backup_db_paths.items()
            if path is not None and path.exists()
        }
        self._gauge_cache: tuple[tuple[int, int], pd.DataFrame] | None = None
        self._weather_cache: tuple[tuple[int, int], pd.DataFrame] | None = None
        self._position_cache: tuple[tuple[int, int], pd.DataFrame] | None = None
        self._timeline_result_cache: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.vision_labeler = None
        if camera_input_dir is not None and vision_weights is not None:
            if vision_full_csv is None or vision_slim_csv is None:
                raise ValueError(
                    "vision full/slim CSV paths are required when online labeling is enabled"
                )
            self.vision_labeler = IncrementalVisionLabeler(
                input_dir=camera_input_dir,
                weights_path=vision_weights,
                full_csv_path=vision_full_csv,
                slim_csv_path=vision_slim_csv,
                device=self.device,
                batch_size=vision_batch_size,
                num_workers=vision_num_workers,
                max_images_per_refresh=vision_max_images_per_refresh,
                refresh_interval_s=vision_refresh_interval_s,
            )

    @staticmethod
    def _time_cache_key(start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
        return int(start.value), int(end.value)

    @staticmethod
    def _database_link_bounds(
        terminal_id: str, path: Path
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        table = "phy_data" if terminal_id == TERMINAL_001 else "phy_bb_data"
        where = "" if terminal_id == TERMINAL_001 else " WHERE terminalId = ?"
        params = [] if terminal_id == TERMINAL_001 else [terminal_id]
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                f"SELECT MIN(localTime), MAX(localTime) FROM {table}{where}", params
            ).fetchone()
        if not row or not row[0] or not row[1]:
            return None
        return pd.Timestamp(row[0]), pd.Timestamp(row[1])

    @staticmethod
    def _legacy_backup_link(
        path: Path, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        query = """
            SELECT localTime, satelliteId, earthStationId, phyModcod,
                   phyRssi, rssi, snr, lastCniValue, freqOffset, td, ncr
            FROM phy_data
            WHERE localTime >= ? AND localTime <= ?
              AND satelliteId IS NOT NULL AND satelliteId != 4294967295
              AND phyRssi IS NOT NULL AND rssi IS NOT NULL
              AND snr IS NOT NULL AND snr != 255
              AND lastCniValue IS NOT NULL
              AND freqOffset IS NOT NULL AND freqOffset != 0
              AND td IS NOT NULL AND td != 0
            ORDER BY localTime
        """
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            frame = pd.read_sql_query(
                query, connection, params=[start.isoformat(), end.isoformat()]
            )
        if frame.empty:
            return frame
        frame["localTime"] = pd.to_datetime(frame["localTime"], errors="coerce")
        frame = frame.dropna(subset=["localTime"]).sort_values("localTime")
        duplicate_columns = [
            "satelliteId", "earthStationId", "phyModcod", "rssi", "phyRssi",
            "lastCniValue", "snr", "freqOffset", "td", "ncr",
        ]
        near_duplicate = (
            frame.groupby(duplicate_columns, dropna=False)["localTime"]
            .diff().dt.total_seconds().le(1.0)
        )
        frame = frame.loc[~near_duplicate].copy()
        return frame.set_index("localTime")[
            ["satelliteId", "earthStationId", *LINK_COLUMNS]
        ].sort_index()

    def _new_terminal_backup_link(
        self, terminal_id: str, path: Path,
        start: pd.Timestamp, end: pd.Timestamp,
    ) -> pd.DataFrame:
        query = """
            SELECT b.localTime, b.trackNo, b.phaseNo, b.snr,
                   b.rsrp, b.nipower, b.freqOffset, b.timeOffset,
                   r.gain, r.chanRssi, r.carrRssi
            FROM phy_bb_data AS b
            JOIN phy_rssi_data AS r
              ON r.terminalId = b.terminalId AND r.localTime = b.localTime
            WHERE b.terminalId = ?
              AND b.localTime >= ? AND b.localTime <= ?
              AND b.validMeasBb = 1 AND r.validMeasRssi = 1
              AND b.snr IS NOT NULL AND b.snr != 0
              AND b.trackNo IS NOT NULL AND b.phaseNo IS NOT NULL
              AND r.chanRssi IS NOT NULL AND r.chanRssi != 0
              AND r.carrRssi IS NOT NULL AND r.carrRssi != 0
            ORDER BY b.localTime
        """
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            frame = pd.read_sql_query(
                query, connection,
                params=[terminal_id, start.isoformat(), end.isoformat()],
            )
        if frame.empty:
            return frame
        frame["localTime"] = pd.to_datetime(frame["localTime"], errors="coerce")
        frame = frame.dropna(subset=["localTime"]).sort_values("localTime")
        frame["satelliteId"] = protocol_satellite_id(
            frame["trackNo"], frame["phaseNo"]
        )
        mapped = self.builders[terminal_id].adapter.transform(frame)
        for index, column in enumerate(LINK_COLUMNS):
            frame[column] = mapped[:, index]
        frame["earthStationId"] = 0
        return frame.set_index("localTime")[
            ["satelliteId", "earthStationId", *LINK_COLUMNS]
        ].sort_index()

    def _read_terminal_link(
        self, terminal_id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        backup_path = self.backup_db_paths.get(terminal_id)
        backup_bounds = self._backup_link_bounds.get(terminal_id)
        backup_covers_range = bool(
            backup_bounds is not None
            and backup_bounds[0] <= start
            and end <= backup_bounds[1]
        )
        live_latest = (
            None
            if self.backup_only
            else self.builders[terminal_id].latest_link_time()
        )
        live_overlaps_range = live_latest is None or start <= live_latest
        live = (
            self.builders[terminal_id]._read_link(start, end)
            if not self.backup_only
            and not backup_covers_range
            and live_overlaps_range
            else pd.DataFrame()
        )
        if backup_path is None or not backup_path.exists():
            return live
        if terminal_id == TERMINAL_001:
            backup = self._legacy_backup_link(backup_path, start, end)
        else:
            backup = self._new_terminal_backup_link(
                terminal_id, backup_path, start, end
            )
        parts = []
        for priority, frame in enumerate((live, backup)):
            if frame is None or frame.empty:
                continue
            item = frame.reset_index()
            item["_source_priority"] = priority
            parts.append(item)
        if not parts:
            return live
        combined = pd.concat(parts, ignore_index=True).sort_values(
            ["localTime", "_source_priority"]
        )
        combined = combined.drop_duplicates(
            ["localTime", "satelliteId"], keep="first"
        ).drop(columns="_source_priority")
        return combined.set_index("localTime").sort_index()

    @property
    def db_path(self) -> Path:
        return self.runtimes[TERMINAL_001].db_path

    def _read_gauge(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        cache_key = self._time_cache_key(start, end)
        if self._gauge_cache is not None and self._gauge_cache[0] == cache_key:
            return self._gauge_cache[1]
        query = """
            SELECT datetime AS timestamp, rainfall
            FROM weather_station
            WHERE terminalId = ?
              AND datetime >= ?
              AND datetime <= ?
              AND rainfall IS NOT NULL AND rainfall >= 0
            ORDER BY datetime
        """
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            frame = pd.read_sql_query(
                query,
                conn,
                params=[
                    TERMINAL_001,
                    start.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    end.strftime("%Y-%m-%d %H:%M:%S.%f"),
                ],
            )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["rainfall"] = pd.to_numeric(frame["rainfall"], errors="coerce")
        frame = frame.dropna().sort_values("timestamp")
        self._gauge_cache = (cache_key, frame)
        return frame

    def _read_weather(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        cache_key = self._time_cache_key(start, end)
        if self._weather_cache is not None and self._weather_cache[0] == cache_key:
            return self._weather_cache[1]
        sensor_query = """
            SELECT timestamp, temperature, humidity, pressure
            FROM weather_data
            WHERE terminalId = ?
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp
        """
        station_query = """
            SELECT datetime AS timestamp, temperature, humidity, pressure
            FROM weather_station
            WHERE terminalId = ?
              AND datetime >= ?
              AND datetime <= ?
            ORDER BY datetime
        """
        time_params = [
            TERMINAL_001,
            start.strftime("%Y-%m-%d %H:%M:%S.%f"),
            end.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ]
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            sensor = pd.read_sql_query(
                sensor_query,
                conn,
                params=time_params,
            )
            station = pd.read_sql_query(
                station_query,
                conn,
                params=time_params,
            )
        sensor["source_priority"] = 0
        station["source_priority"] = 1
        sources = [frame for frame in (sensor, station) if not frame.empty]
        if not sources:
            frame = pd.DataFrame(columns=["timestamp", *WEATHER_COLUMNS])
            self._weather_cache = (cache_key, frame)
            return frame
        frame = pd.concat(sources, ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = (
            frame.dropna(subset=["timestamp"])
            .sort_values(["timestamp", "source_priority"])
            .drop_duplicates("timestamp", keep="first")
            .drop(columns="source_priority")
        )
        frame = _clean_weather(frame)
        self._weather_cache = (cache_key, frame)
        return frame

    def _read_position_cached(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        cache_key = self._time_cache_key(start, end)
        if self._position_cache is not None and self._position_cache[0] == cache_key:
            return self._position_cache[1]
        position = self.builders[TERMINAL_001]._read_position(start, end)
        self._position_cache = (cache_key, position)
        return position

    def _align_position(
        self,
        terminal_id: str,
        link: pd.DataFrame,
        query_start: pd.Timestamp | None = None,
        query_end: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, float]:
        if link.empty:
            return link, 0.0
        position = self._read_position_cached(
            query_start if query_start is not None else link.index.min(),
            query_end if query_end is not None else link.index.max(),
        )
        source = link.reset_index().rename(columns={link.index.name or "index": "localTime"})
        parts: list[pd.DataFrame] = []
        total_rows = 0
        matched_rows = 0
        geo_mean = self.minute_transforms.feature_mean[4:8]
        position_groups = (
            {
                int(satellite): group.reset_index().sort_values("localTime")
                for satellite, group in position.groupby("satId")
            }
            if not position.empty
            else {}
        )
        for satellite, group in source.groupby("satelliteId"):
            total_rows += len(group)
            right = position_groups.get(int(satellite))
            if right is None:
                if terminal_id == TERMINAL_001:
                    continue
                merged = group.copy()
                for column, value in zip(GEO_COLUMNS, geo_mean):
                    merged[column] = float(value)
            else:
                merged = pd.merge_asof(
                    group.sort_values("localTime"),
                    right[["localTime", *GEO_COLUMNS]].sort_values("localTime"),
                    on="localTime",
                    direction="nearest",
                    tolerance=self.position_tolerance,
                )
                real_position = merged[GEO_COLUMNS].notna().all(axis=1)
                matched_rows += int(real_position.sum())
                if terminal_id == TERMINAL_001:
                    merged = merged.loc[real_position].copy()
                    if merged.empty:
                        continue
                else:
                    merged.loc[:, GEO_COLUMNS] = merged[GEO_COLUMNS].fillna(
                        dict(zip(GEO_COLUMNS, geo_mean.tolist()))
                    )
            parts.append(merged)
        if not parts:
            return source.iloc[0:0].copy(), 0.0
        aligned = pd.concat(parts, ignore_index=True).sort_values("localTime")
        available_ratio = matched_rows / total_rows if total_rows else 0.0
        return aligned, available_ratio

    def _aligned_points(
        self, terminal_id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, float]:
        link = self._read_terminal_link(
            terminal_id, start - pd.Timedelta(seconds=60), end
        )
        aligned, position_ratio = self._align_position(terminal_id, link)
        if aligned.empty:
            return aligned, position_ratio
        weather = self._read_weather(start - pd.Timedelta(seconds=60), end)
        return self._merge_weather(aligned, weather), position_ratio

    def _merge_weather(
        self, link: pd.DataFrame, weather: pd.DataFrame
    ) -> pd.DataFrame:
        if link.empty or weather.empty:
            return link.iloc[0:0].copy()
        source = link
        if "localTime" not in source.columns:
            if source.index.name in ("localTime", "timestamp"):
                source = source.reset_index().rename(
                    columns={source.index.name or "index": "localTime"}
                )
            else:
                raise KeyError(f"link data has no time field: {list(source.columns)}")
        merged = pd.merge_asof(
            source.sort_values("localTime"),
            weather[["timestamp", *WEATHER_COLUMNS]].sort_values("timestamp"),
            left_on="localTime", right_on="timestamp", direction="nearest",
            tolerance=self.weather_tolerance,
        ).dropna(subset=WEATHER_COLUMNS)
        return merged.sort_values("localTime")

    def _weather_aligned_points(
        self, terminal_id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        link = self._read_terminal_link(
            terminal_id, start - pd.Timedelta(seconds=60), end
        )
        weather = self._read_weather(start - pd.Timedelta(seconds=60), end)
        return self._merge_weather(link, weather)

    def _build_minute_samples(
        self,
        terminal_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        max_samples: int,
    ) -> list[dict[str, Any]]:
        gauge = self._read_gauge(start, end)
        link = self._read_terminal_link(
            terminal_id, start - pd.Timedelta(seconds=60), end
        )
        position_aligned, position_ratio = self._align_position(
            terminal_id,
            link,
            start - pd.Timedelta(seconds=60),
            end,
        )
        weather = self._read_weather(start - pd.Timedelta(seconds=60), end)
        aligned = self._merge_weather(position_aligned, weather)
        fallback_aligned = (
            self._merge_weather(link, weather)
            if self.fallback_model is not None else aligned.iloc[0:0]
        )
        if gauge.empty or (aligned.empty and fallback_aligned.empty):
            return []
        images = self.builders[terminal_id]._images()
        link_ns = aligned["localTime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        fallback_ns = fallback_aligned["localTime"].to_numpy(
            dtype="datetime64[ns]"
        ).astype(np.int64)
        image_ns = (
            None
            if images is None or images.empty
            else images.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        )
        window_ns = int(60e9)
        samples: list[dict[str, Any]] = []
        for gauge_row in gauge.itertuples(index=False):
            anchor = pd.Timestamp(gauge_row.timestamp)
            end_ns = int(anchor.value)
            start_ns = end_ns - window_ns
            left = int(np.searchsorted(link_ns, start_ns, side="right"))
            right = int(np.searchsorted(link_ns, end_ns, side="right"))
            window = aligned.iloc[left:right]
            inference_mode = "full_position"
            if len(window) < self.min_phy_points:
                fallback_left = int(np.searchsorted(fallback_ns, start_ns, side="right"))
                fallback_right = int(np.searchsorted(fallback_ns, end_ns, side="right"))
                window = fallback_aligned.iloc[fallback_left:fallback_right]
                if len(window) < self.fallback_min_phy_points:
                    continue
                inference_mode = "fallback_no_position"
            image_vector = np.zeros(4, dtype=np.float32)
            if image_ns is not None:
                insertion = int(np.searchsorted(image_ns, end_ns))
                candidates = [
                    index for index in (insertion - 1, insertion)
                    if 0 <= index < len(image_ns)
                ]
                if candidates:
                    nearest = min(candidates, key=lambda index: abs(int(image_ns[index]) - end_ns))
                    if abs(int(image_ns[nearest]) - end_ns) <= int(self.image_tolerance.value):
                        image_vector = images.iloc[nearest][IMAGE_COLUMNS].to_numpy(np.float32)
            numeric_columns = LINK_COLUMNS + WEATHER_COLUMNS
            if inference_mode == "full_position":
                numeric_columns = LINK_COLUMNS + GEO_COLUMNS + WEATHER_COLUMNS
            numeric = window[numeric_columns].to_numpy(np.float32)
            image_features = np.repeat(image_vector[None, :], len(window), axis=0)
            relative_time = (
                (window["localTime"].astype("int64").to_numpy() - end_ns) / float(window_ns)
            ).astype(np.float32)[:, None]
            satellite_ids = window["satelliteId"].to_numpy(np.int64)
            values, counts = np.unique(satellite_ids, return_counts=True)
            dominant_satellite = int(values[int(np.argmax(counts))])
            samples.append(
                {
                    "features": np.concatenate(
                        [numeric, image_features, relative_time], axis=1
                    ),
                    "satellite_ids": satellite_ids,
                    "minute_rainfall_mm": np.float32(float(gauge_row.rainfall) * 0.1),
                    "anchor_time_ns": np.int64(end_ns),
                    "window_start_ns": np.int64(start_ns),
                    "point_count": np.int32(len(window)),
                    "satellite_count": np.int16(len(values)),
                    "dominant_satellite_id": dominant_satellite,
                    "image_weather": image_vector.copy(),
                    "position_available_ratio": position_ratio,
                    "inference_mode": inference_mode,
                }
            )
        if max_samples > 0:
            samples = samples[-max_samples:]
        return samples

    @torch.inference_mode()
    def _predict_samples(
        self, terminal_id: str, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not samples:
            return []
        predictions = [0.0] * len(samples)
        probabilities = [0.0] * len(samples)
        for mode in ("full_position", "fallback_no_position"):
            indices = [
                index for index, sample in enumerate(samples)
                if sample.get("inference_mode", "full_position") == mode
            ]
            if not indices:
                continue
            if mode == "full_position":
                if terminal_id != TERMINAL_001 and self.transfer_model is not None:
                    model = self.transfer_model
                    transforms = self.transfer_transforms
                    max_points = int(self.transfer_args["max_points"])
                else:
                    model = self.minute_model
                    transforms = self.minute_transforms
                    max_points = self.max_points
            else:
                if self.fallback_model is None or self.fallback_transforms is None:
                    continue
                model = self.fallback_model
                transforms = self.fallback_transforms
                max_points = int(self.fallback_args["max_points"])
            subset = [samples[index] for index in indices]
            splits = np.full(len(subset), "inference", dtype="<U9")
            dataset = MinuteRainDataset(
                subset, splits, "inference", transforms, max_points
            )
            loader = DataLoader(
                dataset, batch_size=min(128, len(dataset)), shuffle=False,
                collate_fn=collate_minutes,
            )
            offset = 0
            for batch in loader:
                output = model(
                    batch["features"].to(self.device),
                    batch["satellite_ids"].to(self.device),
                    batch["valid_mask"].to(self.device),
                )
                batch_predictions = output["prediction"].cpu().tolist()
                batch_probabilities = torch.sigmoid(output["rain_logit"]).cpu().tolist()
                for local_index, (prediction, probability) in enumerate(
                    zip(batch_predictions, batch_probabilities)
                ):
                    global_index = indices[offset + local_index]
                    predictions[global_index] = prediction
                    probabilities[global_index] = probability
                offset += len(batch_predictions)

        inferred_at = datetime.now().isoformat(timespec="seconds")
        rows = []
        for sample, prediction, probability in zip(samples, predictions, probabilities):
            inference_mode = sample.get("inference_mode", "full_position")
            terminal_mode = (
                "native_001" if terminal_id == TERMINAL_001
                else "frozen_001_zscore_alignment"
            )
            transfer_mode = f"{terminal_mode}:{inference_mode}"
            active_transforms = (
                self.fallback_transforms
                if inference_mode == "fallback_no_position"
                else (
                    self.transfer_transforms
                    if terminal_id != TERMINAL_001 and self.transfer_transforms is not None
                    else self.minute_transforms
                )
            )
            observed = float(sample["minute_rainfall_mm"])
            image_weather = np.asarray(
                sample.get("image_weather", np.zeros(4, dtype=np.float32)),
                dtype=np.float32,
            )
            reported = float(prediction) if probability >= self.minute_probability_threshold else 0.0
            start = pd.to_datetime(int(sample["window_start_ns"]), unit="ns")
            end = pd.to_datetime(int(sample["anchor_time_ns"]), unit="ns")
            rows.append(
                {
                    "terminal_id": terminal_id,
                    "satellite_id": int(sample["dominant_satellite_id"]),
                    "pass_start": start.isoformat(),
                    "pass_end": end.isoformat(),
                    "points": int(sample["point_count"]),
                    "pred_rainfall_mm": round(float(prediction), 6),
                    "reported_rainfall_mm": round(reported, 6),
                    "rain_probability": round(float(probability), 6),
                    "rain_rate_mean": round(reported * 60.0, 6),
                    "prob_sunny": round(float(image_weather[0]), 6),
                    "prob_cloudy": round(float(image_weather[1]), 6),
                    "prob_rain": round(float(image_weather[2]), 6),
                    "image_available": int(float(image_weather[3]) > 0.5),
                    "observed_rainfall_mm": round(observed, 2),
                    "observed_available": 1,
                    "observed_reason": None,
                    "absolute_error_mm": round(abs(reported - observed), 6),
                    "checkpoint_satellite_known": int(
                        all(
                            int(satellite) in active_transforms.satellite_to_index
                            for satellite in np.unique(sample["satellite_ids"])
                        )
                    ),
                    "baseline_source": "checkpoint_train_dry_mean",
                    "position_source": (
                        "omitted_fallback"
                        if inference_mode == "fallback_no_position"
                        else (
                            "same_satellite_nearest"
                            if terminal_id == TERMINAL_001
                            else "shared_position_with_training_mean_fallback"
                        )
                    ),
                    "position_available_ratio": round(
                        float(sample["position_available_ratio"]), 6
                    ),
                    "transfer_mode": transfer_mode,
                    "inferred_at": inferred_at,
                }
            )
        return rows

    def _terminal_result(
        self, terminal_id: str, query_date: date, max_passes: int
    ) -> dict[str, Any]:
        start = pd.Timestamp(datetime.combine(query_date, datetime.min.time()))
        end = start + pd.Timedelta(days=1)
        historical = query_date < date.today()
        # A historical day is materialized only after the complete day has been
        # inferred. max_passes limits the API response, not persisted coverage.
        samples = self._build_minute_samples(
            terminal_id, start, end, 0 if historical else max_passes
        )
        rows = self._predict_samples(terminal_id, samples)
        self.history.upsert_many(rows)
        if historical:
            self.history.mark_day_materialized(query_date, terminal_id, self.model_version)
        visible_rows = rows[-max_passes:] if max_passes > 0 else rows
        return self._result_from_rows(
            terminal_id, query_date, visible_rows, "minute_inference"
        )

    def update_recent_once(self) -> dict[str, Any]:
        persisted = 0
        states = []
        vision_state = None
        if self.vision_labeler is not None:
            vision_state = self.vision_labeler.refresh()
            if int(vision_state.get("last_labeled", 0)) > 0:
                for builder in self.builders.values():
                    builder._image_cache = None
                    builder._image_mtime = None
                # Revisit only link tails that can overlap the new image. Clearing
                # all progress markers would re-run 24 hours for every terminal
                # whenever the camera emits a frame, starving dashboard queries.
                latest_label = pd.to_datetime(
                    vision_state.get("latest_label_time"), errors="coerce"
                )
                if pd.notna(latest_label):
                    overlap = self.image_tolerance + pd.Timedelta(minutes=2)
                    for terminal_id, previous in list(
                        self._last_worker_link_times.items()
                    ):
                        latest_link = self.builders[terminal_id].latest_link_time()
                        if (
                            latest_link is not None
                            and abs(latest_link - latest_label) <= overlap
                        ):
                            self._last_worker_link_times[terminal_id] = (
                                previous - self.image_tolerance
                            )
                self._timeline_result_cache.clear()
        with self._lock:
            for terminal_id, builder in self.builders.items():
                latest = builder.latest_link_time()
                if latest is None:
                    states.append({"terminal_id": terminal_id, "status": "no_data", "persisted": 0})
                    continue
                previous = self._last_worker_link_times.get(terminal_id)
                if previous is not None and latest == previous:
                    states.append({"terminal_id": terminal_id, "status": "unchanged", "persisted": 0})
                    continue
                # Revisit a short overlap so the latest gauge-anchored minute can be
                # updated, without re-running the entire 24-hour tail every poll.
                lookback_start = latest - pd.Timedelta(hours=self.worker_lookback_hours)
                start = (
                    lookback_start
                    if previous is None
                    else max(lookback_start, previous - pd.Timedelta(minutes=2))
                )
                samples = self._build_minute_samples(
                    terminal_id, start, latest + pd.Timedelta(minutes=2), self.worker_max_passes
                )
                rows = self._predict_samples(terminal_id, samples)
                count = self.history.upsert_many(rows)
                self._last_worker_link_times[terminal_id] = latest
                persisted += count
                states.append(
                    {
                        "terminal_id": terminal_id,
                        "status": "ok" if rows else "no_valid_minute",
                        "latest_link_time": latest.isoformat(),
                        "persisted": count,
                    }
                )
            self._cache.clear()
            self._dropout_cache = None
            self.worker_state = {
                "status": "ok",
                "last_update": datetime.now().isoformat(timespec="seconds"),
                "persisted": persisted,
                "terminals": states,
                "vision_labels": vision_state,
            }
            return copy.deepcopy(self.worker_state)

    @staticmethod
    def _deduplicate_minute_rows(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["terminal_id"]), pd.Timestamp(row["pass_end"]).isoformat())
            previous = latest.get(key)
            if previous is None or str(row.get("inferred_at") or "") >= str(
                previous.get("inferred_at") or ""
            ):
                latest[key] = row
        return sorted(latest.values(), key=lambda row: row["pass_end"])

    @classmethod
    def _minute_model_time_series(
        cls,
        rows: list[dict[str, Any]],
        start: pd.Timestamp,
        end: pd.Timestamp,
        resolution_minutes: int,
    ) -> tuple[dict[str, dict[str, list[list[Any]]]], dict[str, Any]]:
        frequency = f"{resolution_minutes}min"
        index = pd.date_range(
            start=start.floor(frequency),
            end=end.ceil(frequency),
            freq=frequency,
            inclusive="left",
        )
        deduplicated = cls._deduplicate_minute_rows(rows)
        amount_arrays: dict[str, np.ndarray] = {}
        observed_amount_arrays: dict[str, np.ndarray] = {}
        rate_arrays: dict[str, np.ndarray] = {}
        series: dict[str, dict[str, list[list[Any]]]] = {}

        for terminal_id in dashboard_app.TERMINAL_NAMES:
            amounts = np.full(len(index), np.nan, dtype=np.float64)
            terminal_rows = [
                row for row in deduplicated if row["terminal_id"] == terminal_id
            ]
            by_bin: dict[pd.Timestamp, list[float]] = {}
            observed_by_bin: dict[pd.Timestamp, list[float]] = {}
            probability_by_bin: dict[pd.Timestamp, list[float]] = {}
            point_count_by_bin: dict[pd.Timestamp, list[int]] = {}
            for row in terminal_rows:
                anchor = pd.Timestamp(row["pass_end"]).floor(frequency)
                if start.floor(frequency) <= anchor < end.ceil(frequency):
                    by_bin.setdefault(anchor, []).append(
                        max(float(row["reported_rainfall_mm"]), 0.0)
                    )
                    if (
                        row.get("observed_available")
                        and row.get("observed_rainfall_mm") is not None
                    ):
                        observed_by_bin.setdefault(anchor, []).append(
                            max(float(row["observed_rainfall_mm"]), 0.0)
                        )
                    if row.get("rain_probability") is not None:
                        probability_by_bin.setdefault(anchor, []).append(
                            float(row["rain_probability"])
                        )
                    point_count_by_bin.setdefault(anchor, []).append(
                        int(row.get("points") or 0)
                    )
            observed_amounts = np.full(len(index), np.nan, dtype=np.float64)
            probabilities = np.full(len(index), np.nan, dtype=np.float64)
            point_counts = np.full(len(index), np.nan, dtype=np.float64)
            for anchor, values in by_bin.items():
                position = int(index.get_indexer([anchor])[0])
                if position >= 0:
                    amounts[position] = float(np.sum(values))
                    observed_values = observed_by_bin.get(anchor)
                    if observed_values:
                        observed_amounts[position] = float(np.sum(observed_values))
                    probability_values = probability_by_bin.get(anchor)
                    if probability_values:
                        probabilities[position] = float(max(probability_values))
                    point_values = point_count_by_bin.get(anchor)
                    if point_values:
                        point_counts[position] = float(sum(point_values))
            rates = amounts * 60.0 / float(resolution_minutes)
            cumulative = np.full(len(index), np.nan, dtype=np.float64)
            observed_cumulative = np.full(len(index), np.nan, dtype=np.float64)
            running_prediction = 0.0
            running_observed = 0.0
            prediction_started = False
            observed_started = False
            for position, amount in enumerate(amounts):
                if np.isfinite(amount):
                    running_prediction += float(amount)
                    prediction_started = True
                if np.isfinite(observed_amounts[position]):
                    running_observed += float(observed_amounts[position])
                    observed_started = True
                if prediction_started:
                    cumulative[position] = running_prediction
                if observed_started:
                    observed_cumulative[position] = running_observed
            amount_arrays[terminal_id] = amounts
            observed_amount_arrays[terminal_id] = observed_amounts
            rate_arrays[terminal_id] = rates
            series[terminal_id] = {
                "minute_amount_mm": [
                    [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, amounts)
                ],
                "observed_minute_amount_mm": [
                    [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, observed_amounts)
                ],
                "rain_probability": [
                    [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, probabilities)
                ],
                "input_phy_point_count": [
                    [timestamp.isoformat(), int(value) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, point_counts)
                ],
                "rate_mm_h": [
                    [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, rates)
                ],
                "coverage_cumulative_mm": [
                    [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, cumulative)
                ],
                "observed_coverage_cumulative_mm": [
                    [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, observed_cumulative)
                ],
            }

        consensus_amount = np.full(len(index), np.nan, dtype=np.float64)
        consensus_rate = np.full(len(index), np.nan, dtype=np.float64)
        consensus_cumulative = np.full(len(index), np.nan, dtype=np.float64)
        consensus_observed_cumulative = np.full(len(index), np.nan, dtype=np.float64)
        terminal_count = np.zeros(len(index), dtype=np.int64)
        spreads = []
        running = 0.0
        running_observed = 0.0
        consensus_started = False
        observed_started = False
        for position in range(len(index)):
            values = [
                float(amount_arrays[terminal_id][position])
                for terminal_id in dashboard_app.TERMINAL_NAMES
                if np.isfinite(amount_arrays[terminal_id][position])
            ]
            terminal_count[position] = len(values)
            if len(values) >= 2:
                consensus_amount[position] = float(np.median(values))
                consensus_rate[position] = (
                    consensus_amount[position] * 60.0 / float(resolution_minutes)
                )
                spreads.append(
                    (max(values) - min(values)) * 60.0 / float(resolution_minutes)
                )
                running += consensus_amount[position]
                consensus_started = True
                observed_values = [
                    float(observed_amount_arrays[terminal_id][position])
                    for terminal_id in dashboard_app.TERMINAL_NAMES
                    if np.isfinite(amount_arrays[terminal_id][position])
                    and np.isfinite(observed_amount_arrays[terminal_id][position])
                ]
                if observed_values:
                    running_observed += float(np.median(observed_values))
                    observed_started = True
            if consensus_started:
                consensus_cumulative[position] = running
            if observed_started:
                consensus_observed_cumulative[position] = running_observed

        pairwise = []
        terminal_ids = list(dashboard_app.TERMINAL_NAMES)
        for left_index, left_id in enumerate(terminal_ids):
            for right_id in terminal_ids[left_index + 1 :]:
                left = rate_arrays[left_id]
                right = rate_arrays[right_id]
                valid = np.isfinite(left) & np.isfinite(right)
                pairwise.append(
                    {
                        "left_terminal_id": left_id,
                        "right_terminal_id": right_id,
                        "overlap_bins": int(valid.sum()),
                        "mae_mm_h": (
                            round(float(np.mean(np.abs(left[valid] - right[valid]))), 6)
                            if valid.any()
                            else None
                        ),
                        "matched_passes": 0,
                        "matched_mae_mm": None,
                        "matched_rate_mae_mm_h": None,
                    }
                )
        series["consensus"] = {
            "rate_mm_h": [
                [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                for timestamp, value in zip(index, consensus_rate)
            ],
            "coverage_cumulative_mm": [
                [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                for timestamp, value in zip(index, consensus_cumulative)
            ],
            "observed_coverage_cumulative_mm": [
                [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                for timestamp, value in zip(index, consensus_observed_cumulative)
            ],
            "terminal_count": [
                [timestamp.isoformat(), int(value)]
                for timestamp, value in zip(index, terminal_count)
            ],
        }
        return series, {
            "consensus_bins": int(np.isfinite(consensus_rate).sum()),
            "mean_spread_mm_h": (
                round(float(np.mean(spreads)), 6) if spreads else None
            ),
            "pairwise": pairwise,
        }

    @classmethod
    def _minute_consistency_groups(
        cls, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        by_anchor: dict[str, list[dict[str, Any]]] = {}
        for row in cls._deduplicate_minute_rows(rows):
            anchor = pd.Timestamp(row["pass_end"]).isoformat()
            by_anchor.setdefault(anchor, []).append(row)
        groups = []
        for anchor, anchor_rows in sorted(by_anchor.items()):
            terminal_rows = {
                str(row["terminal_id"]): row for row in anchor_rows
            }
            if len(terminal_rows) < 2:
                continue
            model_rates = []
            terminals = []
            for terminal_id, row in sorted(terminal_rows.items()):
                rainfall = max(float(row["reported_rainfall_mm"]), 0.0)
                rate = rainfall * 60.0
                model_rates.append(rate)
                terminals.append(
                    {
                        "terminal_id": terminal_id,
                        "terminal_name": dashboard_app.TERMINAL_NAMES[terminal_id],
                        "satellite_id": int(row["satellite_id"]),
                        "pass_start": row["pass_start"],
                        "pass_end": row["pass_end"],
                        "duration_s": 60.0,
                        "rainfall_mm": round(rainfall, 6),
                        "rain_rate_mm_h": round(rate, 6),
                    }
                )
            observed_values = [
                float(row["observed_rainfall_mm"])
                for row in terminal_rows.values()
                if row.get("observed_available")
                and row.get("observed_rainfall_mm") is not None
            ]
            observed = float(np.median(observed_values)) if observed_values else None
            end = pd.Timestamp(anchor)
            groups.append(
                {
                    "group_id": len(groups),
                    "common_start": (end - pd.Timedelta(seconds=60)).isoformat(),
                    "common_end": end.isoformat(),
                    "common_duration_s": 60.0,
                    "terminal_count": len(terminals),
                    "same_satellite_id": len(
                        {row["satellite_id"] for row in terminals}
                    ) == 1,
                    "terminals": terminals,
                    "rate_min_mm_h": round(min(model_rates), 6),
                    "rate_max_mm_h": round(max(model_rates), 6),
                    "rate_range_mm_h": round(max(model_rates) - min(model_rates), 6),
                    "rate_mean_mm_h": round(float(np.mean(model_rates)), 6),
                    "rain_decision_agree": len(
                        {value > 0 for value in model_rates}
                    ) == 1,
                    "observed_overlap_rainfall_mm": (
                        round(observed, 6) if observed is not None else None
                    ),
                    "observed_overlap_rate_mm_h": (
                        round(observed * 60.0, 6) if observed is not None else None
                    ),
                }
            )
        rainy_agreement = [
            group
            for group in groups
            if (group["observed_overlap_rainfall_mm"] or 0.0) > 0
            and group["rain_decision_agree"]
            and group["rate_min_mm_h"] > 0
        ]
        best = min(
            rainy_agreement,
            key=lambda group: group["rate_range_mm_h"],
            default=None,
        )
        summary = {
            "group_count": len(groups),
            "triple_group_count": sum(
                group["terminal_count"] == 3 for group in groups
            ),
            "observed_rainy_group_count": sum(
                (group["observed_overlap_rainfall_mm"] or 0.0) > 0
                for group in groups
            ),
            "rain_decision_agreement": (
                round(
                    float(np.mean([group["rain_decision_agree"] for group in groups])),
                    6,
                )
                if groups
                else None
            ),
            "mean_rate_range_mm_h": (
                round(
                    float(np.mean([group["rate_range_mm_h"] for group in groups])),
                    6,
                )
                if groups
                else None
            ),
            "best_rainy_group_id": best["group_id"] if best else None,
            "best_rainy_group_start": best["common_start"] if best else None,
            "best_rainy_group_rate_range_mm_h": (
                best["rate_range_mm_h"] if best else None
            ),
        }
        return groups, summary

    def _ensure_timeline_history(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> None:
        first_day = start.normalize()
        last_day = (end - pd.Timedelta(microseconds=1)).normalize()
        for day in pd.date_range(first_day, last_day, freq="1D"):
            self.query_date(day.date(), max_passes=2000, force_recompute=False)

    def timeline(
        self, start: datetime, end: datetime, resolution_minutes: int
    ) -> dict[str, Any]:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        cache_key = (int(start_ts.value), int(end_ts.value), int(resolution_minutes))
        historical = end_ts <= pd.Timestamp.now().normalize()
        if historical and cache_key in self._timeline_result_cache:
            return self._timeline_result_cache[cache_key]
        cache_path = self.timeline_cache_dir / (
            f"{cache_key[0]}_{cache_key[1]}_{cache_key[2]}.json.gz"
        )
        if historical and cache_path.exists():
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                result = json.load(handle)
            if len(self._timeline_result_cache) >= 32:
                self._timeline_result_cache.pop(next(iter(self._timeline_result_cache)))
            self._timeline_result_cache[cache_key] = result
            return result
        self._ensure_timeline_history(pd.Timestamp(start), pd.Timestamp(end))
        result = super().timeline(start, end, resolution_minutes)
        minute_rows = self._deduplicate_minute_rows(self.history.query_range(
            start.isoformat(), end.isoformat(), limit=10000
        ))
        rows_by_terminal: dict[str, list[dict[str, Any]]] = {}
        for row in minute_rows:
            rows_by_terminal.setdefault(str(row["terminal_id"]), []).append(row)

        # Keep the original pass bars, but annotate each bar with the integral
        # of overlapping minute estimates so the existing tooltip remains useful.
        for pass_row in result["passes"]:
            pass_start = pd.Timestamp(pass_row["pass_start"])
            pass_end = pd.Timestamp(pass_row["pass_end"])
            amount = 0.0
            probabilities = []
            found = False
            for row in rows_by_terminal.get(pass_row["terminal_id"], []):
                row_start = pd.Timestamp(row["pass_start"])
                row_end = pd.Timestamp(row["pass_end"])
                overlap = (min(pass_end, row_end) - max(pass_start, row_start)).total_seconds()
                if overlap > 0:
                    amount += float(row["reported_rainfall_mm"]) * min(overlap / 60.0, 1.0)
                    probabilities.append(float(row.get("rain_probability") or 0.0))
                    found = True
            pass_row["reported_rainfall_mm"] = round(amount, 6) if found else None
            pass_row["rain_probability"] = max(probabilities) if probabilities else None

        result["model_series"], result["consistency"] = self._minute_model_time_series(
            minute_rows, start_ts, end_ts, resolution_minutes
        )
        result["cumulative_comparison_mode"] = "model_valid_minutes"
        (
            result["consistency_groups"],
            result["consistency_group_summary"],
        ) = self._minute_consistency_groups(minute_rows)
        # The gauge rainfall field is a one-minute amount stored with a factor-10
        # scale error; expose the corrected unit in this minute-model service.
        observed_running = 0.0
        for row in result["rain"]:
            if row.get("rainfall") is not None:
                row["rainfall"] = round(float(row["rainfall"]) * 0.1, 6)
                observed_running += float(row["rainfall"])
                row["rainfall_cumulative_delta"] = round(observed_running, 6)
        result["interpretation_note"] = (
            f"模型曲线为{self.sampling_ratio_label}模型逐分钟反演；"
            "每个值对应前一分钟累计雨量，"
            "雨强按分钟雨量乘60换算。累计图中每台终端的模型值与真实值"
            "仅累加该终端具有有效反演的相同分钟，不包含无PHY覆盖分钟。"
        )
        if historical:
            if len(self._timeline_result_cache) >= 32:
                self._timeline_result_cache.pop(next(iter(self._timeline_result_cache)))
            self._timeline_result_cache[cache_key] = result
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
            temporary.replace(cache_path)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-terminal minute-rain dashboard")
    parser.add_argument("--config-002", required=True)
    parser.add_argument("--config-003", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--fallback-checkpoint-path")
    parser.add_argument(
        "--transfer-checkpoint-path",
        help="Optional full-position checkpoint used only by terminals 002/003.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-db-path", required=True)
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--worker-lookback-hours", type=float, default=24.0)
    parser.add_argument("--worker-max-samples", type=int, default=256)
    parser.add_argument("--min-phy-points", type=int, default=10)
    parser.add_argument("--fallback-min-phy-points", type=int, default=3)
    parser.add_argument("--position-tolerance-s", type=float, default=5.0)
    parser.add_argument("--weather-tolerance-s", type=float, default=60.0)
    parser.add_argument("--image-tolerance-s", type=float, default=600.0)
    parser.add_argument("--camera-input-dir")
    parser.add_argument("--vision-weights")
    parser.add_argument("--vision-full-csv")
    parser.add_argument("--vision-slim-csv")
    parser.add_argument("--vision-refresh-interval-s", type=float, default=60.0)
    parser.add_argument("--vision-max-images-per-refresh", type=int, default=8192)
    parser.add_argument("--vision-batch-size", type=int, default=256)
    parser.add_argument("--vision-num-workers", type=int, default=8)
    parser.add_argument("--probability-threshold", type=float)
    parser.add_argument(
        "--backup-db-001",
        help="Recovered legacy-terminal database merged with the live 001 link stream.",
    )
    parser.add_argument(
        "--backup-db-002",
        help="Recovered 002 database merged with the live new-protocol stream.",
    )
    parser.add_argument(
        "--backup-db-003",
        help="003 backup database merged with the live new-protocol stream.",
    )
    parser.add_argument(
        "--link-analysis-dir",
        default=str(STAGE1_ROOT / "link_reliability_analysis" / "artifacts"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8041)
    args = parser.parse_args()
    runner = MinuteThreeTerminalRunner(
        config_002=Path(args.config_002),
        config_003=Path(args.config_003),
        checkpoint_path=Path(args.checkpoint_path),
        fallback_checkpoint_path=(
            Path(args.fallback_checkpoint_path) if args.fallback_checkpoint_path else None
        ),
        transfer_checkpoint_path=(
            Path(args.transfer_checkpoint_path) if args.transfer_checkpoint_path else None
        ),
        device_name=args.device,
        history_db_path=Path(args.history_db_path),
        poll_interval_s=args.poll_interval_s,
        worker_lookback_hours=args.worker_lookback_hours,
        worker_max_samples=args.worker_max_samples,
        link_analysis_dir=Path(args.link_analysis_dir),
        min_phy_points=args.min_phy_points,
        fallback_min_phy_points=args.fallback_min_phy_points,
        position_tolerance_s=args.position_tolerance_s,
        weather_tolerance_s=args.weather_tolerance_s,
        image_tolerance_s=args.image_tolerance_s,
        probability_threshold=args.probability_threshold,
        backup_db_001=Path(args.backup_db_001) if args.backup_db_001 else None,
        backup_db_002=Path(args.backup_db_002) if args.backup_db_002 else None,
        backup_db_003=Path(args.backup_db_003) if args.backup_db_003 else None,
        camera_input_dir=(
            Path(args.camera_input_dir) if args.camera_input_dir else None
        ),
        vision_weights=Path(args.vision_weights) if args.vision_weights else None,
        vision_full_csv=Path(args.vision_full_csv) if args.vision_full_csv else None,
        vision_slim_csv=Path(args.vision_slim_csv) if args.vision_slim_csv else None,
        vision_refresh_interval_s=args.vision_refresh_interval_s,
        vision_max_images_per_refresh=args.vision_max_images_per_refresh,
        vision_batch_size=args.vision_batch_size,
        vision_num_workers=args.vision_num_workers,
    )
    _configure_minute_dashboard_text(runner.sampling_ratio_label)
    uvicorn.run(dashboard_app.create_app(runner), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
