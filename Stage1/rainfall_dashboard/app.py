#!/usr/bin/env python3
"""Three-terminal Stage1 rainfall retrieval dashboard and API."""
from __future__ import annotations

import copy
import json
import re
import sqlite3
import sys
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MOE_SRC = ROOT.parent / "MoE" / "lora-moe" / "src"
STATIC_DIR = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(MOE_SRC))

from lora_moe.history import RainRetrievalHistory  # noqa: E402


TERMINAL_NAMES = {
    "01-31-0005-0001": "终端 001",
    "01-31-0005-0002": "终端 002",
    "01-31-0005-0003": "终端 003",
}

TERMINAL_NOMINAL_INTERVAL_S = {
    "01-31-0005-0001": 2.0,
    "01-31-0005-0002": 4.0,
    "01-31-0005-0003": 4.0,
}

RAIN_RATE_BINS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")]
RAIN_RATE_LABELS = [
    "0-0.1",
    "0.1-0.5",
    "0.5-1",
    "1-2",
    "2-5",
    "5-10",
    "10-20",
    "20-50",
    ">=50",
]
LINK_ANALYSIS_RAIN_EDGES = [
    -1e-12, 1e-12, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")
]
LINK_ANALYSIS_RAIN_LABELS = [
    "0", "0-0.1", "0.1-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20",
    "20-50", ">=50",
]
DISPLAY_STRONG_SNR_THRESHOLD_DB = -10.0


def load_weather_station(db_path: str, start_time: str | None = None,
                         end_time: str | None = None) -> pd.DataFrame:
    clauses, params = [], []
    if start_time:
        clauses.append("datetime(datetime) >= datetime(?)")
        params.append(start_time)
    if end_time:
        clauses.append("datetime(datetime) <= datetime(?)")
        params.append(end_time)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT datetime AS timestamp, temperature, humidity, pressure,
               rainfall, rainfall_cumulative
        FROM weather_station {where} ORDER BY datetime
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(query, connection, params=params)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()


def _nearest_cumulative(station: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    if station.empty or "rainfall_cumulative" not in station:
        return None
    index = station.index.get_indexer(
        [timestamp], method="nearest", tolerance=pd.Timedelta("5min")
    )[0]
    if index < 0:
        return None
    value = pd.to_numeric(pd.Series([station.iloc[index]["rainfall_cumulative"]]),
                          errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def compute_pass_labels(station: pd.DataFrame, start: pd.Timestamp,
                        end: pd.Timestamp) -> tuple[np.ndarray | None, dict]:
    start_value = _nearest_cumulative(station, start)
    end_value = _nearest_cumulative(station, end)
    if start_value is None or end_value is None:
        return None, {"drop_reason": "missing_cumulative_boundary"}
    rainfall = end_value - start_value
    if rainfall < -1e-4:
        return None, {"drop_reason": "negative_cumulative_delta"}
    window = pd.to_numeric(station.loc[start:end].get("rainfall"), errors="coerce")
    return np.asarray([max(rainfall, 0.0)], dtype=np.float32), {
        "rain_rate_mean": float(window.mean()) if window.notna().any() else None,
        "rain_rate_max": float(window.max()) if window.notna().any() else None,
        "rainy_ratio": float((window > 0).mean()) if window.notna().any() else None,
    }


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    max_passes: int = Field(default=500, ge=1, le=2000)


class OllamaGenerateRequest(BaseModel):
    model: str = "stage1-three-terminal"
    prompt: str = Field(min_length=1)
    stream: bool = False


def _parse_query_date(text: str, now: datetime | None = None) -> date:
    now = now or datetime.now()
    if "前天" in text:
        return (now - timedelta(days=2)).date()
    if "昨天" in text or "昨日" in text:
        return (now - timedelta(days=1)).date()
    if "今天" in text or "今日" in text:
        return now.date()
    match = re.search(r"(?:(\d{4})[年/-])?(\d{1,2})[月/-](\d{1,2})日?", text)
    if match:
        return date(
            int(match.group(1) or now.year),
            int(match.group(2)),
            int(match.group(3)),
        )
    raise ValueError("未识别日期，请输入“今天”“昨天”或“2026-07-23”。")


class ThreeTerminalRunner:
    def __init__(
        self,
        config_002: Path,
        config_003: Path,
        device_name: str,
        history_db_path: Path,
        poll_interval_s: float,
        worker_lookback_hours: float,
        worker_max_passes: int,
        link_analysis_dir: Path,
        runtimes: dict[str, Any] | None = None,
        builders: dict[str, Any] | None = None,
    ):
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        if not runtimes or not builders or set(runtimes) != set(builders):
            raise ValueError("dashboard requires matching minute-service readers")
        self.runtimes = runtimes
        self.builders = builders
        self.cfg, self.meta, self.mapper, self.model = {}, {}, None, None
        self.regression_output = "conditional"
        self.probability_threshold = 0.5
        self.model_version = "reader-only"
        self.history = RainRetrievalHistory(history_db_path)
        self.poll_interval_s = max(float(poll_interval_s), 0.0)
        self.worker_lookback_hours = max(float(worker_lookback_hours), 0.1)
        self.worker_max_passes = max(int(worker_max_passes), 1)
        self.link_analysis_dir = link_analysis_dir
        self._link_analysis_summary: dict[str, Any] | None = None
        self._link_analysis_passes: pd.DataFrame | None = None
        self._link_analysis_gaps: pd.DataFrame | None = None
        self._position_link_dashboard: dict[str, Any] | None = None
        self._position_link_passes: pd.DataFrame | None = None
        self._position_retrieval_cache: (
            tuple[datetime, pd.DataFrame, int] | None
        ) = None
        self._cache: dict[tuple[str, int], tuple[datetime, dict[str, Any]]] = {}
        self._cache_ttl = timedelta(seconds=30)
        self._dropout_cache: tuple[float, datetime, dict[str, Any]] | None = None
        self._dropout_cache_ttl = timedelta(minutes=5)
        self._lock = threading.Lock()
        self._worker_stop = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._last_worker_link_times: dict[str, pd.Timestamp] = {}
        self.worker_state: dict[str, Any] = {
            "status": "not_started",
            "last_update": None,
            "persisted": 0,
        }

    def _read_terminal_link(
        self, terminal_id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Read one terminal's link stream; minute service extends this with backups."""
        return self.builders[terminal_id]._read_link(start, end)

    @staticmethod
    def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        reported = [float(row["reported_rainfall_mm"]) for row in rows]
        rainy = [value for value in reported if value > 0]
        observed = [
            float(row["observed_rainfall_mm"])
            for row in rows
            if row.get("observed_available") and row.get("observed_rainfall_mm") is not None
        ]
        errors = [
            float(row["absolute_error_mm"])
            for row in rows
            if row.get("absolute_error_mm") is not None
        ]
        return {
            "pass_count": len(rows),
            "rainy_pass_count": len(rainy),
            "max_reported_rainfall_mm": round(max(reported), 6) if rows else 0.0,
            "mean_reported_rainfall_mm": (
                round(sum(reported) / len(reported), 6) if rows else 0.0
            ),
            "observed_pass_count": len(observed),
            "observed_rainy_pass_count": sum(value > 0 for value in observed),
            "max_observed_rainfall_mm": (
                round(max(observed), 6) if observed else None
            ),
            "mae_mm": round(sum(errors) / len(errors), 6) if errors else None,
        }

    def _result_from_rows(
        self,
        terminal_id: str,
        query_date: date,
        rows: list[dict[str, Any]],
        source: str,
    ) -> dict[str, Any]:
        latest = self.builders[terminal_id].latest_link_time()
        query_latest_pass_end = max(
            (pd.Timestamp(row["pass_end"]) for row in rows),
            default=None,
        )
        return {
            "terminal_id": terminal_id,
            "terminal_name": TERMINAL_NAMES[terminal_id],
            "status": "ok" if rows else "no_link_pass",
            "source": source,
            "query_date": query_date.isoformat(),
            "query_latest_pass_end": (
                query_latest_pass_end.isoformat()
                if query_latest_pass_end is not None
                else None
            ),
            "latest_link_time": latest.isoformat() if latest is not None else None,
            "latest_link_age_s": (
                round(max((pd.Timestamp.now() - latest).total_seconds(), 0.0), 3)
                if latest is not None
                else None
            ),
            "summary": self._summary(rows),
            "predictions": rows,
        }

    def query_date(
        self,
        query_date: date,
        max_passes: int = 500,
        *,
        force_recompute: bool = False,
    ) -> dict[str, Any]:
        key = (
            self.model_version,
            query_date.isoformat(),
            int(max_passes),
            bool(force_recompute),
        )
        with self._lock:
            if key in self._cache:
                cached_at, cached = self._cache[key]
                if datetime.now() - cached_at <= self._cache_ttl:
                    return copy.deepcopy(cached)
            terminals = []
            for terminal_id in self.builders:
                use_history = not force_recompute and (
                    query_date == date.today()
                    or self.history.is_day_materialized(
                        query_date, terminal_id, self.model_version
                    )
                )
                if use_history:
                    rows = self.history.query_day(
                        query_date,
                        terminal_id=terminal_id,
                        model_version=self.model_version,
                        limit=max_passes,
                    )
                    terminals.append(
                        self._result_from_rows(
                            terminal_id, query_date, rows, "history"
                        )
                    )
                else:
                    terminals.append(
                        self._terminal_result(
                            terminal_id, query_date, max_passes
                        )
                    )
            result = {
                "status": "ok",
                "query_date": query_date.isoformat(),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "model_version": self.model_version,
                "aggregation_note": (
                    "每条模型记录对应一个雨量计锚点前的一分钟窗口。"
                ),
                "terminals": terminals,
            }
            self._cache[key] = (datetime.now(), copy.deepcopy(result))
            return result

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                state = self.update_recent_once()
                print(
                    "[three-terminal-worker] "
                    + json.dumps(state, ensure_ascii=False),
                    flush=True,
                )
            except Exception as exc:
                self.worker_state = {
                    "status": "error",
                    "last_update": datetime.now().isoformat(timespec="seconds"),
                    "error": repr(exc),
                }
                print(
                    "[three-terminal-worker] "
                    + json.dumps(self.worker_state, ensure_ascii=False),
                    flush=True,
                )
            self._worker_stop.wait(self.poll_interval_s)

    def start_worker(self) -> None:
        if self.poll_interval_s <= 0 or (
            self._worker_thread is not None and self._worker_thread.is_alive()
        ):
            return
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="three-terminal-history-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop_worker(self) -> None:
        self._worker_stop.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=max(self.poll_interval_s, 1.0) + 5.0)

    @staticmethod
    def _timeline_passes(
        link: pd.DataFrame,
        terminal_id: str,
        gap_threshold_s: float,
        min_pass_points: int,
    ) -> list[dict[str, Any]]:
        if link.empty:
            return []
        passes = []
        for satellite_id, satellite_link in link.groupby("satelliteId"):
            satellite_link = satellite_link.sort_index()
            gaps = satellite_link.index.to_series().diff().dt.total_seconds()
            segment_ids = (
                (gaps > gap_threshold_s) | gaps.isna()
            ).cumsum()
            for _, segment in satellite_link.groupby(segment_ids):
                if len(segment) < min_pass_points:
                    continue
                internal_gaps = (
                    segment.index.to_series().diff().dt.total_seconds().dropna()
                )
                snr = pd.to_numeric(segment["snr"], errors="coerce")
                strong_snr = snr.ge(DISPLAY_STRONG_SNR_THRESHOLD_DB)
                passes.append(
                    {
                        "terminal_id": terminal_id,
                        "satellite_id": int(satellite_id),
                        "pass_start": segment.index[0].isoformat(),
                        "pass_end": segment.index[-1].isoformat(),
                        "points": int(len(segment)),
                        "duration_s": round(
                            (
                                segment.index[-1] - segment.index[0]
                            ).total_seconds(),
                            3,
                        ),
                        "max_internal_gap_s": (
                            round(float(internal_gaps.max()), 3)
                            if len(internal_gaps)
                            else 0.0
                        ),
                        "mean_rssi": round(
                            float(
                                pd.to_numeric(
                                    segment["rssi"], errors="coerce"
                                ).mean()
                            ),
                            4,
                        ),
                        "mean_snr": round(
                            float(snr.mean()),
                            4,
                        ),
                        "strong_snr_points": int(strong_snr.sum()),
                        "strong_snr_ratio": round(float(strong_snr.mean()), 6),
                        "snr_threshold_db": DISPLAY_STRONG_SNR_THRESHOLD_DB,
                    }
                )
        passes.sort(key=lambda item: item["pass_start"])
        return passes

    @staticmethod
    def _model_time_series(
        passes: list[dict[str, Any]],
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
        bin_seconds = float(resolution_minutes * 60)
        series: dict[str, dict[str, list[list[Any]]]] = {}
        rate_arrays: dict[str, np.ndarray] = {}
        coverage_arrays: dict[str, np.ndarray] = {}

        for terminal_id in TERMINAL_NAMES:
            numerator = np.zeros(len(index), dtype=np.float64)
            overlap_seconds = np.zeros(len(index), dtype=np.float64)
            for pass_row in passes:
                if pass_row["terminal_id"] != terminal_id:
                    continue
                rainfall = pass_row.get("reported_rainfall_mm")
                duration_s = float(pass_row.get("duration_s") or 0.0)
                if rainfall is None or duration_s <= 0:
                    continue
                pass_start = max(pd.Timestamp(pass_row["pass_start"]), start)
                pass_end = min(pd.Timestamp(pass_row["pass_end"]), end)
                if pass_end <= pass_start:
                    continue
                rate = max(float(rainfall), 0.0) * 3600.0 / duration_s
                first = max(int(index.searchsorted(pass_start, side="right")) - 1, 0)
                last = min(int(index.searchsorted(pass_end, side="left")), len(index) - 1)
                for position in range(first, last + 1):
                    bin_start = index[position]
                    bin_end = bin_start + pd.Timedelta(seconds=bin_seconds)
                    overlap = (
                        min(pass_end, bin_end) - max(pass_start, bin_start)
                    ).total_seconds()
                    if overlap > 0:
                        numerator[position] += rate * overlap
                        overlap_seconds[position] += overlap

            rates = np.full(len(index), np.nan, dtype=np.float64)
            valid = overlap_seconds > 0
            rates[valid] = numerator[valid] / overlap_seconds[valid]
            cumulative = np.full(len(index), np.nan, dtype=np.float64)
            running = 0.0
            has_coverage = False
            for position, rate in enumerate(rates):
                if np.isfinite(rate):
                    covered = min(overlap_seconds[position], bin_seconds)
                    running += float(rate) * covered / 3600.0
                    has_coverage = True
                if has_coverage:
                    cumulative[position] = running
            rate_arrays[terminal_id] = rates
            coverage_arrays[terminal_id] = overlap_seconds
            series[terminal_id] = {
                "rate_mm_h": [
                    [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, rates)
                ],
                "coverage_cumulative_mm": [
                    [timestamp.isoformat(), round(float(value), 6) if np.isfinite(value) else None]
                    for timestamp, value in zip(index, cumulative)
                ],
            }

        consensus_rate = np.full(len(index), np.nan, dtype=np.float64)
        consensus_cumulative = np.full(len(index), np.nan, dtype=np.float64)
        terminal_count = np.zeros(len(index), dtype=np.int64)
        spread = np.full(len(index), np.nan, dtype=np.float64)
        running = 0.0
        has_consensus = False
        for position in range(len(index)):
            values = [
                float(rate_arrays[terminal_id][position])
                for terminal_id in TERMINAL_NAMES
                if np.isfinite(rate_arrays[terminal_id][position])
            ]
            terminal_count[position] = len(values)
            if len(values) >= 2:
                consensus_rate[position] = median(values)
                spread[position] = max(values) - min(values)
                covered = max(
                    min(coverage_arrays[terminal_id][position], bin_seconds)
                    for terminal_id in TERMINAL_NAMES
                    if np.isfinite(rate_arrays[terminal_id][position])
                )
                running += consensus_rate[position] * covered / 3600.0
                has_consensus = True
            if has_consensus:
                consensus_cumulative[position] = running

        pairwise = []
        terminal_ids = list(TERMINAL_NAMES)
        for left_index, left_id in enumerate(terminal_ids):
            for right_id in terminal_ids[left_index + 1 :]:
                left = rate_arrays[left_id]
                right = rate_arrays[right_id]
                valid = np.isfinite(left) & np.isfinite(right)
                left_passes = [
                    item
                    for item in passes
                    if item["terminal_id"] == left_id
                    and item.get("reported_rainfall_mm") is not None
                ]
                right_by_satellite: dict[int, list[dict[str, Any]]] = {}
                for item in passes:
                    if item["terminal_id"] == right_id and item.get("reported_rainfall_mm") is not None:
                        right_by_satellite.setdefault(int(item["satellite_id"]), []).append(item)
                used_right: set[tuple[int, str]] = set()
                rainfall_errors = []
                rate_errors = []
                for left_pass in left_passes:
                    satellite_id = int(left_pass["satellite_id"])
                    left_start = pd.Timestamp(left_pass["pass_start"])
                    left_end = pd.Timestamp(left_pass["pass_end"])
                    candidates = []
                    for right_pass in right_by_satellite.get(satellite_id, []):
                        right_key = (satellite_id, right_pass["pass_start"])
                        if right_key in used_right:
                            continue
                        right_start = pd.Timestamp(right_pass["pass_start"])
                        right_end = pd.Timestamp(right_pass["pass_end"])
                        overlap = (
                            min(left_end, right_end) - max(left_start, right_start)
                        ).total_seconds()
                        start_difference = abs((right_start - left_start).total_seconds())
                        if overlap > 0 or start_difference <= 60.0:
                            candidates.append((start_difference, right_key, right_pass))
                    if not candidates:
                        continue
                    _, right_key, right_pass = min(candidates, key=lambda item: item[0])
                    used_right.add(right_key)
                    left_rain = float(left_pass["reported_rainfall_mm"])
                    right_rain = float(right_pass["reported_rainfall_mm"])
                    rainfall_errors.append(abs(left_rain - right_rain))
                    left_duration = max(float(left_pass["duration_s"]), 1e-6)
                    right_duration = max(float(right_pass["duration_s"]), 1e-6)
                    rate_errors.append(
                        abs(left_rain * 3600.0 / left_duration - right_rain * 3600.0 / right_duration)
                    )
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
                        "matched_passes": len(rainfall_errors),
                        "matched_mae_mm": (
                            round(float(np.mean(rainfall_errors)), 6)
                            if rainfall_errors
                            else None
                        ),
                        "matched_rate_mae_mm_h": (
                            round(float(np.mean(rate_errors)), 6)
                            if rate_errors
                            else None
                        ),
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
            "terminal_count": [
                [timestamp.isoformat(), int(value)]
                for timestamp, value in zip(index, terminal_count)
            ],
        }
        summary = {
            "consensus_bins": int(np.isfinite(consensus_rate).sum()),
            "mean_spread_mm_h": (
                round(float(np.nanmean(spread)), 6)
                if np.isfinite(spread).any()
                else None
            ),
            "pairwise": pairwise,
        }
        return series, summary

    @staticmethod
    def _pass_consistency_groups(
        passes: list[dict[str, Any]], station: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        valid_passes = [
            item
            for item in passes
            if item.get("reported_rainfall_mm") is not None
            and float(item.get("duration_s") or 0.0) > 0
        ]
        if len(valid_passes) < 2:
            return [], {
                "group_count": 0,
                "triple_group_count": 0,
                "rain_decision_agreement": None,
                "mean_rate_range_mm_h": None,
            }

        parents = list(range(len(valid_passes)))
        members = [{index} for index in range(len(valid_passes))]
        terminals = [{valid_passes[index]["terminal_id"]} for index in range(len(valid_passes))]

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        candidates = []
        for left_index, left in enumerate(valid_passes):
            left_start = pd.Timestamp(left["pass_start"])
            left_end = pd.Timestamp(left["pass_end"])
            for right_index in range(left_index + 1, len(valid_passes)):
                right = valid_passes[right_index]
                if left["terminal_id"] == right["terminal_id"]:
                    continue
                right_start = pd.Timestamp(right["pass_start"])
                right_end = pd.Timestamp(right["pass_end"])
                overlap_s = (
                    min(left_end, right_end) - max(left_start, right_start)
                ).total_seconds()
                if overlap_s < 10.0:
                    continue
                shorter_s = min(float(left["duration_s"]), float(right["duration_s"]))
                fraction = overlap_s / max(shorter_s, 1e-6)
                if fraction >= 0.5:
                    candidates.append((fraction, overlap_s, left_index, right_index))

        for _, _, left_index, right_index in sorted(candidates, reverse=True):
            left_root, right_root = find(left_index), find(right_index)
            if left_root == right_root or terminals[left_root] & terminals[right_root]:
                continue
            combined = members[left_root] | members[right_root]
            common_start = max(
                pd.Timestamp(valid_passes[index]["pass_start"]) for index in combined
            )
            common_end = min(
                pd.Timestamp(valid_passes[index]["pass_end"]) for index in combined
            )
            if (common_end - common_start).total_seconds() < 10.0:
                continue
            parents[right_root] = left_root
            members[left_root] = combined
            terminals[left_root] |= terminals[right_root]

        components = {}
        for index in range(len(valid_passes)):
            components.setdefault(find(index), set()).add(index)

        groups = []
        for component in components.values():
            if len(component) < 2:
                continue
            rows = [valid_passes[index] for index in component]
            common_start = max(pd.Timestamp(row["pass_start"]) for row in rows)
            common_end = min(pd.Timestamp(row["pass_end"]) for row in rows)
            model_rates = []
            terminal_rows = []
            for row in sorted(rows, key=lambda item: item["terminal_id"]):
                duration_s = float(row["duration_s"])
                rainfall_mm = float(row["reported_rainfall_mm"])
                rate = rainfall_mm * 3600.0 / duration_s
                model_rates.append(rate)
                terminal_rows.append(
                    {
                        "terminal_id": row["terminal_id"],
                        "terminal_name": TERMINAL_NAMES[row["terminal_id"]],
                        "satellite_id": int(row["satellite_id"]),
                        "pass_start": row["pass_start"],
                        "pass_end": row["pass_end"],
                        "duration_s": round(duration_s, 3),
                        "rainfall_mm": round(rainfall_mm, 6),
                        "rain_rate_mm_h": round(rate, 6),
                    }
                )
            labels, _ = compute_pass_labels(station, common_start, common_end)
            observed_mm = float(labels[0]) if labels is not None else None
            common_duration_s = (common_end - common_start).total_seconds()
            groups.append(
                {
                    "group_id": len(groups),
                    "common_start": common_start.isoformat(),
                    "common_end": common_end.isoformat(),
                    "common_duration_s": round(common_duration_s, 3),
                    "terminal_count": len(rows),
                    "same_satellite_id": len({int(row["satellite_id"]) for row in rows}) == 1,
                    "terminals": terminal_rows,
                    "rate_min_mm_h": round(min(model_rates), 6),
                    "rate_max_mm_h": round(max(model_rates), 6),
                    "rate_range_mm_h": round(max(model_rates) - min(model_rates), 6),
                    "rate_mean_mm_h": round(float(np.mean(model_rates)), 6),
                    "rain_decision_agree": len({value > 0 for value in model_rates}) == 1,
                    "observed_overlap_rainfall_mm": (
                        round(observed_mm, 6) if observed_mm is not None else None
                    ),
                    "observed_overlap_rate_mm_h": (
                        round(observed_mm * 3600.0 / common_duration_s, 6)
                        if observed_mm is not None and common_duration_s > 0
                        else None
                    ),
                }
            )
        groups.sort(key=lambda item: item["common_start"])
        for group_id, group in enumerate(groups):
            group["group_id"] = group_id
        rainy_agreement_groups = [
            group
            for group in groups
            if (group["observed_overlap_rainfall_mm"] or 0.0) > 0
            and group["rain_decision_agree"]
            and group["rate_min_mm_h"] > 0
        ]
        best_rainy_group = min(
            rainy_agreement_groups,
            key=lambda group: group["rate_range_mm_h"],
            default=None,
        )
        summary = {
            "group_count": len(groups),
            "triple_group_count": sum(group["terminal_count"] == 3 for group in groups),
            "observed_rainy_group_count": sum(
                (group["observed_overlap_rainfall_mm"] or 0.0) > 0 for group in groups
            ),
            "rain_decision_agreement": (
                round(float(np.mean([group["rain_decision_agree"] for group in groups])), 6)
                if groups
                else None
            ),
            "mean_rate_range_mm_h": (
                round(float(np.mean([group["rate_range_mm_h"] for group in groups])), 6)
                if groups
                else None
            ),
            "best_rainy_group_id": (
                best_rainy_group["group_id"] if best_rainy_group else None
            ),
            "best_rainy_group_start": (
                best_rainy_group["common_start"] if best_rainy_group else None
            ),
            "best_rainy_group_rate_range_mm_h": (
                best_rainy_group["rate_range_mm_h"] if best_rainy_group else None
            ),
        }
        return groups, summary

    def _link_time_bounds(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        bounds = []
        db_path = next(iter(self.runtimes.values())).db_path
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            for terminal_id in self.builders:
                table = "phy_data" if terminal_id.endswith("0001") else "phy_bb_data"
                row = conn.execute(
                    f"SELECT MIN(localTime), MAX(localTime) FROM {table} WHERE terminalId = ?",
                    [terminal_id],
                ).fetchone()
                if row and row[0] and row[1]:
                    bounds.append((pd.Timestamp(row[0]), pd.Timestamp(row[1])))
        if not bounds:
            return None, None
        return max(item[0] for item in bounds), max(item[1] for item in bounds)

    def link_dropout_stats(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        session_gap_s: float = 900.0,
    ) -> dict[str, Any]:
        now = datetime.now()
        if start is None and end is None and self._dropout_cache is not None:
            cached_gap_s, cached_at, cached = self._dropout_cache
            if (
                cached_gap_s == session_gap_s
                and now - cached_at <= self._dropout_cache_ttl
            ):
                return copy.deepcopy(cached)
        default_start, default_end = self._link_time_bounds()
        start_ts = pd.Timestamp(start) if start is not None else default_start
        end_ts = pd.Timestamp(end) if end is not None else default_end
        if start_ts is None or end_ts is None or end_ts <= start_ts:
            return {"status": "no_data", "rows": [], "terminals": []}

        station = load_weather_station(
            str(next(iter(self.runtimes.values())).db_path),
            start_time=(start_ts - pd.Timedelta(minutes=5)).isoformat(),
            end_time=(end_ts + pd.Timedelta(minutes=5)).isoformat(),
        )
        cumulative_by_day: dict[date, tuple[np.ndarray, np.ndarray]] = {}
        if not station.empty:
            cumulative = pd.to_numeric(
                station["rainfall_cumulative"], errors="coerce"
            )
            valid_station = station.loc[cumulative.notna()].copy()
            valid_station["_cumulative"] = cumulative.loc[cumulative.notna()]
            for day_value, day_rows in valid_station.groupby(
                valid_station.index.date
            ):
                if len(day_rows) >= 2:
                    cumulative_by_day[day_value] = (
                        np.asarray(day_rows.index.view("int64"), dtype=np.float64),
                        day_rows["_cumulative"].to_numpy(dtype=np.float64),
                    )

        def rainfall_delta(pass_start: pd.Timestamp, pass_end: pd.Timestamp) -> float | None:
            if pass_start.date() != pass_end.date():
                return None
            day_values = cumulative_by_day.get(pass_start.date())
            if day_values is None:
                return None
            timestamps, values = day_values
            if pass_start.value < timestamps[0] or pass_end.value > timestamps[-1]:
                return None
            start_value = float(np.interp(float(pass_start.value), timestamps, values))
            end_value = float(np.interp(float(pass_end.value), timestamps, values))
            delta = end_value - start_value
            return max(delta, 0.0) if delta >= -1e-4 else None

        sessions = []
        for terminal_id, builder in self.builders.items():
            link = self._read_terminal_link(terminal_id, start_ts, end_ts)
            terminal_sessions = self._timeline_passes(
                link,
                terminal_id,
                session_gap_s,
                min_pass_points=self.runtimes[terminal_id].min_pass_points,
            )
            nominal_s = TERMINAL_NOMINAL_INTERVAL_S[terminal_id]
            for item in terminal_sessions:
                duration_s = float(item["duration_s"])
                if duration_s < nominal_s:
                    continue
                actual = int(item["points"])
                expected_2s = max(int(np.floor(duration_s / 2.0)) + 1, 1)
                expected_adjusted = max(int(np.floor(duration_s / nominal_s)) + 1, 1)
                rainfall_mm = rainfall_delta(
                    pd.Timestamp(item["pass_start"]),
                    pd.Timestamp(item["pass_end"]),
                )
                if rainfall_mm is None:
                    continue
                rain_rate = rainfall_mm * 3600.0 / duration_s
                sessions.append(
                    {
                        **item,
                        "rainfall_mm": round(rainfall_mm, 6),
                        "rain_rate_mm_h": round(rain_rate, 6),
                        "nominal_interval_s": nominal_s,
                        "expected_points_2s": expected_2s,
                        "expected_points_adjusted": expected_adjusted,
                        "dropout_rate_2s": round(max(1.0 - actual / expected_2s, 0.0), 6),
                        "dropout_rate_adjusted": round(
                            max(1.0 - actual / expected_adjusted, 0.0), 6
                        ),
                    }
                )

        aggregate_rows = []
        for terminal_id in self.builders:
            terminal_sessions = [
                item for item in sessions if item["terminal_id"] == terminal_id
            ]
            rates = np.asarray(
                [item["rain_rate_mm_h"] for item in terminal_sessions], dtype=np.float64
            )
            bin_ids = np.digitize(rates, RAIN_RATE_BINS[1:-1], right=False)
            for bin_index, label in enumerate(RAIN_RATE_LABELS):
                selected = [
                    item
                    for item, item_bin in zip(terminal_sessions, bin_ids)
                    if item_bin == bin_index
                ]
                strict = np.asarray(
                    [item["dropout_rate_2s"] for item in selected], dtype=np.float64
                )
                adjusted = np.asarray(
                    [item["dropout_rate_adjusted"] for item in selected], dtype=np.float64
                )
                aggregate_rows.append(
                    {
                        "terminal_id": terminal_id,
                        "rain_rate_bin": label,
                        "sample_count": len(selected),
                        "mean_dropout_2s": round(float(strict.mean()), 6) if len(strict) else None,
                        "median_dropout_2s": round(float(np.median(strict)), 6) if len(strict) else None,
                        "p90_dropout_2s": round(float(np.quantile(strict, 0.9)), 6) if len(strict) else None,
                        "mean_dropout_adjusted": (
                            round(float(adjusted.mean()), 6) if len(adjusted) else None
                        ),
                        "median_dropout_adjusted": (
                            round(float(np.median(adjusted)), 6) if len(adjusted) else None
                        ),
                        "p90_dropout_adjusted": (
                            round(float(np.quantile(adjusted, 0.9)), 6) if len(adjusted) else None
                        ),
                    }
                )

        result = {
            "status": "ok",
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "session_gap_s": session_gap_s,
            "rain_rate_bins": RAIN_RATE_LABELS,
            "generated_at": now.isoformat(timespec="seconds"),
            "terminals": [
                {
                    "terminal_id": terminal_id,
                    "terminal_name": TERMINAL_NAMES[terminal_id],
                    "nominal_interval_s": TERMINAL_NOMINAL_INTERVAL_S[terminal_id],
                    "session_count": sum(
                        item["terminal_id"] == terminal_id for item in sessions
                    ),
                }
                for terminal_id in self.builders
            ],
            "rows": aggregate_rows,
            "rainy_sessions": [item for item in sessions if item["rain_rate_mm_h"] > 0],
            "interpretation_note": (
                "断链率按观测到的首末链路点构成的窗口计算，是缺少独立星历可见窗口时的下界。"
            ),
        }
        if start is None and end is None:
            self._dropout_cache = (session_gap_s, now, copy.deepcopy(result))
        return result

    def link_reliability_analysis(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        summary_path = self.link_analysis_dir / "dashboard_summary.json"
        pass_path = self.link_analysis_dir / "pass_diagnostics.csv"
        gap_path = self.link_analysis_dir / "inter_satellite_gaps.csv"
        if not summary_path.exists() or not pass_path.exists() or not gap_path.exists():
            return {
                "status": "not_generated",
                "message": "请先运行 Stage1/link_reliability_analysis/analyze_raw_link_reliability.py",
            }
        if self._link_analysis_summary is None:
            self._link_analysis_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if self._link_analysis_passes is None:
            self._link_analysis_passes = pd.read_csv(
                pass_path, parse_dates=["pass_start", "pass_end"]
            )
        if self._link_analysis_gaps is None:
            self._link_analysis_gaps = pd.read_csv(
                gap_path,
                parse_dates=[
                    "previous_session_end", "next_session_start",
                    "coverage_boundary_end", "network_gap_start", "network_gap_end",
                ],
            )
        rows = self._link_analysis_passes
        selected = rows
        if start is not None:
            selected = selected[selected["pass_start"] >= pd.Timestamp(start)]
        if end is not None:
            selected = selected[selected["pass_start"] < pd.Timestamp(end)]

        rain_rows = selected.dropna(subset=["rain_rate_mm_h"]).copy()
        rain_rows["rain_rate_bin"] = pd.cut(
            rain_rows["rain_rate_mm_h"],
            LINK_ANALYSIS_RAIN_EDGES,
            labels=LINK_ANALYSIS_RAIN_LABELS,
            include_lowest=True,
            right=False,
        )
        rain_summary = (
            rain_rows.groupby("rain_rate_bin", observed=False)
            .agg(
                sample_count=("satellite_id", "size"),
                mean_dropout_2s=("dropout_rate", "mean"),
                mean_dropout_empirical=("dropout_rate_empirical", "mean"),
                median_dropout_empirical=("dropout_rate_empirical", "median"),
                p90_dropout_empirical=("dropout_rate_empirical", lambda values: values.quantile(0.9)),
                mean_snr=("mean_snr", "mean"),
                mean_dropout_excess=("dropout_excess_vs_sat_month_dry", "mean"),
            )
            .reset_index()
        )
        causes = (
            selected["diagnostic_cause"].value_counts().rename_axis("cause")
            .rename("sessions").reset_index()
        )
        condition_frames = []
        condition_specs = [
            (
                "hour",
                pd.cut(
                    selected["pass_start"].dt.hour,
                    [-1, 5, 11, 17, 23],
                    labels=["00-05", "06-11", "12-17", "18-23"],
                ),
            ),
            (
                "max_elevation_deg",
                pd.cut(
                    selected["max_elevation_deg"],
                    [-np.inf, 10, 20, 40, 60, np.inf],
                    labels=["<10", "10-20", "20-40", "40-60", ">=60"],
                ),
            ),
        ]
        for dimension, categories in condition_specs:
            condition = selected.assign(_category=categories).groupby(
                "_category", observed=False
            ).agg(
                sample_count=("satellite_id", "size"),
                mean_dropout_empirical=("dropout_rate_empirical", "mean"),
                mean_snr=("mean_snr", "mean"),
            ).reset_index(names="category")
            condition.insert(0, "dimension", dimension)
            condition_frames.append(condition)
        available_months = sorted(rows["month"].dropna().astype(str).unique())
        month_start = pd.Timestamp(start).strftime("%Y-%m") if start is not None else None
        month_end = pd.Timestamp(end - timedelta(microseconds=1)).strftime("%Y-%m") if end is not None else None
        monthly = [
            item for item in self._link_analysis_summary["monthly"]
            if (month_start is None or item["month"] >= month_start)
            and (month_end is None or item["month"] <= month_end)
        ]
        quality = [
            item for item in self._link_analysis_summary["quality"]
            if (month_start is None or str(item.get("month")) >= month_start)
            and (month_end is None or str(item.get("month")) <= month_end)
        ]
        quality_by_rain = [
            item for item in self._link_analysis_summary.get("quality_by_rain", [])
            if (month_start is None or str(item.get("month")) >= month_start)
            and (month_end is None or str(item.get("month")) <= month_end)
        ]
        selected_gaps = self._link_analysis_gaps
        if start is not None:
            selected_gaps = selected_gaps[
                selected_gaps.next_session_start >= pd.Timestamp(start)
            ]
        if end is not None:
            selected_gaps = selected_gaps[
                selected_gaps.next_session_start < pd.Timestamp(end)
            ]
        largest_gaps = selected_gaps.loc[selected_gaps.network_gap_s > 0].nlargest(
            50, "network_gap_s"
        )
        positive_gaps = selected_gaps.loc[selected_gaps.network_gap_s > 0, "network_gap_s"]
        continuous_count = int(selected_gaps.network_gap_s.eq(0).sum())
        gap_summary = {
            "transition_count": int(len(selected_gaps)),
            "continuous_handover_count": continuous_count,
            "continuous_handover_rate": (
                float(continuous_count / len(selected_gaps))
                if len(selected_gaps) else None
            ),
            "positive_gap_count": int(len(positive_gaps)),
            "median_positive_gap_s": (
                float(positive_gaps.median()) if len(positive_gaps) else None
            ),
        }

        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            return json.loads(
                frame.replace({np.nan: None}).to_json(
                    orient="records", date_format="iso"
                )
            )

        return {
            "status": "ok",
            "generated_at": self._link_analysis_summary["generated_at"],
            "query_start": pd.Timestamp(start).isoformat() if start else None,
            "query_end": pd.Timestamp(end).isoformat() if end else None,
            "available_months": available_months,
            "session_count": int(len(selected)),
            "rain_aligned_session_count": int(len(rain_rows)),
            "monthly": monthly,
            "rain_rate": records(rain_summary),
            "quality": quality,
            "quality_by_rain": quality_by_rain,
            "gap_summary": gap_summary,
            "largest_gaps": records(largest_gaps),
            "causes": records(causes),
            "conditions": records(pd.concat(condition_frames, ignore_index=True)),
            "method": self._link_analysis_summary["method"],
            "provenance": self._link_analysis_summary["provenance"],
            "interpretation_note": (
                "严格2秒缺样率按业务标称周期计算；校正缺样率使用各月同星会话实测中位周期。"
                "无缝接续按会话区间的全网空窗为0统计；会话内部允许最多60秒间隔，"
                "因此不等同于原始PHY每2秒连续无中断。位置可见不等同于被调度通信，"
                "疑似干扰与雨相关均为统计诊断。"
            ),
        }

    def _position_retrieval_matches(
        self, position_rows: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        """Match native 001 minute predictions to same-satellite position passes."""
        now = datetime.now()
        if self._position_retrieval_cache is not None:
            cached_at, cached_rows, eligible_count = self._position_retrieval_cache
            if now - cached_at < timedelta(minutes=5):
                return cached_rows, eligible_count

        query = """
            SELECT satellite_id, pass_start, pass_end,
                   observed_rainfall_mm, reported_rainfall_mm
            FROM rain_retrieval_passes
            WHERE terminal_id = ?
              AND observed_available = 1
              AND transfer_mode LIKE '%full_position'
            ORDER BY satellite_id, pass_start
        """
        with sqlite3.connect(
            f"file:{self.history.db_path}?mode=ro", uri=True, timeout=30.0
        ) as conn:
            history = pd.read_sql_query(
                query,
                conn,
                params=["01-31-0005-0001"],
                parse_dates=["pass_start", "pass_end"],
            )
        eligible_count = int(len(history))
        merged_frames: list[pd.DataFrame] = []
        geometry_columns = [
            "identity_key", "max_elevation_deg", "mean_slant_range_km"
        ]
        for satellite_id, minute_rows in history.groupby("satellite_id", sort=False):
            passes = position_rows.loc[
                position_rows.satellite_id.eq(satellite_id)
            ].sort_values("pass_start").reset_index(drop=True)
            if passes.empty:
                continue
            starts = passes.pass_start.to_numpy(dtype="datetime64[ns]")
            ends = passes.pass_end.to_numpy(dtype="datetime64[ns]")
            minute_starts = minute_rows.pass_start.to_numpy(dtype="datetime64[ns]")
            minute_ends = minute_rows.pass_end.to_numpy(dtype="datetime64[ns]")
            pass_indices = np.searchsorted(starts, minute_ends, side="right") - 1
            valid = pass_indices >= 0
            valid_indices = np.flatnonzero(valid)
            valid[valid_indices] &= ends[pass_indices[valid_indices]] >= minute_starts[valid_indices]
            if not valid.any():
                continue
            matched_minutes = minute_rows.iloc[np.flatnonzero(valid)].reset_index(drop=True)
            matched_geometry = passes.iloc[pass_indices[valid]][geometry_columns].reset_index(drop=True)
            merged_frames.append(pd.concat([matched_minutes, matched_geometry], axis=1))

        columns = list(history.columns) + geometry_columns
        matches = (
            pd.concat(merged_frames, ignore_index=True)
            if merged_frames
            else pd.DataFrame(columns=columns)
        )
        self._position_retrieval_cache = (now, matches, eligible_count)
        return matches, eligible_count

    @staticmethod
    def _summarize_position_retrieval(
        rows: pd.DataFrame, eligible_count: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def finite(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        def metrics(frame: pd.DataFrame) -> dict[str, Any]:
            if frame.empty:
                return {
                    "sample_count": 0, "rainy_sample_count": 0,
                    "mae_mm": None, "rainy_mae_mm": None,
                    "bias_mm": None, "correlation": None,
                }
            observed = frame.observed_rainfall_mm.astype(float)
            predicted = frame.reported_rainfall_mm.astype(float)
            rainy = observed > 0
            correlation = (
                observed.corr(predicted)
                if len(frame) >= 3 and observed.std() > 0 and predicted.std() > 0
                else np.nan
            )
            return {
                "sample_count": int(len(frame)),
                "rainy_sample_count": int(rainy.sum()),
                "mae_mm": finite((predicted - observed).abs().mean()),
                "rainy_mae_mm": finite(
                    (predicted[rainy] - observed[rainy]).abs().mean()
                ) if rainy.any() else None,
                "bias_mm": finite((predicted - observed).mean()),
                "correlation": finite(correlation),
            }

        overview = metrics(rows)
        overview["eligible_sample_count"] = int(eligible_count)
        overview["geometry_match_rate"] = (
            float(len(rows) / eligible_count) if eligible_count else None
        )
        specifications = [
            (
                "elevation_deg", "max_elevation_deg",
                [0, 5, 10, 20, 30, 45, 60, 90.000001],
                ["[0, 5)", "[5, 10)", "[10, 20)", "[20, 30)",
                 "[30, 45)", "[45, 60)", "[60, 90]"],
            ),
            (
                "slant_range_km", "mean_slant_range_km",
                [0, 1000, 1500, 2000, 3000, float("inf")],
                ["[0, 1000)", "[1000, 1500)", "[1500, 2000)",
                 "[2000, 3000)", ">=3000"],
            ),
        ]
        summaries: list[dict[str, Any]] = []
        for dimension, column, bins, labels in specifications:
            categories = pd.cut(
                pd.to_numeric(rows[column], errors="coerce"),
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            )
            for label in labels:
                item = metrics(rows.loc[categories.eq(label)])
                item.update({"dimension": dimension, "category": label})
                summaries.append(item)
        return overview, summaries

    def position_link_analysis(self, identity_key: str | None = None) -> dict[str, Any]:
        artifact_dirs = sorted(self.link_analysis_dir.glob("position_link_*"))
        if not artifact_dirs:
            return {"status": "not_generated", "message": "星地位置关联统计尚未生成"}
        artifact_dir = artifact_dirs[-1]
        dashboard_path = artifact_dir / "position_link_dashboard.json"
        passes_path = artifact_dir / "position_link_passes.csv"
        if not dashboard_path.exists() or not passes_path.exists():
            return {"status": "not_generated", "message": "星地位置关联统计文件不完整"}
        if self._position_link_dashboard is None:
            self._position_link_dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
        if self._position_link_passes is None:
            self._position_link_passes = pd.read_csv(
                passes_path, parse_dates=["pass_start", "pass_end"]
            )
        rows = self._position_link_passes
        selected = rows.loc[rows.identity_key.eq(identity_key)] if identity_key else rows
        retrieval_rows, eligible_retrieval_count = self._position_retrieval_matches(rows)
        if identity_key:
            retrieval_rows = retrieval_rows.loc[
                retrieval_rows.identity_key.eq(identity_key)
            ]
            # The unmatched denominator cannot be assigned to an identity key.
            eligible_retrieval_count = 0
        retrieval_overview, retrieval_summary = self._summarize_position_retrieval(
            retrieval_rows, eligible_retrieval_count
        )
        maximum = 1800 if identity_key else 2500
        if len(selected) > maximum:
            indices = np.linspace(0, len(selected) - 1, maximum, dtype=int)
            plot_rows = selected.iloc[indices]
        else:
            plot_rows = selected

        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            return json.loads(frame.replace({np.nan: None}).to_json(
                orient="records", date_format="iso"
            ))

        columns = [
            "identity_key", "latest_let_id", "physical_norad_id", "physical_name",
            "mapping_status", "pass_count", "position_matched_passes",
            "mean_dropout_rate", "median_dropout_rate", "p90_dropout_rate",
            "mean_rssi", "mean_phy_rssi", "mean_snr",
            "median_max_elevation_deg", "median_slant_range_km",
            "median_altitude_km",
        ]
        satellites = pd.DataFrame(self._position_link_dashboard["satellites"])
        satellite_columns = [column for column in columns if column in satellites]
        detail_columns = [
            "identity_key", "pass_start", "pass_end", "actual_phy_points",
            "expected_phy_points", "dropout_rate", "mean_rssi", "mean_snr",
            "max_elevation_deg", "mean_slant_range_km", "longitude_deg",
            "latitude_deg", "altitude_km", "ecef_x_km", "ecef_y_km", "ecef_z_km",
        ]
        geometry_summary = sorted(
            self._position_link_dashboard["geometry_summary"],
            key=lambda row: (
                row["dimension"],
                float(re.search(r"-?\d+(?:\.\d+)?", row["category"]).group()),
            ),
        )
        return {
            "status": "ok", "generated_at": self._position_link_dashboard["generated_at"],
            "selected_identity_key": identity_key, "overview": self._position_link_dashboard["overview"],
            "method": self._position_link_dashboard["method"],
            "provenance": self._position_link_dashboard["provenance"],
            "monthly": self._position_link_dashboard.get("monthly", []),
            "geometry_summary": geometry_summary,
            "retrieval_geometry_overview": retrieval_overview,
            "retrieval_geometry_summary": retrieval_summary,
            "phy_quality": self._position_link_dashboard.get("phy_quality", []),
            "position_quality": self._position_link_dashboard.get("position_quality", []),
            "satellites": records(satellites[satellite_columns]),
            "plot_passes": records(plot_rows[detail_columns]),
            "selected_passes": records(selected[detail_columns].tail(200)),
        }

    def timeline(
        self,
        start: datetime,
        end: datetime,
        resolution_minutes: int,
    ) -> dict[str, Any]:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        station = load_weather_station(
            str(next(iter(self.runtimes.values())).db_path),
            start_time=(start_ts - pd.Timedelta(minutes=5)).isoformat(),
            end_time=(end_ts + pd.Timedelta(minutes=5)).isoformat(),
        )
        station_window = station.loc[start_ts:end_ts]
        rain = []
        cumulative_values = pd.to_numeric(
            station_window.get("rainfall_cumulative"), errors="coerce"
        )
        cumulative_base = (
            float(cumulative_values.dropna().iloc[0])
            if cumulative_values is not None and cumulative_values.notna().any()
            else None
        )
        for timestamp, row in station_window.iterrows():
            rainfall = pd.to_numeric(
                pd.Series([row.get("rainfall")]), errors="coerce"
            ).iloc[0]
            cumulative = pd.to_numeric(
                pd.Series([row.get("rainfall_cumulative")]), errors="coerce"
            ).iloc[0]
            rain.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "rainfall": (
                        round(float(rainfall), 6)
                        if pd.notna(rainfall)
                        else None
                    ),
                    "rainfall_cumulative": (
                        round(float(cumulative), 6)
                        if pd.notna(cumulative)
                        else None
                    ),
                    "rainfall_cumulative_delta": (
                        round(max(float(cumulative) - cumulative_base, 0.0), 6)
                        if pd.notna(cumulative) and cumulative_base is not None
                        else None
                    ),
                }
            )

        terminals = []
        all_passes = []
        density: dict[str, list[list[Any]]] = {}
        strong_snr_density: dict[str, list[list[Any]]] = {}
        mean_snr_series: dict[str, list[list[Any]]] = {}
        history_rows = self.history.query_range(
            start.isoformat(),
            end.isoformat(),
            limit=10000,
        )
        history_by_terminal_sat: dict[
            tuple[str, int], list[dict[str, Any]]
        ] = {}
        for row in history_rows:
            key = (str(row["terminal_id"]), int(row["satellite_id"]))
            history_by_terminal_sat.setdefault(key, []).append(row)

        for terminal_id, builder in self.builders.items():
            runtime = self.runtimes[terminal_id]
            link = self._read_terminal_link(terminal_id, start_ts, end_ts)
            passes = self._timeline_passes(
                link,
                terminal_id,
                runtime.pass_gap_threshold_s,
                runtime.min_pass_points,
            )
            for pass_row in passes:
                labels, metadata = compute_pass_labels(
                    station,
                    pd.Timestamp(pass_row["pass_start"]),
                    pd.Timestamp(pass_row["pass_end"]),
                )
                pass_row["observed_rainfall_mm"] = (
                    round(float(labels[0]), 6) if labels is not None else None
                )
                pass_row["observed_reason"] = (
                    None
                    if labels is not None
                    else metadata.get("drop_reason", "unavailable")
                )
                candidates = history_by_terminal_sat.get(
                    (terminal_id, int(pass_row["satellite_id"])), []
                )
                pass_start = pd.Timestamp(pass_row["pass_start"])
                nearest = min(
                    candidates,
                    key=lambda item: abs(
                        (
                            pd.Timestamp(item["pass_start"]) - pass_start
                        ).total_seconds()
                    ),
                    default=None,
                )
                if nearest is not None and abs(
                    (
                        pd.Timestamp(nearest["pass_start"]) - pass_start
                    ).total_seconds()
                ) <= 2.0:
                    pass_row["reported_rainfall_mm"] = nearest.get(
                        "reported_rainfall_mm"
                    )
                    pass_row["rain_probability"] = nearest.get(
                        "rain_probability"
                    )
                else:
                    pass_row["reported_rainfall_mm"] = None
                    pass_row["rain_probability"] = None
            all_passes.extend(passes)

            if link.empty:
                last_in_window = None
            else:
                last_in_window = link.index[-1]
            frequency = f"{resolution_minutes}min"
            full_index = pd.date_range(
                start=start_ts.floor(frequency),
                end=end_ts.ceil(frequency),
                freq=frequency,
                inclusive="left",
            )
            counts = (
                link.resample(frequency).size()
                if not link.empty
                else pd.Series(dtype="int64")
            )
            counts = counts.reindex(full_index, fill_value=0)
            density[terminal_id] = [
                [timestamp.isoformat(), int(value)]
                for timestamp, value in counts.items()
            ]
            if link.empty:
                strong_counts = pd.Series(dtype="int64")
                mean_snr = pd.Series(dtype="float64")
            else:
                link_snr = pd.to_numeric(link["snr"], errors="coerce")
                strong_counts = link_snr.ge(
                    DISPLAY_STRONG_SNR_THRESHOLD_DB
                ).resample(frequency).sum()
                mean_snr = link_snr.resample(frequency).mean()
            strong_counts = strong_counts.reindex(full_index, fill_value=0)
            mean_snr = mean_snr.reindex(full_index)
            strong_snr_density[terminal_id] = [
                [timestamp.isoformat(), int(value)]
                for timestamp, value in strong_counts.items()
            ]
            mean_snr_series[terminal_id] = [
                [timestamp.isoformat(), round(float(value), 4) if pd.notna(value) else None]
                for timestamp, value in mean_snr.items()
            ]
            latest = builder.latest_link_time()
            is_active = bool(
                latest is not None
                and (pd.Timestamp.now() - latest).total_seconds()
                <= max(runtime.pass_gap_threshold_s, self.poll_interval_s * 2)
            )
            active_satellite = None
            if is_active:
                recent_link = self._read_terminal_link(
                    terminal_id,
                    latest - pd.Timedelta(
                        seconds=runtime.pass_gap_threshold_s
                    ),
                    latest + pd.Timedelta(seconds=1),
                )
                if not recent_link.empty:
                    active_satellite = int(
                        recent_link.iloc[-1]["satelliteId"]
                    )
            terminals.append(
                {
                    "terminal_id": terminal_id,
                    "terminal_name": TERMINAL_NAMES[terminal_id],
                    "active": is_active,
                    "active_satellite_id": active_satellite,
                    "latest_link_time": (
                        latest.isoformat() if latest is not None else None
                    ),
                    "last_link_in_window": (
                        last_in_window.isoformat()
                        if last_in_window is not None
                        else None
                    ),
                    "pass_count": len(passes),
                    "link_points": int(len(link)),
                }
            )

        model_series, consistency = self._model_time_series(
            all_passes, start_ts, end_ts, resolution_minutes
        )
        consistency_groups, consistency_group_summary = self._pass_consistency_groups(
            all_passes, station
        )

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "resolution_minutes": resolution_minutes,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "terminals": terminals,
            "passes": all_passes,
            "link_density": density,
            "strong_snr_density": strong_snr_density,
            "mean_snr_series": mean_snr_series,
            "display_snr_threshold_db": DISPLAY_STRONG_SNR_THRESHOLD_DB,
            "rain": rain,
            "model_series": model_series,
            "consistency": consistency,
            "consistency_groups": consistency_groups,
            "consistency_group_summary": consistency_group_summary,
            "interpretation_note": (
                "空白表示未观测到有效链路；模型累计量仅积分卫星覆盖时段，不代表无过境时段的降雨。"
            ),
        }

    def latest_data_date(self) -> date | None:
        latest = [
            builder.latest_link_time()
            for builder in self.builders.values()
        ]
        valid = [value for value in latest if value is not None]
        return max(valid).date() if valid else None

    @staticmethod
    def summarize(result: dict[str, Any]) -> str:
        lines = [f"{result['query_date']} 三终端 Stage1 降雨反演："]
        for terminal in result["terminals"]:
            summary = terminal["summary"]
            if summary["pass_count"] == 0:
                latest = terminal["latest_link_time"] or "无"
                lines.append(
                    f"{terminal['terminal_name']}：当日无有效链路过境；"
                    f"最后链路数据为 {latest}。"
                )
            else:
                lines.append(
                    f"{terminal['terminal_name']}：{summary['pass_count']} 次过境，"
                    f"{summary['rainy_pass_count']} 次达到有雨阈值，"
                    f"最大反演雨量 {summary['max_reported_rainfall_mm']:.3f} mm，"
                    f"过境均值 {summary['mean_reported_rainfall_mm']:.3f} mm；"
                    f"雨量计有效记录 {summary['observed_pass_count']} 条，"
                    f"其中有雨 {summary['observed_rainy_pass_count']} 条，"
                    + (
                        f"过境 MAE {summary['mae_mm']:.3f} mm。"
                        if summary["mae_mm"] is not None
                        else "暂无可计算误差的雨量计记录。"
                    )
                )
        return "\n".join(lines)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>三终端降雨反演</title>
  <style>
    :root{--ink:#18201d;--muted:#66716c;--line:#d7ddd9;--paper:#f4f6f3;--panel:#fff;--green:#176b4b;--rain:#1769aa;--warn:#a34b19}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;letter-spacing:0}
    header{background:#173d31;color:#fff;padding:22px max(24px,calc((100vw - 1440px)/2));display:flex;align-items:end;justify-content:space-between;gap:24px}header>div:first-child{flex:0 0 auto}
    h1{font-size:25px;margin:0 0 3px;font-weight:650;white-space:nowrap}header p{margin:0;color:#c9d9d2}.model{min-width:0;max-width:70vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,monospace;font-size:12px;color:#c9d9d2;text-align:right}
    main{max-width:1680px;margin:auto;padding:20px 24px 40px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
    input,select,button{height:38px;border:1px solid #b8c1bc;background:#fff;color:var(--ink);padding:0 12px;font:inherit;border-radius:5px}
    input[type=date]{min-width:160px}button{cursor:pointer}button.primary{background:var(--green);border-color:var(--green);color:#fff}button:hover{filter:brightness(.96)}
    .date-picker{position:relative}.date-picker input{width:160px;cursor:pointer}.calendar{display:none;position:absolute;z-index:20;top:43px;left:0;width:294px;padding:12px;background:#fff;border:1px solid var(--line);border-radius:7px;box-shadow:0 12px 30px rgba(22,42,34,.16)}.calendar.open{display:block}.calendar-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}.calendar-head strong{font-size:15px}.calendar-head button{width:34px;height:32px;padding:0;font-size:18px}.calendar-week,.calendar-days{display:grid;grid-template-columns:repeat(7,1fr);gap:3px}.calendar-week span{text-align:center;color:var(--muted);font-size:11px;padding:4px 0}.calendar-day{height:34px;padding:0;border-color:transparent;background:#fff;position:relative}.calendar-day.muted{color:#aeb6b1}.calendar-day.rainy{background:#e2f1ff;color:#145f9b;font-weight:650}.calendar-day.rainy::after{content:"";position:absolute;width:4px;height:4px;border-radius:50%;background:#1769aa;left:calc(50% - 2px);bottom:3px}.calendar-day.selected{outline:2px solid var(--green);outline-offset:-2px}.calendar-legend{margin-top:8px;color:var(--muted);font-size:11px}.calendar-legend i{display:inline-block;width:10px;height:10px;background:#e2f1ff;border:1px solid #86bce5;margin-right:5px;vertical-align:-1px}
    #notice{margin-left:auto;color:var(--muted)}.terminal-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
    .timeline-panel{background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:0 0 18px;padding:15px 16px 8px}.timeline-heading{display:flex;justify-content:space-between;align-items:start;gap:16px;margin-bottom:10px}.timeline-heading h2{font-size:17px}.timeline-heading p,.timeline-note{margin:2px 0 0;color:var(--muted);font-size:12px}.timeline-status{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:8px}.timeline-status-item{border-left:3px solid #9aa69f;background:#f6f8f6;padding:8px 10px;min-width:0}.timeline-status-item.active{border-left-color:var(--green)}.timeline-status-item strong{display:block}.timeline-status-item span{display:block;color:var(--muted);font-size:12px;overflow-wrap:anywhere}.consistency{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 4px}.consistency-item{background:#edf3ef;border-left:3px solid #496c5d;padding:7px 10px}.consistency-item strong{display:block;font-size:15px}.consistency-item span{color:var(--muted);font-size:11px}.timeline-chart{height:780px;width:100%}.timeline-note{padding:0 80px 6px}
    .reliability-panel{background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:0 0 18px;padding:15px 16px 12px}.reliability-chart{height:1100px;width:100%}.reliability-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.reliability-meta{color:var(--muted);font-size:12px;margin:2px 0 0}.reliability-table{margin-top:8px;max-height:280px;overflow:auto;border:1px solid var(--line)}.reliability-table th{z-index:1}.provenance{margin-top:10px;padding:9px 11px;background:#f6f8f6;color:var(--muted);font-size:11px;overflow-wrap:anywhere}.cause-item{background:#f2f5f3;border-left:3px solid #728a7f;padding:7px 10px}.cause-item strong{display:block;font-size:15px}.cause-item span{font-size:11px;color:var(--muted)}
    .position-chart{height:880px;width:100%}.orbit-chart{height:470px;width:100%}.position-accuracy-chart{height:430px;width:100%}.position-table-grid{display:grid;grid-template-columns:minmax(620px,1.25fr) minmax(500px,1fr);gap:12px;margin-top:10px}.position-table-grid .reliability-table{margin:0;max-height:340px}.position-controls select{min-width:310px}
    .accuracy-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.accuracy-chart{height:360px;min-width:0;border-left:1px solid var(--line)}.accuracy-chart:first-child{border-left:0}
    .terminal{background:var(--panel);border:1px solid var(--line);border-radius:7px;min-width:0}.terminal-head{padding:15px 16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:10px}
    h2{font-size:17px;margin:0}.terminal-id{font:12px ui-monospace,monospace;color:var(--muted)}.status{font-size:12px;color:var(--green)}.status.empty{color:var(--warn)}
    .metrics{display:grid;grid-template-columns:repeat(3,1fr);border-bottom:1px solid var(--line)}.metric{padding:12px 14px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.metric:nth-child(3n){border-right:0}.metric:nth-last-child(-n+3){border-bottom:0}
    .metric strong{display:block;font-size:21px;font-weight:650}.metric span,.freshness{font-size:12px;color:var(--muted)}.freshness{padding:10px 16px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
    .table-wrap{overflow:auto;max-height:440px}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{padding:9px 10px;text-align:left;border-bottom:1px solid #edf0ee}th{position:sticky;top:0;background:#f8faf8;color:var(--muted);font-weight:600}td.rain{color:var(--rain);font-weight:650}
    .empty-state{padding:36px 16px;text-align:center;color:var(--muted)}footer{margin-top:14px;color:var(--muted);font-size:12px}
    @media(max-width:1050px){.terminal-grid,.accuracy-grid,.position-table-grid{grid-template-columns:1fr}.accuracy-chart{border-left:0;border-top:1px solid var(--line)}}@media(max-width:620px){header{align-items:start;flex-direction:column}.model{max-width:100%;text-align:left}main{padding:14px 10px}.toolbar>*{flex:1 1 auto}#notice{flex-basis:100%}.timeline-panel,.reliability-panel{padding:12px 6px}.timeline-heading{padding:0 6px}.timeline-status{grid-template-columns:1fr;padding:0 6px}.consistency{padding:0 6px}.timeline-chart{height:850px}.reliability-chart{height:1240px}.position-chart{height:1200px}.orbit-chart{height:720px}.position-accuracy-chart{height:680px}.reliability-controls{padding:0 6px}.position-controls select{min-width:0;width:100%}}
  </style>
</head>
<body>
<header><div><h1>三终端降雨反演</h1><p>Stage1 小模型 · 分钟级对比</p></div><div class="model" id="model">正在加载模型信息</div></header>
<main>
  <div class="toolbar">
    <div class="date-picker"><input id="date" type="text" readonly aria-label="查询日期" aria-haspopup="dialog"><div class="calendar" id="calendar"><div class="calendar-head"><button id="calendar-prev" type="button" title="上个月">‹</button><strong id="calendar-title"></strong><button id="calendar-next" type="button" title="下个月">›</button></div><div class="calendar-week"><span>一</span><span>二</span><span>三</span><span>四</span><span>五</span><span>六</span><span>日</span></div><div class="calendar-days" id="calendar-days"></div><div class="calendar-legend"><i></i>蓝色表示雨量计记录到降雨</div></div></div>
    <button class="primary" id="query">查询</button>
    <button data-shift="0">今天</button><button data-shift="-1">昨天</button>
    <button id="latest">最近有数据日</button>
    <select id="demo-date" aria-label="典型日期"><option value="">典型日期</option><option value="2026-06-03">6月3日 · 分钟拟合高相关</option><option value="2026-06-04">6月4日 · 有雨判定较稳定</option><option value="2026-06-20">6月20日 · 中等降雨过程</option><option value="2026-06-22">6月22日 · 较强降雨且累计贴合</option><option value="2026-07-21">7月21日 · 后期较强降雨</option><option value="2026-08-10">8月10日 · 台风强降雨过程</option></select><span id="notice"></span>
  </div>
  <section class="timeline-panel">
    <div class="timeline-heading"><div><h2>卫星通信状态与降雨时间轴</h2><p>拖动底部滑块或滚轮缩放，四层图保持同一时间范围。</p></div><span class="status" id="timeline-updated"></span></div>
    <div class="timeline-status" id="timeline-status"></div>
    <div class="consistency" id="consistency"></div>
    <div class="timeline-chart" id="timeline-chart"></div>
    <p class="timeline-note" id="timeline-note"></p>
  </section>
  <section class="timeline-panel">
    <div class="timeline-heading"><div><h2>分钟降雨反演准确性</h2><p>横坐标为雨量计真实一分钟降雨量，纵坐标为模型反演一分钟降雨量；虚线表示理想预测。累计量接近时，逐分钟正负误差仍可能相互抵消。</p></div></div>
    <div class="accuracy-grid"><div class="accuracy-chart" id="accuracy-001"></div><div class="accuracy-chart" id="accuracy-002"></div><div class="accuracy-chart" id="accuracy-003"></div></div>
  </section>
  <section class="reliability-panel">
    <div class="timeline-heading"><div><h2>原始链路可靠性与长期趋势</h2><p class="reliability-meta" id="reliability-meta">正在读取工控机原始备份统计</p></div><div class="reliability-controls"><select id="reliability-month" aria-label="链路分析月份"><option value="">全部月份</option></select><button id="reliability-refresh">刷新分析</button></div></div>
    <div class="consistency" id="reliability-summary"></div>
    <div class="reliability-chart" id="reliability-chart"></div>
    <div class="consistency" id="reliability-causes"></div>
    <div class="reliability-table"><table><thead><tr><th>月份</th><th>通信会话</th><th>日均会话</th><th>全体空窗中位数</th><th>正空窗 P90</th><th>无缝接续次数</th><th>无缝接续率</th><th>链路时间覆盖率</th><th>&gt;1 h 空窗</th><th>最长空窗</th><th>同星重访</th><th>周期校正缺样</th></tr></thead><tbody id="reliability-table-body"></tbody></table></div>
    <div class="timeline-heading" style="margin-top:14px"><div><h2>最长无链路空窗</h2><p>按所选月份列出最长 50 个全网 PHY 空窗；前后卫星不同不影响统计。</p></div></div>
    <div class="reliability-table"><table><thead><tr><th>空窗开始</th><th>空窗结束</th><th>时长</th><th>边界卫星</th><th>下一卫星</th><th>同期平均雨强</th><th>诊断</th></tr></thead><tbody id="largest-gap-body"></tbody></table></div>
    <div class="provenance" id="reliability-provenance"></div>
    <p class="timeline-note" id="reliability-note"></p>
  </section>
  <section class="reliability-panel">
    <div class="timeline-heading"><div><h2>链路质量与星地相对位置</h2><p class="reliability-meta" id="position-meta">正在读取位置机会与 PHY 会话统计</p></div><div class="reliability-controls position-controls"><select id="position-satellite" aria-label="选择卫星"><option value="">全部卫星</option></select></div></div>
    <div class="consistency" id="position-summary"></div>
    <div class="position-chart" id="position-chart"></div>
    <div class="timeline-heading"><div><h2>轨道高度与过境几何时序</h2><p>按通信会话开始时间展示卫星高度、最大仰角和平均星地距离；选择单颗卫星后可查看连续变化。</p></div></div>
    <div class="orbit-chart" id="orbit-chart"></div>
    <div class="timeline-heading"><div><h2>星地位置与分钟反演误差</h2><p>仅统计 001 终端使用完整位置输入的分钟记录，并按同卫星 ID 和时间重叠关联到通信会话；002/003 的零样本迁移结果不计入。</p></div></div>
    <div class="consistency" id="position-accuracy-summary"></div>
    <div class="position-accuracy-chart" id="position-accuracy-chart"></div>
    <p class="timeline-note">该图是描述性分箱统计。各位置区间的真实有雨样本量不同，误差差异不能单独解释为位置对模型精度的因果影响。</p>
    <div class="timeline-heading"><div><h2>月度星地链路统计</h2><p>仅统计完成同卫星位置匹配的 PHY 通信会话。</p></div></div>
    <div class="reliability-table"><table><thead><tr><th>月份</th><th>PHY会话</th><th>物理卫星</th><th>平均断链率</th><th>RSSI</th><th>SNR</th><th>平均最大仰角</th><th>平均星地距离</th><th>平均高度</th></tr></thead><tbody id="position-monthly-body"></tbody></table></div>
    <div class="position-table-grid">
      <div><div class="timeline-heading"><div><h2>逐星统计</h2><p>物理 ID 仅展示严格映射成功的 NORAD ID。</p></div></div><div class="reliability-table"><table><thead><tr><th>当前LET ID</th><th>物理ID</th><th>卫星</th><th>会话</th><th>平均断链率</th><th>P90</th><th>RSSI</th><th>SNR</th><th>仰角</th><th>距离</th><th>高度</th></tr></thead><tbody id="position-satellite-body"></tbody></table></div></div>
      <div><div class="timeline-heading"><div><h2>所选卫星会话</h2><p>断链率按会话首末时间内每 2 秒一个期望点计算。</p></div></div><div class="reliability-table"><table><thead><tr><th>开始</th><th>实际/期望</th><th>断链率</th><th>仰角</th><th>距离</th><th>经纬度/高度</th><th>ECEF XYZ</th></tr></thead><tbody id="position-pass-body"></tbody></table></div></div>
    </div>
    <div class="provenance" id="position-provenance"></div>
    <p class="timeline-note">经纬度/海拔与 ECEF 是同一卫星位置的两种坐标表达；图中的断链率只针对已发生的 PHY 通信会话。位置可见但未调度通信单独统计，不解释为链路故障。</p>
  </section>
  <section class="terminal-grid" id="grid"></section>
  <footer>模型值与雨量计值均对应雨量计锚定的一分钟窗口；“—”表示该窗口缺少有效观测。002/003 为冻结 001 模型的零样本迁移结果。</footer>
</main>
<script src="/static/echarts.min.js"></script>
<script>
const $=s=>document.querySelector(s), pad=n=>String(n).padStart(2,"0");
const localDate=d=>`${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`;
$("#date").value=localDate(new Date());
let rainyDates=new Set(),calendarMonth=new Date(new Date().getFullYear(),new Date().getMonth(),1);
function renderCalendar(){
  const year=calendarMonth.getFullYear(),month=calendarMonth.getMonth(),selected=$("#date").value;
  $("#calendar-title").textContent=`${year} 年 ${month+1} 月`;
  const first=(new Date(year,month,1).getDay()+6)%7,start=new Date(year,month,1-first),days=[];
  for(let index=0;index<42;index++){const current=new Date(start);current.setDate(start.getDate()+index);const value=localDate(current),button=document.createElement("button");button.type="button";button.className=`calendar-day${current.getMonth()!==month?" muted":""}${rainyDates.has(value)?" rainy":""}${value===selected?" selected":""}`;button.textContent=current.getDate();button.title=rainyDates.has(value)?`${value} · 有雨`:value;button.onclick=()=>{$("#date").value=value;calendarMonth=new Date(current.getFullYear(),current.getMonth(),1);$("#calendar").classList.remove("open");load()};days.push(button)}
  $("#calendar-days").replaceChildren(...days);
}
$("#date").onclick=event=>{event.stopPropagation();const selected=new Date(`${$("#date").value}T00:00:00`);calendarMonth=new Date(selected.getFullYear(),selected.getMonth(),1);renderCalendar();$("#calendar").classList.toggle("open")};
$("#calendar-prev").onclick=event=>{event.stopPropagation();calendarMonth=new Date(calendarMonth.getFullYear(),calendarMonth.getMonth()-1,1);renderCalendar()};
$("#calendar-next").onclick=event=>{event.stopPropagation();calendarMonth=new Date(calendarMonth.getFullYear(),calendarMonth.getMonth()+1,1);renderCalendar()};
$("#calendar").onclick=event=>event.stopPropagation();document.addEventListener("click",()=>$("#calendar").classList.remove("open"));
fetch("/api/rainy-dates").then(response=>response.json()).then(data=>{rainyDates=new Set(data.dates.map(row=>row.query_date));renderCalendar()}).catch(error=>console.error("rainy dates failed",error));
document.querySelectorAll("[data-shift]").forEach(b=>b.onclick=()=>{const d=new Date();d.setDate(d.getDate()+Number(b.dataset.shift));$("#date").value=localDate(d);load()});
$("#demo-date").onchange=event=>{if(event.target.value){$("#date").value=event.target.value;load()}};
const fmtTime=s=>s?s.replace("T"," ").slice(0,19):"无";
const terminalColors={"01-31-0005-0001":"#237a57","01-31-0005-0002":"#2474a6","01-31-0005-0003":"#b5682d"};
let timelineChart=null,reliabilityChart=null,positionChart=null,orbitChart=null,positionAccuracyChart=null,accuracyCharts=[],loading=false;
function renderTimelineStatus(terminals,end){
  const todayStart=new Date();todayStart.setHours(0,0,0,0);const historical=Date.parse(end)<=todayStart.getTime();
  $("#timeline-status").innerHTML=terminals.map(t=>{const state=historical?(t.pass_count?"历史窗口":"无历史链路"):(t.active?"过境中":"无实时链路");return `<div class="timeline-status-item ${t.active?"active":""}" style="border-left-color:${t.active?terminalColors[t.terminal_id]:"#9aa69f"}"><strong>${t.terminal_name} · ${state}</strong><span>${t.active_satellite_id?`当前卫星 ${t.active_satellite_id} · `:""}窗口内 ${t.pass_count} 次过境 / ${t.link_points} 个链路点</span><span>最后链路：${fmtTime(t.latest_link_time)}</span></div>`}).join("");
}
function renderConsistency(d){
  const pairs=d.consistency.pairwise.filter(x=>x.matched_passes>0||x.overlap_bins>0);
  const items=[`<div class="consistency-item"><strong>${d.consistency.consensus_bins}</strong><span>至少两终端重叠分钟</span></div>`];
  if(d.consistency.mean_spread_mm_h!=null)items.push(`<div class="consistency-item"><strong>${Number(d.consistency.mean_spread_mm_h).toFixed(3)} mm/h</strong><span>重叠时段平均极差</span></div>`);
  pairs.forEach(x=>{const a=x.left_terminal_id.slice(-3),b=x.right_terminal_id.slice(-3);if(x.matched_passes>0)items.push(`<div class="consistency-item"><strong>${Number(x.matched_mae_mm).toFixed(3)} mm</strong><span>${a}/${b} 同星过境 MAE · ${x.matched_passes} 对</span></div>`);else items.push(`<div class="consistency-item"><strong>${Number(x.mae_mm_h).toFixed(3)} mm/h</strong><span>${a}/${b} 同时段雨强 MAE · ${x.overlap_bins} min</span></div>`)});
  $("#consistency").innerHTML=items.join("");
}
function renderTimeline(d){
  renderTimelineStatus(d.terminals,d.end);
  $("#consistency").innerHTML="";
  $("#timeline-note").textContent=d.interpretation_note;
  if(!window.echarts){$("#timeline-chart").textContent="ECharts 静态资源加载失败";return}
  if(!timelineChart)timelineChart=echarts.init($("#timeline-chart"),null,{renderer:"canvas"});
  const laneIds=d.terminals.map(t=>t.terminal_id), laneNames=d.terminals.map(t=>t.terminal_name);
  const laneIndex=Object.fromEntries(laneIds.map((id,i)=>[id,i]));
  const passData=d.passes.map(p=>({value:[Date.parse(p.pass_start),Date.parse(p.pass_end),laneIndex[p.terminal_id],p.satellite_id,p.points,p.max_internal_gap_s,p.observed_rainfall_mm,p.reported_rainfall_mm,p.mean_rssi,p.mean_snr,p.terminal_id,p.strong_snr_points,p.strong_snr_ratio,p.rain_probability]}));
  const renderPass=(params,api)=>{
    const start=api.coord([api.value(0),api.value(2)]), end=api.coord([api.value(1),api.value(2)]);
    const height=Math.max(8,api.size([0,1])[1]*0.56);
    const rect=echarts.graphic.clipRectByRect({x:start[0],y:start[1]-height/2,width:Math.max(end[0]-start[0],2),height},{x:params.coordSys.x,y:params.coordSys.y,width:params.coordSys.width,height:params.coordSys.height});
    if(!rect)return null;const ratio=Number(api.value(12)||0),children=[{type:"rect",transition:["shape"],shape:rect,style:{fill:terminalColors[api.value(10)],opacity:.35+.65*ratio},emphasis:{style:{opacity:1,stroke:"#17221e",lineWidth:1}}}];
    if(rect.width>42)children.push({type:"text",silent:true,style:{x:rect.x+4,y:rect.y+rect.height/2,text:String(api.value(3)),fill:"#fff",font:"10px sans-serif",verticalAlign:"middle",overflow:"truncate",width:rect.width-8}});
    return {type:"group",children};
  };
  const densitySeries=laneIds.map((id,i)=>({name:`${laneNames[i]} 链路点`,type:"line",xAxisIndex:1,yAxisIndex:1,showSymbol:false,smooth:false,lineStyle:{width:1.5,color:terminalColors[id],opacity:.72},itemStyle:{color:terminalColors[id]},data:d.link_density[id]||[]}));
  const qualityHighlightSeries=laneIds.map((id,i)=>{const total=d.link_density[id]||[],strong=(d.strong_snr_density||{})[id]||[],strongByTime=Object.fromEntries(strong.map(row=>[row[0],Number(row[1]||0)]));return {name:`${laneNames[i]} 高质量分钟`,type:"scatter",xAxisIndex:1,yAxisIndex:1,silent:false,symbol:"circle",data:total.flatMap(row=>{const count=Number(row[1]||0),good=strongByTime[row[0]]||0,ratio=count?good/count:0;return good>0?[{value:[row[0],count],qualityRatio:ratio,strongPoints:good,totalPoints:count}]:[]}),symbolSize:(value,params)=>3+5*Math.sqrt(params.data.qualityRatio),itemStyle:{color:terminalColors[id],opacity:.35,borderColor:"#fff",borderWidth:.7},emphasis:{itemStyle:{opacity:1,borderColor:"#17221e",borderWidth:1}}}});
  const rainfall=d.rain.map(x=>[x.timestamp,x.rainfall]);
  const cumulative=d.rain.map(x=>[x.timestamp,x.rainfall_cumulative_delta]);
  const modelRateSeries=laneIds.map((id,i)=>({name:`${laneNames[i]} 模型分钟雨量`,type:"line",xAxisIndex:2,yAxisIndex:2,data:d.model_series[id].minute_amount_mm,showSymbol:false,connectNulls:false,lineStyle:{color:terminalColors[id],width:1.8},itemStyle:{color:terminalColors[id]}}));
  const probabilitySeries=laneIds.map((id,i)=>({name:`${laneNames[i]} 有雨概率`,type:"line",xAxisIndex:2,yAxisIndex:4,data:d.model_series[id].rain_probability||[],showSymbol:false,connectNulls:false,lineStyle:{color:terminalColors[id],width:1.1,type:"dotted",opacity:.75},itemStyle:{color:terminalColors[id]}}));
  const modelCumSeries=laneIds.map((id,i)=>({name:`${laneNames[i]} 模型累计`,type:"line",xAxisIndex:3,yAxisIndex:3,data:d.model_series[id].coverage_cumulative_mm,showSymbol:false,connectNulls:false,lineStyle:{color:terminalColors[id],width:1.5,type:"dashed"},itemStyle:{color:terminalColors[id]}}));
  const alignedCumulative=d.cumulative_comparison_mode==="model_valid_minutes";
  const observedCoverageSeries=alignedCumulative?laneIds.map((id,i)=>({name:`${laneNames[i]} 同窗真实累计`,type:"line",xAxisIndex:3,yAxisIndex:3,data:d.model_series[id].observed_coverage_cumulative_mm||[],showSymbol:false,connectNulls:false,lineStyle:{color:terminalColors[id],width:1.3,type:"dotted",opacity:.72},itemStyle:{color:terminalColors[id]}})):[];
  const cumulativeSeries=alignedCumulative?[...observedCoverageSeries,...modelCumSeries]:[{name:"雨量计累计",type:"line",xAxisIndex:3,yAxisIndex:3,data:cumulative,showSymbol:false,connectNulls:false,lineStyle:{color:"#164f78",width:2.6},areaStyle:{color:"rgba(75,145,194,.08)"}},...modelCumSeries];
  const cumulativeLegend=alignedCumulative?[...observedCoverageSeries.map(s=>s.name),...modelCumSeries.map(s=>s.name)]:["雨量计累计",...modelCumSeries.map(s=>s.name)];
  timelineChart.setOption({
    animation:false,
    color:Object.values(terminalColors),
    axisPointer:{link:[{xAxisIndex:[0,1,2,3]}],label:{backgroundColor:"#34413b"}},
    tooltip:{trigger:"item",confine:true,formatter:p=>{
      if(p.seriesType==="custom"){const v=p.data.value;return `<strong>${laneNames[v[2]]} · 卫星 ${v[3]}</strong><br>开始：${new Date(v[0]).toLocaleString()}<br>结束：${new Date(v[1]).toLocaleString()}<br>PHY点：${v[4]} · SNR≥${d.display_snr_threshold_db??-10}dB：${v[11]}（${(100*Number(v[12]||0)).toFixed(1)}%）<br>最大内部间隔：${Number(v[5]).toFixed(1)} s<br>平均 RSSI：${Number(v[8]).toFixed(2)} · 平均 SNR：${Number(v[9]).toFixed(2)} dB<br>雨量计过境雨量：${v[6]==null?"不可用":Number(v[6]).toFixed(3)+" mm"}<br>模型反演：${v[7]==null?"未入库":Number(v[7]).toFixed(3)+" mm"}<br>有雨概率：${v[13]==null?"不可用":(100*Number(v[13])).toFixed(1)+"%"}`}
      if(p.seriesName.includes("高质量分钟")){return `${p.seriesName}<br>${new Date(p.value[0]).toLocaleString()}<br>总PHY点：${p.data.totalPoints}<br>SNR≥${d.display_snr_threshold_db??-10}dB：${p.data.strongPoints}（${(100*p.data.qualityRatio).toFixed(1)}%）`}
      const value=Array.isArray(p.value)?p.value[1]:p.value,unit=p.seriesName.includes("概率")?"%":p.seriesName.includes("SNR")?" dB":p.seriesName.includes("PHY")?" 点":" mm";return `${p.seriesName}<br>${new Date(p.value[0]).toLocaleString()}<br>${value==null?"不可用":p.seriesName.includes("概率")?(100*Number(value)).toFixed(1)+unit:Number(value).toFixed(3)+unit}`;
    }},
    legend:[
      {top:174,right:6,data:densitySeries.map(s=>s.name),textStyle:{fontSize:11}},
      {top:342,right:6,data:["雨量计分钟雨量",...modelRateSeries.map(s=>s.name),...probabilitySeries.map(s=>s.name)],textStyle:{fontSize:11}},
      {top:537,right:6,data:cumulativeLegend,textStyle:{fontSize:11}}
    ],
    title:[
      {text:"观测到的卫星过境",left:4,top:8,textStyle:{fontSize:13,fontWeight:600}},
      {text:`每分钟链路点数（圆点高亮 SNR≥${d.display_snr_threshold_db??-10} dB）`,left:4,top:178,textStyle:{fontSize:13,fontWeight:600}},
      {text:"雨量计、模型一分钟降雨量与有雨概率",left:4,top:325,textStyle:{fontSize:13,fontWeight:600}},
      {text:alignedCumulative?"同有效分钟真实累计与模型累计":"真实累计与模型覆盖累计",left:4,top:520,textStyle:{fontSize:13,fontWeight:600}}
    ],
    grid:[
      {left:96,right:24,top:42,height:110},
      {left:96,right:24,top:210,height:80},
      {left:96,right:24,top:380,height:96},
      {left:96,right:24,top:575,height:96}
    ],
    xAxis:[0,1,2,3].map((_,i)=>({type:"time",gridIndex:i,min:d.start,max:d.end,axisLabel:{show:i===3,hideOverlap:true},axisTick:{show:i===3},axisLine:{show:i===3},splitLine:{show:true,lineStyle:{color:"#edf0ee"}}})),
    yAxis:[
      {type:"category",gridIndex:0,data:laneNames,inverse:true,axisTick:{show:false},axisLine:{show:false}},
      {type:"value",gridIndex:1,name:"点/min",min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},
      {type:"value",gridIndex:2,min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},
      {type:"value",gridIndex:3,min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},
      {type:"value",gridIndex:2,min:0,max:1,position:"right",axisLabel:{formatter:value=>`${Math.round(value*100)}%`},splitLine:{show:false}}
    ],
    dataZoom:[
      {type:"inside",xAxisIndex:[0,1,2,3],filterMode:"none"},
      {type:"slider",xAxisIndex:[0,1,2,3],filterMode:"none",bottom:12,height:22}
    ],
    series:[
      {name:"卫星过境",type:"custom",renderItem:renderPass,xAxisIndex:0,yAxisIndex:0,encode:{x:[0,1],y:2},data:passData},
      ...densitySeries,
      ...qualityHighlightSeries,
      {name:"雨量计分钟雨量",type:"bar",xAxisIndex:2,yAxisIndex:2,data:rainfall,barMaxWidth:5,itemStyle:{color:"#75a9ca",opacity:.65}},
      ...modelRateSeries,
      ...probabilitySeries,
      ...cumulativeSeries
    ]
  },true);
  $("#timeline-updated").textContent=`更新于 ${d.generated_at.replace("T"," ")}`;
}
function renderPassConsistency(d){
  const groups=(d.consistency_groups||[]).filter(group=>group&&Array.isArray(group.terminals)),summary=d.consistency_group_summary||{group_count:0,triple_group_count:0,observed_rainy_group_count:0,rain_decision_agreement:null,mean_rate_range_mm_h:null};
  const chips=[
    `<div class="consistency-item"><strong>${summary.group_count}</strong><span>重叠过境组</span></div>`,
    `<div class="consistency-item"><strong>${summary.triple_group_count}</strong><span>三终端共同组</span></div>`,
    `<div class="consistency-item"><strong>${summary.observed_rainy_group_count}</strong><span>共同窗口有雨</span></div>`,
    `<div class="consistency-item"><strong>${summary.rain_decision_agreement==null?"—":(summary.rain_decision_agreement*100).toFixed(1)+"%"}</strong><span>有雨/无雨判定一致率</span></div>`,
    `<div class="consistency-item"><strong>${summary.mean_rate_range_mm_h==null?"—":Number(summary.mean_rate_range_mm_h).toFixed(3)+" mm/h"}</strong><span>平均终端极差</span></div>`,
    ...(summary.best_rainy_group_start?[`<div class="consistency-item"><strong>${summary.best_rainy_group_start.slice(11,19)}</strong><span>当日最佳有雨一致组 · 极差 ${Number(summary.best_rainy_group_rate_range_mm_h).toFixed(3)} mm/h</span></div>`]:[])
  ];
  $("#pass-consistency-summary").innerHTML=chips.join("");
  if(!window.echarts)return;
  if(!passConsistencyChart)passConsistencyChart=echarts.init($("#pass-consistency-chart"),null,{renderer:"canvas"});
  if(!groups.length){passConsistencyChart.clear();passConsistencyChart.setOption({title:{text:"所选日期没有可配对的重叠过境",left:"center",top:"middle",textStyle:{fontSize:14,color:"#66716c",fontWeight:400}}});return}
  const categories=groups.map(group=>group.common_start.slice(11,16));
  const spreadData=groups.map((group,index)=>[index,group.rate_min_mm_h,group.rate_max_mm_h]);
  const renderSpread=(params,api)=>{const low=api.coord([api.value(0),api.value(1)]),high=api.coord([api.value(0),api.value(2)]);return{type:"group",children:[{type:"line",shape:{x1:low[0],y1:low[1],x2:high[0],y2:high[1]},style:{stroke:"#9aa69f",lineWidth:2}},{type:"line",shape:{x1:low[0]-4,y1:low[1],x2:low[0]+4,y2:low[1]},style:{stroke:"#9aa69f",lineWidth:2}},{type:"line",shape:{x1:high[0]-4,y1:high[1],x2:high[0]+4,y2:high[1]},style:{stroke:"#9aa69f",lineWidth:2}}]}};
  const terminalSeries=Object.keys(terminalColors).map(id=>({name:`终端 ${id.slice(-3)}`,type:"scatter",symbolSize:(value,params)=>params&&params.data&&params.data.group&&params.data.group.group_id===summary.best_rainy_group_id?14:9,itemStyle:{color:terminalColors[id]},data:groups.flatMap((group,index)=>{const row=group.terminals.find(item=>item&&item.terminal_id===id);return row?[{value:[index,row.rain_rate_mm_h],detail:row,group}]:[]})}));
  const gaugeSeries={name:"雨量计共同窗口",type:"scatter",symbol:"diamond",symbolSize:10,itemStyle:{color:"#111"},data:groups.flatMap((group,index)=>group.observed_overlap_rate_mm_h==null?[]:[{value:[index,group.observed_overlap_rate_mm_h],group}])};
  passConsistencyChart.clear();
  passConsistencyChart.setOption({animation:false,legend:{top:0,right:8,selected:{"雨量计共同窗口":false}},grid:{left:72,right:24,top:48,bottom:76},tooltip:{trigger:"item",confine:true,formatter:item=>{if(!item||item.seriesType==="custom"||!item.data)return "";const group=item.data.group;if(!group)return "";if(item.seriesName==="雨量计共同窗口")return `<strong>${group.common_start.replace("T"," ").slice(0,19)}</strong><br>共同窗口 ${Number(group.common_duration_s).toFixed(1)} s<br>雨量计：${Number(item.value[1]).toFixed(3)} mm/h`;const row=item.data.detail;if(!row)return "";return `<strong>${item.seriesName} · 卫星ID ${row.satellite_id}</strong><br>过境：${row.pass_start.slice(11,19)} 至 ${row.pass_end.slice(11,19)}<br>过境累计反演：${Number(row.rainfall_mm).toFixed(3)} mm<br>过境平均雨强：${Number(row.rain_rate_mm_h).toFixed(3)} mm/h<br>共同窗口：${group.common_start.slice(11,19)} 至 ${group.common_end.slice(11,19)}<br>终端极差：${Number(group.rate_range_mm_h).toFixed(3)} mm/h`}},xAxis:{type:"category",data:categories,name:"重叠过境开始时间",nameLocation:"middle",nameGap:48,axisLabel:{rotate:45,interval:Math.max(Math.ceil(groups.length/24)-1,0)}},yAxis:{type:"value",name:"过境平均雨强 / mm/h",min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},dataZoom:[{type:"inside",filterMode:"none"},{type:"slider",bottom:8,height:20,start:groups.length>30?0:0,end:groups.length>30?Math.min(100,3000/groups.length):100}],series:[{name:"终端极差",type:"custom",renderItem:renderSpread,silent:true,data:spreadData},...terminalSeries,gaugeSeries]},true);
}
function renderAccuracy(d){
  if(!window.echarts)return;
  d.terminals.forEach((terminal,index)=>{
    const element=document.getElementById(`accuracy-00${index+1}`);
    if(!element)return;
    if(!accuracyCharts[index])accuracyCharts[index]=echarts.init(element,null,{renderer:"canvas"});
    const rows=(terminal.predictions||[]).filter(row=>row.observed_available&&row.observed_rainfall_mm!=null);
    const full=[],fallback=[];
    rows.forEach(row=>{
      const item={value:[Number(row.observed_rainfall_mm),Number(row.reported_rainfall_mm)],detail:row};
      (String(row.transfer_mode||"").includes("fallback_no_position")?fallback:full).push(item);
    });
    const observed=rows.map(row=>Number(row.observed_rainfall_mm)),predicted=rows.map(row=>Number(row.reported_rainfall_mm));
    const mae=rows.length?rows.reduce((sum,row)=>sum+Math.abs(Number(row.reported_rainfall_mm)-Number(row.observed_rainfall_mm)),0)/rows.length:null;
    const rainy=rows.filter(row=>Number(row.observed_rainfall_mm)>0);
    const mean=values=>values.reduce((sum,value)=>sum+value,0)/values.length;
    const correlation=(left,right)=>{if(left.length<3)return null;const lm=mean(left),rm=mean(right),numerator=left.reduce((sum,value,i)=>sum+(value-lm)*(right[i]-rm),0),ld=Math.sqrt(left.reduce((sum,value)=>sum+(value-lm)**2,0)),rd=Math.sqrt(right.reduce((sum,value)=>sum+(value-rm)**2,0));return ld&&rd?numerator/(ld*rd):null};
    const corr=correlation(observed,predicted);
    const values=rows.flatMap(row=>[Number(row.observed_rainfall_mm)||0,Number(row.reported_rainfall_mm)||0]);
    const maximum=Math.max(.05,...values)*1.08;
    const color=terminalColors[terminal.terminal_id];
    accuracyCharts[index].setOption({
      animation:false,
      title:{text:terminal.terminal_name,subtext:`n=${rows.length} · 有雨=${rainy.length} · MAE=${mae==null?"—":mae.toFixed(3)} mm · r=${corr==null?"—":corr.toFixed(3)}`,left:"center",top:4,textStyle:{fontSize:14,fontWeight:600},subtextStyle:{fontSize:10,color:"#66716c"}},
      grid:{left:58,right:20,top:60,bottom:54},
      tooltip:{trigger:"item",confine:true,formatter:item=>{const row=item.data.detail;return `<strong>${terminal.terminal_name}</strong><br>${fmtTime(row.pass_end)}<br>真实：${Number(row.observed_rainfall_mm).toFixed(3)} mm<br>反演：${Number(row.reported_rainfall_mm).toFixed(3)} mm<br>绝对误差：${Number(row.absolute_error_mm).toFixed(3)} mm${String(row.transfer_mode||"").includes("fallback_no_position")?"<br><span style='color:#8a918d'>无位置回退</span>":""}`}},
      xAxis:{type:"value",name:"真实分钟雨量 / mm",min:0,max:maximum,nameLocation:"middle",nameGap:34,splitLine:{lineStyle:{color:"#edf0ee"}}},
      yAxis:{type:"value",name:"反演分钟雨量 / mm",min:0,max:maximum,nameLocation:"middle",nameGap:40,splitLine:{lineStyle:{color:"#edf0ee"}}},
      series:[
        {name:"完整模型",type:"scatter",data:full,symbolSize:7,itemStyle:{color,opacity:.72},markLine:{silent:true,symbol:"none",lineStyle:{color:"#7c8580",type:"dashed",width:1.2},data:[[{coord:[0,0]},{coord:[maximum,maximum]}]]}},
        {name:"无位置",type:"scatter",data:fallback,symbolSize:7,itemStyle:{color,opacity:.32,borderColor:color,borderWidth:1}}
      ]
    },true);
  });
}
function renderReliability(d){
  if(d.status!=="ok"){$("#reliability-meta").textContent=d.message||"链路分析不可用";return}
  const monthSelect=$("#reliability-month"),selected=monthSelect.value;
  if(monthSelect.options.length===1)d.available_months.forEach(month=>monthSelect.add(new Option(month,month)));
  monthSelect.value=selected;
  const months=d.monthly.map(row=>row.month),pct=value=>value==null?null:Number(value)*100;
  const duration=value=>{value=Number(value);return value>=86400?`${(value/86400).toFixed(2)} d`:value>=3600?`${(value/3600).toFixed(2)} h`:value>=60?`${(value/60).toFixed(1)} min`:`${value.toFixed(1)} s`};
  const qualityByMonth={};
  d.quality.forEach(row=>{qualityByMonth[row.month]??={};qualityByMonth[row.month][row.reason]=Number(row.rows)});
  const qualityRate=(month,reasons)=>{const values=qualityByMonth[month]||{},total=Object.values(values).reduce((a,b)=>a+b,0);return total?100*reasons.reduce((sum,key)=>sum+(values[key]||0),0)/total:null};
  const rainLabels=d.rain_rate.map(row=>row.rain_rate_bin),rainCounts=d.rain_rate.map(row=>row.sample_count);
  const qualityRainByBin={};
  (d.quality_by_rain||[]).forEach(row=>{const item=qualityRainByBin[row.rain_rate_bin]??={raw:0,noLock:0};item.raw+=Number(row.raw_rows||0);item.noLock+=Number(row.no_satellite_lock_rows||0)});
  const noLockByRain=rainLabels.map(label=>{const item=qualityRainByBin[label];return item&&item.raw>=10?100*item.noLock/item.raw:null});
  const totalSessions=d.monthly.reduce((sum,row)=>sum+Number(row.communication_sessions||0),0);
  const rainSessions=d.monthly.reduce((sum,row)=>sum+Number(row.rainy_passes||0),0);
  const gapSummary=d.gap_summary||{};
  const condition=Object.fromEntries(d.conditions.map(row=>[`${row.dimension}|${row.category}`,row]));
  const highElevation=condition["max_elevation_deg|>=60"],evening=condition["hour|18-23"];
  const firstMonth=d.monthly[0],lastMonth=d.monthly[d.monthly.length-1];
  $("#reliability-summary").innerHTML=[
    `<div class="consistency-item"><strong>${totalSessions.toLocaleString()}</strong><span>有效 PHY 通信会话</span></div>`,
    `<div class="consistency-item"><strong>${d.rain_aligned_session_count.toLocaleString()}</strong><span>已对齐雨量计会话</span></div>`,
    `<div class="consistency-item"><strong>${rainSessions.toLocaleString()}</strong><span>雨量计有雨会话</span></div>`,
    `<div class="consistency-item"><strong>${Number(d.method.near_duplicates_removed||0).toLocaleString()}</strong><span>剔除 1 秒近重复</span></div>`,
    `<div class="consistency-item"><strong>${Number(gapSummary.transition_count||0).toLocaleString()}</strong><span>查询范围会话转换</span></div>`,
    `<div class="consistency-item"><strong>${Number(gapSummary.continuous_handover_count||0).toLocaleString()}</strong><span>查询范围无缝接续</span></div>`,
    `<div class="consistency-item"><strong>${gapSummary.continuous_handover_rate==null?"—":pct(gapSummary.continuous_handover_rate).toFixed(1)+"%"}</strong><span>查询范围无缝接续率</span></div>`,
    `<div class="consistency-item"><strong>${gapSummary.median_positive_gap_s==null?"—":duration(gapSummary.median_positive_gap_s)}</strong><span>查询范围正空窗中位数</span></div>`,
    ...(lastMonth?[`<div class="consistency-item"><strong>${pct(lastMonth.continuous_handover_rate).toFixed(1)}%</strong><span>${lastMonth.month} 无缝卫星接续率</span></div>`]:[]),
    ...(lastMonth?[`<div class="consistency-item"><strong>${pct(lastMonth.link_time_coverage_rate).toFixed(1)}%</strong><span>${lastMonth.month} 链路时间覆盖率</span></div>`]:[]),
    ...(highElevation?[`<div class="consistency-item"><strong>${pct(highElevation.mean_dropout_empirical).toFixed(1)}%</strong><span>最大仰角 ≥60° 会话缺样</span></div>`]:[]),
    ...(evening?[`<div class="consistency-item"><strong>${pct(evening.mean_dropout_empirical).toFixed(1)}%</strong><span>18–23 时会话缺样</span></div>`]:[])
  ].join("");
  const causeNames={normal_or_low_dropout:"正常或低缺样",unexplained_partial_dropout:"无雨/证据不足的部分缺样",dry_weak_signal_interference_candidate:"干燥弱信号/疑似干扰",rain_associated_partial_dropout:"雨相关部分缺样"};
  $("#reliability-causes").innerHTML=d.causes.map(row=>`<div class="cause-item"><strong>${Number(row.sessions).toLocaleString()}</strong><span>${causeNames[row.cause]||row.cause}</span></div>`).join("");
  $("#reliability-table-body").innerHTML=d.monthly.map(row=>`<tr><td>${row.month}</td><td>${Number(row.communication_sessions).toLocaleString()}</td><td>${Number(row.communication_sessions_per_day).toFixed(1)}</td><td>${duration(row.median_network_gap_s)}</td><td>${duration(row.p90_positive_network_gap_s)}</td><td>${Number(row.continuous_handover_count).toLocaleString()}</td><td>${pct(row.continuous_handover_rate).toFixed(1)}%</td><td>${pct(row.link_time_coverage_rate).toFixed(1)}%</td><td>${Number(row.outage_gt_1h_count)}</td><td>${duration(row.max_network_gap_s)}</td><td>${Number(row.median_same_sat_communication_revisit_h).toFixed(2)} h</td><td>${pct(row.mean_dropout_rate_empirical).toFixed(1)}%</td></tr>`).join("");
  $("#largest-gap-body").innerHTML=d.largest_gaps.map(row=>`<tr><td>${fmtTime(row.network_gap_start)}</td><td>${fmtTime(row.network_gap_end)}</td><td>${duration(row.network_gap_s)}</td><td>${row.coverage_boundary_satellite_id}</td><td>${row.next_session_satellite_id}</td><td>${row.gap_mean_rain_rate_mm_h==null?"—":Number(row.gap_mean_rain_rate_mm_h).toFixed(3)+" mm/h"}</td><td>${row.outage_gt_1h?"采集离线/长空窗候选":"普通星间空窗"}</td></tr>`).join("");
  const provenance=d.provenance.filter(row=>row.path).map(row=>`${row.role||"来源"}：${row.path}${row.sha256?`（SHA-256 ${row.sha256.slice(0,12)}…）`:""}`);
  $("#reliability-provenance").textContent=`数据溯源：${provenance.join("；")}`;
  $("#reliability-meta").textContent=`离线统计生成于 ${d.generated_at.replace("T"," ")} · ${months.length?months.join("、"):"所选区间无会话"}`;
  $("#reliability-note").textContent=d.interpretation_note;
  if(!window.echarts)return;
  if(!reliabilityChart)reliabilityChart=echarts.init($("#reliability-chart"),null,{renderer:"canvas"});
  const commonLine={type:"line",showSymbol:true,symbolSize:7,lineStyle:{width:2}};
  reliabilityChart.setOption({
    animation:false,
    color:["#237a57","#b5682d","#2474a6","#7e5a9b","#a24b38","#8a948f"],
    tooltip:{trigger:"axis",confine:true},
    title:[
      {text:"通信密度与全网无链路空窗",left:5,top:4,textStyle:{fontSize:13,fontWeight:600}},
      {text:"会话区间无缝接续次数与比例",left:5,top:205,textStyle:{fontSize:13,fontWeight:600}},
      {text:"严格 2 秒与实测周期校正缺样率",left:5,top:405,textStyle:{fontSize:13,fontWeight:600}},
      {text:"雨强、缺样率与链路 SNR",left:5,top:605,textStyle:{fontSize:13,fontWeight:600}},
      {text:"工控机原始 PHY 质量构成",left:5,top:825,textStyle:{fontSize:13,fontWeight:600}}
    ],
    legend:[
      {top:2,right:8,data:["日均通信会话","全体空窗中位数"]},
      {top:203,right:8,data:["无缝接续次数","无缝接续比例"]},
      {top:403,right:8,data:["严格 2 秒缺样率","实测周期校正缺样率"]},
      {top:603,right:8,data:["周期校正缺样率","同期未锁星比例","平均 SNR"]},
      {top:823,right:8,data:["有效","未锁定卫星","字段异常","1 秒近重复"]}
    ],
    grid:[
      {left:72,right:72,top:42,height:115},
      {left:72,right:72,top:245,height:110},
      {left:72,right:72,top:445,height:110},
      {left:72,right:72,top:645,height:125},
      {left:72,right:30,top:865,height:125}
    ],
    xAxis:[
      {type:"category",gridIndex:0,data:months},
      {type:"category",gridIndex:1,data:months},
      {type:"category",gridIndex:2,data:months},
      {type:"category",gridIndex:3,data:rainLabels,name:"过境平均雨强 / mm/h",nameLocation:"middle",nameGap:30,axisLabel:{interval:0}},
      {type:"category",gridIndex:4,data:months}
    ],
    yAxis:[
      {type:"value",gridIndex:0,name:"会话/日",min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},
      {type:"value",gridIndex:0,name:"空窗/s",min:0,position:"right",splitLine:{show:false}},
      {type:"value",gridIndex:1,name:"接续次数",min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},
      {type:"value",gridIndex:1,name:"接续比例/%",min:0,max:100,position:"right",splitLine:{show:false}},
      {type:"value",gridIndex:2,name:"缺样率/%",min:0,max:100,splitLine:{lineStyle:{color:"#edf0ee"}}},
      {type:"value",gridIndex:3,name:"缺样率/%",min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},
      {type:"value",gridIndex:3,name:"SNR/dB",position:"right",splitLine:{show:false}},
      {type:"value",gridIndex:4,name:"原始记录/%",min:0,max:100,splitLine:{lineStyle:{color:"#edf0ee"}}}
    ],
    series:[
      {name:"日均通信会话",type:"bar",xAxisIndex:0,yAxisIndex:0,data:d.monthly.map(row=>row.communication_sessions_per_day),barMaxWidth:36},
      {name:"全体空窗中位数",...commonLine,xAxisIndex:0,yAxisIndex:1,data:d.monthly.map(row=>row.median_network_gap_s)},
      {name:"无缝接续次数",...commonLine,xAxisIndex:1,yAxisIndex:2,data:d.monthly.map(row=>row.continuous_handover_count)},
      {name:"无缝接续比例",...commonLine,xAxisIndex:1,yAxisIndex:3,data:d.monthly.map(row=>pct(row.continuous_handover_rate))},
      {name:"严格 2 秒缺样率",...commonLine,xAxisIndex:2,yAxisIndex:4,data:d.monthly.map(row=>pct(row.mean_dropout_rate))},
      {name:"实测周期校正缺样率",...commonLine,xAxisIndex:2,yAxisIndex:4,data:d.monthly.map(row=>pct(row.mean_dropout_rate_empirical))},
      {name:"周期校正缺样率",type:"bar",xAxisIndex:3,yAxisIndex:5,data:d.rain_rate.map(row=>({value:pct(row.mean_dropout_empirical),sample_count:row.sample_count})),barMaxWidth:34},
      {name:"同期未锁星比例",...commonLine,xAxisIndex:3,yAxisIndex:5,data:noLockByRain},
      {name:"平均 SNR",...commonLine,xAxisIndex:3,yAxisIndex:6,data:d.rain_rate.map(row=>row.mean_snr)},
      {name:"有效",type:"bar",stack:"quality",xAxisIndex:4,yAxisIndex:7,data:months.map(month=>qualityRate(month,["valid"]))},
      {name:"未锁定卫星",type:"bar",stack:"quality",xAxisIndex:4,yAxisIndex:7,data:months.map(month=>qualityRate(month,["no_satellite_lock"]))},
      {name:"字段异常",type:"bar",stack:"quality",xAxisIndex:4,yAxisIndex:7,data:months.map(month=>qualityRate(month,["invalid_localTime","missing_satelliteId","missing_rssi","missing_snr","invalid_snr_255","missing_freqOffset","invalid_freqOffset_zero","missing_td","invalid_td_zero"]))},
      {name:"1 秒近重复",type:"bar",stack:"quality",xAxisIndex:4,yAxisIndex:7,data:months.map(month=>qualityRate(month,["near_duplicate"]))}
    ]
  },true);
}
async function loadReliability(){
  const month=$("#reliability-month").value;
  let query="";
  if(month){const [year,value]=month.split("-").map(Number),end=new Date(year,value,1);query=`?start=${month}-01T00:00:00&end=${localDate(end)}T00:00:00`}
  $("#reliability-meta").textContent="正在读取工控机原始备份统计";
  try{const response=await fetch(`/api/link-reliability-analysis${query}`);if(!response.ok)throw new Error(await response.text());renderReliability(await response.json())}
  catch(error){$("#reliability-meta").textContent=`链路可靠性分析失败：${error.message}`}
}
$("#reliability-month").onchange=loadReliability;
$("#reliability-refresh").onclick=loadReliability;
function renderPositionAccuracy(d){
  const overview=d.retrieval_geometry_overview||{},summary=d.retrieval_geometry_summary||[];
  const number=value=>value==null?"—":Number(value).toLocaleString(),mm=value=>value==null?"—":`${Number(value).toFixed(4)} mm`,pct=value=>value==null?"—":`${(100*Number(value)).toFixed(1)}%`;
  $("#position-accuracy-summary").innerHTML=[
    `<div class="consistency-item"><strong>${number(overview.sample_count)}</strong><span>位置关联分钟样本</span></div>`,
    `<div class="consistency-item"><strong>${number(overview.rainy_sample_count)}</strong><span>其中真实有雨</span></div>`,
    `<div class="consistency-item"><strong>${mm(overview.mae_mm)}</strong><span>全部天气 MAE</span></div>`,
    `<div class="consistency-item"><strong>${mm(overview.rainy_mae_mm)}</strong><span>有雨分钟 MAE</span></div>`,
    `<div class="consistency-item"><strong>${overview.correlation==null?"—":Number(overview.correlation).toFixed(3)}</strong><span>逐分钟相关系数</span></div>`,
    `<div class="consistency-item"><strong>${pct(overview.geometry_match_rate)}</strong><span>完整位置记录关联率</span></div>`
  ].join("");
  if(!window.echarts)return;
  if(!positionAccuracyChart)positionAccuracyChart=echarts.init($("#position-accuracy-chart"),null,{renderer:"canvas"});
  const elevation=summary.filter(row=>row.dimension==="elevation_deg"),range=summary.filter(row=>row.dimension==="slant_range_km");
  const bar=(name,rows,key,xAxisIndex,yAxisIndex,color)=>({name,type:"bar",xAxisIndex,yAxisIndex,data:rows.map(row=>({value:row[key],detail:row})),barMaxWidth:28,itemStyle:{color}});
  const count=(rows,xAxisIndex,yAxisIndex)=>({name:"样本数",type:"line",xAxisIndex,yAxisIndex,data:rows.map(row=>({value:row.sample_count,detail:row})),symbolSize:5,lineStyle:{color:"#66716c",width:1.3},itemStyle:{color:"#66716c"}});
  positionAccuracyChart.setOption({
    animation:false,
    legend:{top:2,data:["全部天气 MAE","有雨分钟 MAE","样本数"]},
    title:[{text:"最大仰角分箱",left:8,top:30,textStyle:{fontSize:13}},{text:"平均星地距离分箱",left:"52%",top:30,textStyle:{fontSize:13}}],
    grid:[{left:65,right:"54%",top:70,bottom:62},{left:"57%",right:58,top:70,bottom:62}],
    tooltip:{trigger:"axis",confine:true,axisPointer:{type:"shadow"},formatter:items=>{const row=items[0]?.data?.detail;if(!row)return "";return `<strong>${row.category}</strong><br>样本：${row.sample_count}（有雨 ${row.rainy_sample_count}）<br>全部天气 MAE：${row.mae_mm==null?"—":Number(row.mae_mm).toFixed(4)+" mm"}<br>有雨分钟 MAE：${row.rainy_mae_mm==null?"—":Number(row.rainy_mae_mm).toFixed(4)+" mm"}<br>平均偏差：${row.bias_mm==null?"—":Number(row.bias_mm).toFixed(4)+" mm"}`}},
    xAxis:[{type:"category",gridIndex:0,data:elevation.map(row=>row.category),name:"最大仰角 / °",nameLocation:"middle",nameGap:42,axisLabel:{rotate:24}},{type:"category",gridIndex:1,data:range.map(row=>row.category),name:"平均星地距离 / km",nameLocation:"middle",nameGap:48,axisLabel:{rotate:24}}],
    yAxis:[{type:"value",gridIndex:0,name:"MAE / mm",min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},{type:"value",gridIndex:0,name:"样本数",position:"right",min:0,splitLine:{show:false}},{type:"value",gridIndex:1,name:"MAE / mm",min:0,splitLine:{lineStyle:{color:"#edf0ee"}}},{type:"value",gridIndex:1,name:"样本数",position:"right",min:0,splitLine:{show:false}}],
    series:[bar("全部天气 MAE",elevation,"mae_mm",0,0,"#2474a6"),bar("有雨分钟 MAE",elevation,"rainy_mae_mm",0,0,"#b5682d"),count(elevation,0,1),bar("全部天气 MAE",range,"mae_mm",1,2,"#2474a6"),bar("有雨分钟 MAE",range,"rainy_mae_mm",1,2,"#b5682d"),count(range,1,3)]
  },true);
}
function renderPositionLink(d){
  if(d.status!=="ok"){$("#position-meta").textContent=d.message||"星地位置统计不可用";return}
  const pct=value=>value==null?"—":`${(Number(value)*100).toFixed(1)}%`,number=value=>value==null?"—":Number(value).toLocaleString();
  const overview=d.overview||{},satelliteSelect=$("#position-satellite"),selected=satelliteSelect.value;
  if(satelliteSelect.options.length===1)d.satellites.forEach(row=>{const current=row.latest_let_id==null?"待映射":`LET ${Number(row.latest_let_id)}`,physical=row.physical_norad_id==null?"物理ID待映射":`NORAD ${Number(row.physical_norad_id)}`,name=row.physical_name||row.identity_key;satelliteSelect.add(new Option(`${current} · ${physical} · ${name}`,row.identity_key))});
  satelliteSelect.value=selected;
  $("#position-summary").innerHTML=[
    `<div class="consistency-item"><strong>${number(overview.position_opportunities)}</strong><span>位置可见机会</span></div>`,
    `<div class="consistency-item"><strong>${number(overview.visibility_opportunities_with_phy)}</strong><span>可见机会内有 PHY</span></div>`,
    `<div class="consistency-item"><strong>${number(overview.phy_sessions)}</strong><span>有效 PHY 会话</span></div>`,
    `<div class="consistency-item"><strong>${number(overview.position_matched_phy_sessions)}</strong><span>同星位置匹配会话</span></div>`,
    `<div class="consistency-item"><strong>${number(overview.identified_physical_satellites)}</strong><span>已确认物理卫星</span></div>`,
    `<div class="consistency-item"><strong>${pct(overview.mean_dropout_rate)}</strong><span>通信会话平均断链率</span></div>`,
    ...(()=>{const q=d.phy_quality||[],total=q.reduce((sum,row)=>sum+Number(row.rows||0),0),sum=reason=>q.filter(row=>row.quality_reason===reason).reduce((value,row)=>value+Number(row.rows||0),0);return [`<div class="consistency-item"><strong>${total?pct(sum("valid")/total):"—"}</strong><span>原始 PHY 有效率</span></div>`,`<div class="consistency-item"><strong>${total?pct(sum("no_satellite_lock")/total):"—"}</strong><span>未锁定卫星比例</span></div>`,`<div class="consistency-item"><strong>${total?pct(sum("near_duplicate")/total):"—"}</strong><span>1秒近重复比例</span></div>`]})(),
    ...(()=>{const q=d.position_quality||[],total=q.reduce((sum,row)=>sum+Number(row.rows||0),0),valid=q.filter(row=>row.quality_reason==="valid").reduce((sum,row)=>sum+Number(row.rows||0),0);return [`<div class="consistency-item"><strong>${total?pct(valid/total):"—"}</strong><span>原始位置有效率</span></div>`]})()
  ].join("");
  $("#position-monthly-body").innerHTML=(d.monthly||[]).map(row=>`<tr><td>${row.month}</td><td>${number(row.phy_sessions)}</td><td>${number(row.physical_satellites)}</td><td>${pct(row.mean_dropout_rate)}</td><td>${Number(row.mean_rssi).toFixed(2)}</td><td>${Number(row.mean_snr).toFixed(2)}</td><td>${Number(row.mean_max_elevation_deg).toFixed(1)}°</td><td>${Number(row.mean_slant_range_km).toFixed(0)} km</td><td>${row.mean_altitude_km==null?"—":Number(row.mean_altitude_km).toFixed(0)+" km"}</td></tr>`).join("");
  $("#position-satellite-body").innerHTML=d.satellites.map(row=>`<tr><td>${row.latest_let_id==null?"—":Number(row.latest_let_id)}</td><td>${row.physical_norad_id==null?"待映射":Number(row.physical_norad_id)}</td><td>${row.physical_name||row.identity_key}</td><td>${number(row.pass_count)}</td><td>${pct(row.mean_dropout_rate)}</td><td>${pct(row.p90_dropout_rate)}</td><td>${row.mean_rssi==null?"—":Number(row.mean_rssi).toFixed(2)}</td><td>${row.mean_snr==null?"—":Number(row.mean_snr).toFixed(2)}</td><td>${row.median_max_elevation_deg==null?"—":Number(row.median_max_elevation_deg).toFixed(1)+"°"}</td><td>${row.median_slant_range_km==null?"—":Number(row.median_slant_range_km).toFixed(0)+" km"}</td><td>${row.median_altitude_km==null?"—":Number(row.median_altitude_km).toFixed(0)+" km"}</td></tr>`).join("");
  $("#position-pass-body").innerHTML=d.selected_passes.map(row=>`<tr><td>${fmtTime(row.pass_start)}</td><td>${row.actual_phy_points}/${row.expected_phy_points}</td><td>${pct(row.dropout_rate)}</td><td>${row.max_elevation_deg==null?"—":Number(row.max_elevation_deg).toFixed(1)+"°"}</td><td>${row.mean_slant_range_km==null?"—":Number(row.mean_slant_range_km).toFixed(0)+" km"}</td><td>${row.longitude_deg==null?"—":`${Number(row.longitude_deg).toFixed(2)}°, ${Number(row.latitude_deg).toFixed(2)}°, ${Number(row.altitude_km).toFixed(0)} km`}</td><td>${row.ecef_x_km==null?"—":`${Number(row.ecef_x_km).toFixed(0)}, ${Number(row.ecef_y_km).toFixed(0)}, ${Number(row.ecef_z_km).toFixed(0)} km`}</td></tr>`).join("");
  $("#position-meta").textContent=`全量统计生成于 ${d.generated_at.replace("T"," ")} · ${selected?`当前筛选 ${selected}`:"显示全部卫星"}`;
  const provenance=d.provenance||{};$("#position-provenance").textContent=`数据溯源：原始恢复库 ${provenance.raw_database||"—"}（SHA-256 ${String(provenance.raw_database_sha256||"").slice(0,12)}…）；物理映射 ${provenance.mapping_csv||"—"}（SHA-256 ${String(provenance.mapping_sha256||"").slice(0,12)}…）`;
  renderPositionAccuracy(d);
  if(!window.echarts)return;if(!positionChart)positionChart=echarts.init($("#position-chart"),null,{renderer:"canvas"});
  const elevation=d.geometry_summary.filter(row=>row.dimension==="elevation_deg"),range=d.geometry_summary.filter(row=>row.dimension==="slant_range_km"),altitude=d.geometry_summary.filter(row=>row.dimension==="altitude_km");
  const geo=d.plot_passes.filter(row=>row.longitude_deg!=null&&row.latitude_deg!=null).map(row=>({value:[row.longitude_deg,row.latitude_deg,100*row.dropout_rate,row.altitude_km,row.max_elevation_deg,row.mean_slant_range_km],detail:row}));
  const ecef=d.plot_passes.filter(row=>row.ecef_x_km!=null&&row.ecef_y_km!=null).map(row=>({value:[row.ecef_x_km,row.ecef_y_km,100*row.dropout_rate,row.ecef_z_km,row.max_elevation_deg,row.mean_slant_range_km],detail:row}));
  positionChart.setOption({animation:false,color:["#1769aa","#b5682d","#237a57"],title:[{text:"断链率与最大仰角",left:8,top:4,textStyle:{fontSize:13}},{text:"断链率与平均星地距离",left:"34%",top:4,textStyle:{fontSize:13}},{text:"断链率与轨道高度",left:"68%",top:4,textStyle:{fontSize:13}},{text:"地理坐标：经度-纬度",left:8,top:430,textStyle:{fontSize:13}},{text:"ECEF：X-Y 投影",left:"52%",top:430,textStyle:{fontSize:13}}],grid:[{left:62,right:"69%",top:45,height:300},{left:"36%",right:"36%",top:45,height:300},{left:"70%",right:25,top:45,height:300},{left:70,right:"53%",top:475,height:300},{left:"56%",right:35,top:475,height:300}],tooltip:{trigger:"item",confine:true,formatter:item=>{if(["仰角分箱","距离分箱","高度分箱"].includes(item.seriesName))return `${item.name}<br>平均断链率：${Number(item.value).toFixed(1)}%<br>样本：${item.data.samples}`;if(item.seriesName==="上海地面终端")return `<strong>上海地面终端</strong><br>经度 E 121.4160°<br>纬度 N 31.2185°`;const v=item.value,row=item.data.detail;if(item.seriesName==="经纬度")return `${fmtTime(row.pass_start)}<br>经度 ${Number(v[0]).toFixed(3)}° · 纬度 ${Number(v[1]).toFixed(3)}°<br>高度 ${Number(v[3]).toFixed(1)} km<br>仰角 ${Number(v[4]).toFixed(1)}° · 距离 ${Number(v[5]).toFixed(0)} km<br>断链率 ${Number(v[2]).toFixed(1)}%`;return `${fmtTime(row.pass_start)}<br>ECEF X ${Number(v[0]).toFixed(0)} km · Y ${Number(v[1]).toFixed(0)} km · Z ${Number(v[3]).toFixed(0)} km<br>仰角 ${Number(v[4]).toFixed(1)}° · 距离 ${Number(v[5]).toFixed(0)} km<br>断链率 ${Number(v[2]).toFixed(1)}%`}},xAxis:[{type:"category",gridIndex:0,data:elevation.map(row=>row.category),name:"最大仰角 / °",axisLabel:{rotate:25}},{type:"category",gridIndex:1,data:range.map(row=>row.category),name:"平均距离 / km",axisLabel:{rotate:25}},{type:"category",gridIndex:2,data:altitude.map(row=>row.category),name:"轨道高度 / km",axisLabel:{rotate:25}},{type:"value",gridIndex:3,name:"经度 / °"},{type:"value",gridIndex:4,name:"ECEF X / km",scale:true}],yAxis:[{type:"value",gridIndex:0,name:"断链率/%",min:0,max:100},{type:"value",gridIndex:1,name:"断链率/%",min:0,max:100},{type:"value",gridIndex:2,name:"断链率/%",min:0,max:100},{type:"value",gridIndex:3,name:"纬度 / °"},{type:"value",gridIndex:4,name:"ECEF Y / km",scale:true}],visualMap:{type:"continuous",dimension:2,seriesIndex:[3,4],min:0,max:100,right:3,top:500,itemHeight:180,text:["断链高","断链低"],inRange:{color:["#2a9d69","#f2c14e","#c94b45"]}},dataZoom:[{type:"inside",xAxisIndex:[3,4]},{type:"inside",yAxisIndex:[3,4]}],series:[{name:"仰角分箱",type:"bar",xAxisIndex:0,yAxisIndex:0,data:elevation.map(row=>({value:100*Number(row.mean_dropout_rate),samples:row.pass_count,name:row.category})),barMaxWidth:42},{name:"距离分箱",type:"bar",xAxisIndex:1,yAxisIndex:1,data:range.map(row=>({value:100*Number(row.mean_dropout_rate),samples:row.pass_count,name:row.category})),barMaxWidth:42},{name:"高度分箱",type:"bar",xAxisIndex:2,yAxisIndex:2,data:altitude.map(row=>({value:100*Number(row.mean_dropout_rate),samples:row.pass_count,name:row.category})),barMaxWidth:42},{name:"经纬度",type:selected?"line":"scatter",xAxisIndex:3,yAxisIndex:3,data:geo,showSymbol:true,symbolSize:6,lineStyle:{width:selected?1.2:0,opacity:.45},itemStyle:{opacity:.55}},{name:"ECEF",type:selected?"line":"scatter",xAxisIndex:4,yAxisIndex:4,data:ecef,showSymbol:true,symbolSize:6,lineStyle:{width:selected?1.2:0,opacity:.45},itemStyle:{opacity:.55}},{name:"上海地面终端",type:"scatter",xAxisIndex:3,yAxisIndex:3,data:[[121.416,31.2185]],symbolSize:8,z:20,itemStyle:{color:"#d62728",borderColor:"#fff",borderWidth:1.5},label:{show:true,formatter:"地面终端",position:"right",color:"#b42318",fontSize:11}}]},true);
  positionChart.setOption({
    title:[
      {text:"断链率与最大仰角（°）"},
      {text:"断链率与平均星地距离（km）"},
      {text:"断链率与轨道高度（km）"},
      {text:"地理坐标：经度-纬度（°）"},
      {text:"ECEF：X-Y 投影（km）"}
    ],
    xAxis:[
      {name:""},{name:""},{name:""},
      {name:"",scale:true},{name:"",scale:true}
    ],
    yAxis:[
      {name:""},{name:""},{name:""},
      {name:"",scale:true},{name:"",scale:true}
    ]
  });
  if(!orbitChart)orbitChart=echarts.init($("#orbit-chart"),null,{renderer:"canvas"});
  const orbitRows=(d.plot_passes||[]).filter(row=>row.pass_start&&row.altitude_km!=null).sort((a,b)=>Date.parse(a.pass_start)-Date.parse(b.pass_start));
  const orbitType=selected?"line":"scatter",commonOrbit={type:orbitType,showSymbol:true,symbolSize:selected?5:3,connectNulls:false,lineStyle:{width:1.2}};
  orbitChart.setOption({animation:false,color:["#2474a6","#237a57","#b5682d"],legend:{top:0,right:8,data:["卫星高度","最大仰角","平均星地距离"]},grid:[{left:72,right:70,top:45,height:135},{left:72,right:72,top:250,height:130}],tooltip:{trigger:"axis",confine:true,formatter:items=>{const row=items[0]?.data?.detail;if(!row)return "";return `<strong>${fmtTime(row.pass_start)}</strong><br>卫星：${row.identity_key}<br>高度：${Number(row.altitude_km).toFixed(1)} km<br>最大仰角：${Number(row.max_elevation_deg).toFixed(1)}°<br>平均星地距离：${Number(row.mean_slant_range_km).toFixed(0)} km<br>断链率：${(100*Number(row.dropout_rate)).toFixed(1)}%`}},xAxis:[{type:"time",gridIndex:0,axisLabel:{show:false},splitLine:{show:true,lineStyle:{color:"#edf0ee"}}},{type:"time",gridIndex:1,splitLine:{show:true,lineStyle:{color:"#edf0ee"}}}],yAxis:[{type:"value",gridIndex:0,name:"高度 / km",scale:true,splitLine:{lineStyle:{color:"#edf0ee"}}},{type:"value",gridIndex:1,name:"仰角 / °",min:0,max:90,splitLine:{lineStyle:{color:"#edf0ee"}}},{type:"value",gridIndex:1,name:"距离 / km",position:"right",scale:true,splitLine:{show:false}}],dataZoom:[{type:"inside",xAxisIndex:[0,1],filterMode:"none"},{type:"slider",xAxisIndex:[0,1],filterMode:"none",bottom:5,height:20}],series:[{name:"卫星高度",...commonOrbit,xAxisIndex:0,yAxisIndex:0,data:orbitRows.map(row=>({value:[row.pass_start,row.altitude_km],detail:row}))},{name:"最大仰角",...commonOrbit,xAxisIndex:1,yAxisIndex:1,data:orbitRows.map(row=>({value:[row.pass_start,row.max_elevation_deg],detail:row}))},{name:"平均星地距离",...commonOrbit,xAxisIndex:1,yAxisIndex:2,data:orbitRows.map(row=>({value:[row.pass_start,row.mean_slant_range_km],detail:row}))}]},true);
}
async function loadPositionLink(){
  const identity=$("#position-satellite").value,query=identity?`?identity_key=${encodeURIComponent(identity)}`:"";$("#position-meta").textContent="正在读取星地位置关联统计";
  try{const response=await fetch(`/api/position-link-analysis${query}`);if(!response.ok)throw new Error(await response.text());renderPositionLink(await response.json())}catch(error){$("#position-meta").textContent=`星地位置统计失败：${error.message}`}
}
$("#position-satellite").onchange=loadPositionLink;
function terminalHtml(t){
  const s=t.summary, empty=!s.pass_count;
  const mm=v=>v===null||v===undefined?"—":Number(v).toFixed(3);
  const latestLink=t.query_latest_pass_end||t.latest_link_time;
  const rows=t.predictions.map(r=>{const fallback=String(r.transfer_mode||"").includes("fallback_no_position");return `<tr><td>${fmtTime(r.pass_start).slice(11)}</td><td>${r.satellite_id}</td><td>${r.points}${fallback?'<span style="color:#8a918d;font-size:11px">（无位置）</span>':''}</td><td class="${r.reported_rainfall_mm>0?"rain":""}">${mm(r.reported_rainfall_mm)}</td><td>${mm(r.observed_rainfall_mm)}</td><td>${mm(r.absolute_error_mm)}</td><td>${(r.rain_probability*100).toFixed(1)}%</td></tr>`}).join("");
  return `<article class="terminal"><div class="terminal-head"><div><h2>${t.terminal_name}</h2><div class="terminal-id">${t.terminal_id}</div></div><span class="status ${empty?"empty":""}">${empty?"无有效分钟":t.source==="history"?"历史结果":"按所选日期计算"}</span></div>
  <div class="metrics"><div class="metric"><strong>${s.pass_count}</strong><span>有效分钟窗口</span></div><div class="metric"><strong>${s.rainy_pass_count}</strong><span>模型有雨</span></div><div class="metric"><strong>${s.observed_rainy_pass_count}</strong><span>雨量计有雨</span></div><div class="metric"><strong>${Number(s.max_reported_rainfall_mm).toFixed(3)}</strong><span>模型最大 / mm</span></div><div class="metric"><strong>${mm(s.max_observed_rainfall_mm)}</strong><span>雨量计最大 / mm</span></div><div class="metric"><strong>${mm(s.mae_mm)}</strong><span>分钟 MAE / mm</span></div></div>
  <div class="freshness">所选日期：${t.query_date}　${empty?"数据源末次链路":"当日末次有效分钟"}：${fmtTime(latestLink)}</div>
  ${empty?`<div class="empty-state">所选日期没有满足位置、气象对齐及最少采样点要求的分钟窗口。</div>`:`<div class="table-wrap"><table><thead><tr><th>开始</th><th>卫星 ID</th><th>有效 PHY 点数</th><th>模型 / mm</th><th>雨量计 / mm</th><th>绝对误差</th><th>有雨概率</th></tr></thead><tbody>${rows}</tbody></table><div class="freshness">有效 PHY 点数为完成同卫星位置匹配和气象数据对齐后的点数，并非数据库中的原始 PHY 点数。</div></div>`}</article>`;
}
async function fetchJsonWithRetry(url,attempts=2){
  let lastError;
  for(let attempt=0;attempt<attempts;attempt++){
    try{const response=await fetch(url);if(!response.ok)throw new Error(await response.text());return await response.json()}
    catch(error){lastError=error;if(attempt+1<attempts)await new Promise(resolve=>setTimeout(resolve,700))}
  }
  throw lastError;
}
async function load(){
  const day=$("#date").value;if(!day||loading)return;loading=true;$("#notice").textContent="正在读取历史反演结果…";$("#query").disabled=true;
  const start=`${day}T00:00:00`,end=`${localDate(new Date(new Date(start).getTime()+86400000))}T00:00:00`;
  try{const d=await fetchJsonWithRetry(`/api/rainfall?date=${day}&max_passes=2000&recompute=false`),ratio=d.model_version.split("/")[0].replace(/^minute-/,"").replace("to",":").replace("native","原始分布");$("#model").textContent=`三终端 ${ratio} 分钟模型`;$("#model").title=d.model_version;$("#grid").innerHTML=d.terminals.map(terminalHtml).join("");try{renderAccuracy(d)}catch(error){console.error("accuracy render failed",error)}$("#notice").textContent=`${d.query_date} · 历史反演已载入，正在读取原始链路时间轴…`;try{renderTimeline(await fetchJsonWithRetry(`/api/timeline?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&resolution_minutes=1`));$("#notice").textContent=`${d.query_date} · 更新于 ${d.generated_at.replace("T"," ")}`}catch(error){console.error("timeline load failed",error);$("#notice").textContent=`${d.query_date} · 历史反演已载入，时间轴读取失败：${error.message}`}}
  catch(e){$("#notice").textContent=`查询失败：${e.message}`}
  finally{loading=false;$("#query").disabled=false}
}
$("#query").onclick=load;
$("#latest").onclick=async()=>{const d=await (await fetch("/api/latest-data-date")).json();if(d.date){$("#date").value=d.date;load()}};
load();
loadReliability();
loadPositionLink();
setInterval(()=>{if($("#date").value===localDate(new Date()))load()},30000);
window.addEventListener("resize",()=>{if(timelineChart)timelineChart.resize();accuracyCharts.forEach(chart=>chart&&chart.resize());if(reliabilityChart)reliabilityChart.resize();if(positionChart)positionChart.resize();if(orbitChart)orbitChart.resize();if(positionAccuracyChart)positionAccuracyChart.resize()});
</script>
</body></html>"""


def create_app(runner: ThreeTerminalRunner) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runner.start_worker()
        try:
            yield
        finally:
            runner.stop_worker()

    app = FastAPI(
        title="Three-terminal Stage1 Rainfall Demo",
        version="0.1.0",
        lifespan=lifespan,
    )
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": str(runner.device),
            "terminals": list(runner.builders),
            "model_version": runner.model_version,
            "history": runner.history.stats(),
            "worker": runner.worker_state,
        }

    @app.get("/api/latest-data-date")
    def latest_data_date():
        value = runner.latest_data_date()
        return {"date": value.isoformat() if value else None}

    @app.get("/api/timeline")
    def timeline(
        start: datetime,
        end: datetime,
        resolution_minutes: int = Query(default=1, ge=1, le=60),
    ):
        if end <= start:
            raise HTTPException(
                status_code=400, detail="end must be later than start"
            )
        if end - start > timedelta(days=31):
            raise HTTPException(
                status_code=400,
                detail="timeline range cannot exceed 31 days",
            )
        return runner.timeline(start, end, resolution_minutes)

    @app.get("/api/link-dropout-stats")
    def link_dropout_stats(
        start: datetime | None = None,
        end: datetime | None = None,
        session_gap_s: float = Query(default=900.0, ge=60.0, le=3600.0),
    ):
        if (start is None) != (end is None):
            raise HTTPException(
                status_code=400, detail="start and end must be provided together"
            )
        if start is not None and end is not None:
            if end <= start:
                raise HTTPException(
                    status_code=400, detail="end must be later than start"
                )
            if end - start > timedelta(days=180):
                raise HTTPException(
                    status_code=400,
                    detail="dropout statistics range cannot exceed 180 days",
                )
        return runner.link_dropout_stats(start, end, session_gap_s)

    @app.get("/api/link-reliability-analysis")
    def link_reliability_analysis(
        start: datetime | None = None,
        end: datetime | None = None,
    ):
        if (start is None) != (end is None):
            raise HTTPException(status_code=400, detail="start and end must be provided together")
        if start is not None and end is not None and end <= start:
            raise HTTPException(status_code=400, detail="end must be later than start")
        return runner.link_reliability_analysis(start, end)

    @app.get("/api/position-link-analysis")
    def position_link_analysis(identity_key: str | None = None):
        return runner.position_link_analysis(identity_key)

    @app.get("/api/history/stats")
    def history_stats():
        return runner.history.stats()

    @app.get("/api/rainy-dates")
    def rainy_dates():
        rows = runner.history.rainy_dates()
        return {"status": "ok", "count": len(rows), "dates": rows}

    @app.post("/api/history/update")
    def history_update():
        return runner.update_recent_once()

    @app.get("/api/rainfall")
    def rainfall(
        date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        max_passes: int = Query(default=500, ge=1, le=2000),
        recompute: bool = Query(default=False),
    ):
        try:
            parsed = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD") from exc
        return runner.query_date(
            parsed,
            max_passes,
            force_recompute=recompute,
        )

    @app.post("/api/query")
    def natural_query(request: QueryRequest):
        try:
            parsed = _parse_query_date(request.query)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = runner.query_date(parsed, request.max_passes)
        return {"response": runner.summarize(result), "result": result}

    @app.get("/api/tags")
    def ollama_tags():
        return {
            "models": [
                {
                    "name": "stage1-three-terminal",
                    "model": "stage1-three-terminal",
                    "modified_at": datetime.now().astimezone().isoformat(),
                    "size": 0,
                }
            ]
        }

    @app.post("/api/generate")
    def ollama_generate(request: OllamaGenerateRequest):
        if request.stream:
            raise HTTPException(
                status_code=400, detail="demo currently supports stream=false only"
            )
        try:
            parsed = _parse_query_date(request.prompt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = runner.query_date(parsed)
        return {
            "model": request.model,
            "created_at": datetime.now().astimezone().isoformat(),
            "response": runner.summarize(result),
            "done": True,
            "done_reason": "stop",
            "context": [],
            "terminal_results": result,
        }

    return app
