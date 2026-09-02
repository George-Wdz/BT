#!/usr/bin/env python3
"""Traceable PHY availability, data-quality, constellation-density and rain analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


INVALID_SATELLITE_ID = 4294967295
PHY_COLUMNS = [
    "id", "localTime", "satelliteId", "earthStationId", "phyModcod",
    "rssi", "phyRssi", "lastCniValue", "snr", "freqOffset", "td", "ncr",
    "rptNcrTimer", "reportTimestamp", "bdtTime",
]
POSITION_COLUMNS = [
    "id", "localTime", "ueId", "satId", "altitude", "posLatitude",
    "posLongitude", "northSouthDirSpeed", "eastWestDirSpeed",
    "verticalDirSpeed", "ecefPx", "ecefPy", "ecefPz", "longitude",
    "latitude", "satAltitude", "reportTimestamp", "bdtTime",
    "visibleSatCount", "visibleSatPos",
]
RAIN_EDGES = [-1e-12, 1e-12, 0.1, 0.5, 1, 2, 5, 10, 20, 50, np.inf]
RAIN_LABELS = ["0", "0-0.1", "0.1-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", ">=50"]
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_time_bounds(path: Path, column: str, chunksize: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    minimum, maximum = None, None
    for frame in pd.read_csv(path, usecols=[column], chunksize=chunksize):
        values = pd.to_datetime(frame[column], errors="coerce", format="mixed").dropna()
        if values.empty:
            continue
        minimum = values.min() if minimum is None else min(minimum, values.min())
        maximum = values.max() if maximum is None else max(maximum, values.max())
    if minimum is None or maximum is None:
        raise ValueError(f"no valid timestamps in {path}")
    return minimum, maximum


def db_time_bound(path: Path, table: str, column: str, aggregate: str = "MAX") -> pd.Timestamp:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        value = conn.execute(f"SELECT {aggregate}({column}) FROM {table}").fetchone()[0]
    if value is None:
        raise ValueError(f"no {column} values in {path}:{table}")
    return pd.Timestamp(value)


def classify_phy(frame: pd.DataFrame) -> pd.Series:
    reason = pd.Series("valid", index=frame.index, dtype="object")
    timestamp = pd.to_datetime(frame["localTime"], errors="coerce")
    satellite = pd.to_numeric(frame["satelliteId"], errors="coerce")
    numeric = {
        name: pd.to_numeric(frame[name], errors="coerce")
        for name in ["rssi", "snr", "freqOffset", "td"]
    }
    rules = [
        (timestamp.isna(), "invalid_localTime"),
        (satellite.isna(), "missing_satelliteId"),
        (satellite == INVALID_SATELLITE_ID, "no_satellite_lock"),
        (numeric["rssi"].isna(), "missing_rssi"),
        (numeric["snr"].isna(), "missing_snr"),
        (numeric["snr"] == 255, "invalid_snr_255"),
        (numeric["freqOffset"].isna(), "missing_freqOffset"),
        (numeric["freqOffset"] == 0, "invalid_freqOffset_zero"),
        (numeric["td"].isna(), "missing_td"),
        (numeric["td"] == 0, "invalid_td_zero"),
    ]
    unassigned = pd.Series(True, index=frame.index)
    for mask, value in rules:
        selected = unassigned & mask
        reason.loc[selected] = value
        unassigned.loc[selected] = False
    return reason


def iter_phy_sources(early_db: Path, raw_csv: Path, cutover: pd.Timestamp, chunksize: int):
    with sqlite3.connect(f"file:{early_db}?mode=ro", uri=True) as conn:
        query = "SELECT * FROM phy_data WHERE datetime(localTime) <= datetime(?) ORDER BY localTime, id"
        for frame in pd.read_sql_query(query, conn, params=[cutover.isoformat()], chunksize=chunksize):
            frame["source_file"] = str(early_db)
            frame["source_record_id"] = frame["id"]
            yield frame
    for frame in pd.read_csv(raw_csv, chunksize=chunksize, low_memory=False):
        timestamp = pd.to_datetime(frame["localTime"], errors="coerce", format="mixed")
        frame = frame.loc[timestamp > cutover].copy()
        if frame.empty:
            continue
        frame["source_file"] = str(raw_csv)
        frame["source_record_id"] = frame["id"]
        yield frame


def load_phy(early_db: Path, raw_csv: Path, cutover: pd.Timestamp, chunksize: int):
    valid_parts, quality_parts, quality_minute_parts = [], [], []
    for frame in iter_phy_sources(early_db, raw_csv, cutover, chunksize):
        frame["timestamp"] = pd.to_datetime(frame["localTime"], errors="coerce", format="mixed")
        frame["reason"] = classify_phy(frame)
        frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
        quality_parts.append(
            frame.groupby(["month", "reason"], dropna=False).size().rename("rows").reset_index()
        )
        quality_minute_parts.append(
            frame.assign(minute=frame["timestamp"].dt.floor("min"))
            .groupby(["minute", "reason"], dropna=False)
            .size().rename("rows").reset_index()
        )
        selected = frame.loc[frame["reason"] == "valid", [
            "timestamp", "satelliteId", "rssi", "phyRssi", "lastCniValue",
            "snr", "freqOffset", "td", "source_file", "source_record_id",
        ]].copy()
        for col in ["satelliteId", "rssi", "phyRssi", "lastCniValue", "snr", "freqOffset", "td"]:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
        valid_parts.append(selected)
    valid = pd.concat(valid_parts, ignore_index=True).sort_values("timestamp")
    feature_cols = ["satelliteId", "rssi", "phyRssi", "lastCniValue", "snr", "freqOffset", "td"]
    hashes = pd.util.hash_pandas_object(valid[feature_cols], index=False)
    previous = valid.groupby(hashes, sort=False)["timestamp"].diff().dt.total_seconds()
    near_duplicate = previous.between(0, 1.0, inclusive="both")
    duplicate_quality = None
    duplicate_events = valid.loc[near_duplicate, ["timestamp"]].copy()
    if near_duplicate.any():
        duplicate_quality = (
            valid.loc[near_duplicate]
            .assign(month=lambda frame: frame.timestamp.dt.strftime("%Y-%m"), reason="near_duplicate")
            .groupby(["month", "reason"]).size().rename("rows").reset_index()
        )
    valid = valid.loc[~near_duplicate].reset_index(drop=True)
    quality = pd.concat(quality_parts).groupby(["month", "reason"], as_index=False)["rows"].sum()
    quality_minute = (
        pd.concat(quality_minute_parts)
        .groupby(["minute", "reason"], as_index=False)["rows"].sum()
    )
    if duplicate_quality is not None:
        duplicate_by_month = duplicate_quality.set_index("month").rows
        valid_mask = quality.reason == "valid"
        quality.loc[valid_mask, "rows"] -= quality.loc[valid_mask, "month"].map(duplicate_by_month).fillna(0).astype(int)
        quality = pd.concat([quality, duplicate_quality], ignore_index=True)
        duplicate_minute = (
            duplicate_events
            .assign(minute=lambda frame: frame.timestamp.dt.floor("min"))
            .groupby("minute").size().rename("rows").reset_index()
        )
        adjustments = [
            duplicate_minute.assign(reason="near_duplicate"),
            duplicate_minute.assign(reason="valid", rows=lambda frame: -frame.rows),
        ]
        quality_minute = (
            pd.concat([quality_minute, *adjustments], ignore_index=True)
            .groupby(["minute", "reason"], as_index=False)["rows"].sum()
        )
    return valid, quality, quality_minute, int(near_duplicate.sum())


def add_elevation(frame: pd.DataFrame) -> pd.DataFrame:
    values = {}
    for col in ["longitude", "latitude", "satAltitude", "posLongitude", "posLatitude", "altitude"]:
        values[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
    sat_lon, sat_lat = np.radians(values["longitude"]), np.radians(values["latitude"])
    rx_lon, rx_lat = np.radians(values["posLongitude"]), np.radians(values["posLatitude"])

    def ecef(lon, lat, height):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        return np.column_stack(((n + height) * np.cos(lat) * np.cos(lon),
                                (n + height) * np.cos(lat) * np.sin(lon),
                                ((1 - WGS84_E2) * n + height) * np.sin(lat)))

    rho = ecef(sat_lon, sat_lat, values["satAltitude"]) - ecef(rx_lon, rx_lat, values["altitude"])
    east = -np.sin(rx_lon) * rho[:, 0] + np.cos(rx_lon) * rho[:, 1]
    north = (-np.sin(rx_lat) * np.cos(rx_lon) * rho[:, 0]
             - np.sin(rx_lat) * np.sin(rx_lon) * rho[:, 1] + np.cos(rx_lat) * rho[:, 2])
    up = (np.cos(rx_lat) * np.cos(rx_lon) * rho[:, 0]
          + np.cos(rx_lat) * np.sin(rx_lon) * rho[:, 1] + np.sin(rx_lat) * rho[:, 2])
    frame["elevation_deg"] = np.degrees(np.arctan2(up, np.hypot(east, north)))
    return frame


def iter_position_sources(early_db: Path, raw_csv: Path, cutover: pd.Timestamp, chunksize: int):
    with sqlite3.connect(f"file:{early_db}?mode=ro", uri=True) as conn:
        query = "SELECT * FROM position_data WHERE datetime(localTime) <= datetime(?) ORDER BY localTime, id"
        for frame in pd.read_sql_query(query, conn, params=[cutover.isoformat()], chunksize=chunksize):
            yield frame
    for frame in pd.read_csv(raw_csv, chunksize=chunksize, low_memory=False):
        timestamp = pd.to_datetime(frame["localTime"], errors="coerce", format="mixed")
        frame = frame.loc[timestamp > cutover].copy()
        if not frame.empty:
            yield frame


def load_position(early_db: Path, raw_csv: Path, cutover: pd.Timestamp, chunksize: int) -> pd.DataFrame:
    parts = []
    for frame in iter_position_sources(early_db, raw_csv, cutover, chunksize):
        frame["timestamp"] = pd.to_datetime(frame["localTime"], errors="coerce", format="mixed")
        sat = pd.to_numeric(frame["satId"], errors="coerce")
        valid = frame["timestamp"].notna() & sat.notna() & (sat != INVALID_SATELLITE_ID)
        for col in ["longitude", "latitude", "satAltitude", "posLongitude", "posLatitude", "altitude"]:
            number = pd.to_numeric(frame[col], errors="coerce")
            valid &= number.notna() & (number != 0)
        selected = frame.loc[valid, ["timestamp", "satId", "longitude", "latitude", "satAltitude", "posLongitude", "posLatitude", "altitude"]].copy()
        selected = add_elevation(selected)
        parts.append(selected.loc[selected["elevation_deg"] >= 0, ["timestamp", "satId", "elevation_deg"]])
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates(["timestamp", "satId"])


def segment_position(frame: pd.DataFrame, gap_s: float, min_points: int) -> pd.DataFrame:
    parts = []
    for satellite_id, rows in frame.groupby("satId", sort=False):
        rows = rows.sort_values("timestamp").copy()
        new = rows.timestamp.diff().dt.total_seconds().gt(gap_s).fillna(True)
        rows["segment"] = new.cumsum()
        grouped = rows.groupby("segment", sort=False).agg(
            pass_start=("timestamp", "first"), pass_end=("timestamp", "last"),
            position_points=("timestamp", "size"), max_elevation_deg=("elevation_deg", "max"),
        ).reset_index(drop=True)
        grouped["satellite_id"] = int(satellite_id)
        parts.append(grouped)
    result = pd.concat(parts, ignore_index=True)
    result["duration_s"] = (result.pass_end - result.pass_start).dt.total_seconds()
    return result[(result.position_points >= min_points) & (result.duration_s > 0)].sort_values("pass_start").reset_index(drop=True)


def segment_phy(frame: pd.DataFrame, gap_s: float = 60.0, min_points: int = 2) -> pd.DataFrame:
    parts = []
    for satellite_id, rows in frame.groupby("satelliteId", sort=False):
        rows = rows.sort_values("timestamp").copy()
        new = rows.timestamp.diff().dt.total_seconds().gt(gap_s).fillna(True)
        rows["segment"] = new.cumsum()
        grouped = rows.groupby("segment", sort=False).agg(
            pass_start=("timestamp", "first"), pass_end=("timestamp", "last"),
            actual_phy_points=("timestamp", "size"), mean_rssi=("rssi", "mean"),
            mean_phy_rssi=("phyRssi", "mean"), mean_snr=("snr", "mean"),
            max_internal_gap_s=("timestamp", lambda s: s.diff().dt.total_seconds().max() if len(s) > 1 else np.nan),
            median_sample_interval_s=("timestamp", lambda s: s.diff().dt.total_seconds().median() if len(s) > 1 else np.nan),
        ).reset_index(drop=True)
        grouped["satellite_id"] = int(satellite_id)
        parts.append(grouped)
    result = pd.concat(parts, ignore_index=True)
    result["duration_s"] = (result.pass_end - result.pass_start).dt.total_seconds()
    return result[(result.actual_phy_points >= min_points) & (result.duration_s > 0)].sort_values("pass_start").reset_index(drop=True)


def load_station(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        station = pd.read_sql_query(
            "SELECT datetime, rainfall FROM weather_station ORDER BY datetime", conn
        )
    station["timestamp"] = pd.to_datetime(station.pop("datetime"), errors="coerce")
    station["amount_mm"] = pd.to_numeric(station["rainfall"], errors="coerce") * 0.1
    return station.dropna(subset=["timestamp", "amount_mm"]).sort_values("timestamp")


def summarize_quality_by_rain(
    quality_minute: pd.DataFrame,
    station: pd.DataFrame,
) -> pd.DataFrame:
    station_minute = (
        station.assign(minute=station.timestamp.dt.floor("min"))
        .groupby("minute", as_index=False).amount_mm.last()
        .sort_values("minute")
    )
    aligned = pd.merge_asof(
        quality_minute.sort_values("minute"),
        station_minute,
        on="minute",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=90),
    ).dropna(subset=["amount_mm"])
    aligned["rain_rate_mm_h"] = aligned.amount_mm * 60.0
    aligned["month"] = aligned.minute.dt.strftime("%Y-%m")
    aligned["rain_rate_bin"] = pd.cut(
        aligned.rain_rate_mm_h,
        RAIN_EDGES,
        labels=RAIN_LABELS,
        include_lowest=True,
        right=False,
    )
    pivot = aligned.pivot_table(
        index=["month", "rain_rate_bin"],
        columns="reason",
        values="rows",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    total = pivot.sum(axis=1).clip(lower=1)
    no_lock = pivot.get("no_satellite_lock", pd.Series(0, index=pivot.index))
    duplicates = pivot.get("near_duplicate", pd.Series(0, index=pivot.index))
    valid = pivot.get("valid", pd.Series(0, index=pivot.index))
    invalid = total - no_lock - duplicates - valid
    return pd.DataFrame({
        "month": pivot.index.get_level_values("month").astype(str),
        "rain_rate_bin": pivot.index.get_level_values("rain_rate_bin").astype(str),
        "raw_rows": total.astype(int).to_numpy(),
        "valid_rows": valid.astype(int).to_numpy(),
        "no_satellite_lock_rows": no_lock.astype(int).to_numpy(),
        "invalid_payload_rows": invalid.clip(lower=0).astype(int).to_numpy(),
        "near_duplicate_rows": duplicates.astype(int).to_numpy(),
        "valid_rate": (valid / total).to_numpy(),
        "no_lock_rate": (no_lock / total).to_numpy(),
        "invalid_payload_rate": (invalid.clip(lower=0) / total).to_numpy(),
        "near_duplicate_rate": (duplicates / total).to_numpy(),
    })


def integrate_rain(station: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float | None, float | None]:
    timestamps = station["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    amounts = station["amount_mm"].to_numpy(np.float64)
    left = np.searchsorted(timestamps, start.value, side="right")
    right = np.searchsorted(timestamps, end.value + 60_000_000_000, side="right")
    total, covered = 0.0, 0.0
    for ts, amount in zip(timestamps[left:right], amounts[left:right]):
        overlap = max(0, min(end.value, int(ts)) - max(start.value, int(ts) - 60_000_000_000))
        if overlap:
            total += amount * overlap / 60_000_000_000
            covered += overlap
    duration = max((end - start).total_seconds(), 1e-9)
    if covered / 1e9 < duration * 0.9:
        return None, None
    return total, total * 3600.0 / duration


def indexed_arrays(frame: pd.DataFrame, satellite_col: str) -> dict[int, pd.DataFrame]:
    return {int(sat): rows.sort_values("timestamp") for sat, rows in frame.groupby(satellite_col)}


def count_window(rows: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if rows is None or rows.empty:
        return rows.iloc[0:0] if rows is not None else pd.DataFrame()
    times = rows["timestamp"].to_numpy(dtype="datetime64[ns]")
    left = np.searchsorted(times, np.datetime64(start), side="left")
    right = np.searchsorted(times, np.datetime64(end), side="right")
    return rows.iloc[left:right]


def build_pass_diagnostics(position_passes: pd.DataFrame, phy_sessions: pd.DataFrame, station: pd.DataFrame) -> pd.DataFrame:
    position_by_sat = {int(sat): rows for sat, rows in position_passes.groupby("satellite_id")}
    rows = []
    for item in phy_sessions.itertuples():
        actual = int(item.actual_phy_points)
        expected = max(int(np.floor(item.duration_s / 2.0)) + 1, 1)
        rainfall_mm, rain_rate = integrate_rain(station, item.pass_start, item.pass_end)
        visibility = position_by_sat.get(int(item.satellite_id))
        matched = None
        if visibility is not None:
            overlap = np.minimum(visibility.pass_end.values, np.datetime64(item.pass_end)) - np.maximum(visibility.pass_start.values, np.datetime64(item.pass_start))
            overlap_s = overlap.astype("timedelta64[ns]").astype(np.int64) / 1e9
            if len(overlap_s) and overlap_s.max() >= 0:
                matched = visibility.iloc[int(np.argmax(overlap_s))]
        visibility_duration = float(matched.duration_s) if matched is not None else np.nan
        opportunity_expected = max(int(np.floor(visibility_duration / 2.0)) + 1, 1) if pd.notna(visibility_duration) else np.nan
        rows.append({
            "month": item.pass_start.strftime("%Y-%m"),
            "satellite_id": int(item.satellite_id), "pass_start": item.pass_start,
            "pass_end": item.pass_end, "duration_s": item.duration_s,
            "visibility_start": matched.pass_start if matched is not None else pd.NaT,
            "visibility_end": matched.pass_end if matched is not None else pd.NaT,
            "visibility_duration_s": visibility_duration,
            "max_elevation_deg": float(matched.max_elevation_deg) if matched is not None else np.nan,
            "position_points": int(matched.position_points) if matched is not None else 0,
            "expected_phy_points": expected,
            "actual_phy_points": actual, "dropout_rate": max(1.0 - actual / expected, 0.0),
            "visibility_opportunity_expected_points": opportunity_expected,
            "visibility_opportunity_coverage": min(actual / opportunity_expected, 1.0) if pd.notna(opportunity_expected) else np.nan,
            "max_internal_gap_s": float(item.max_internal_gap_s),
            "median_sample_interval_s": float(item.median_sample_interval_s),
            "mean_rssi": float(item.mean_rssi), "mean_phy_rssi": float(item.mean_phy_rssi),
            "mean_snr": float(item.mean_snr),
            "rainfall_mm": rainfall_mm, "rain_rate_mm_h": rain_rate,
        })
    result = pd.DataFrame(rows)
    cadence = result.groupby("month").median_sample_interval_s.median().clip(lower=2.0)
    result["empirical_nominal_interval_s"] = result.month.map(cadence)
    result["expected_phy_points_empirical"] = (
        np.floor(result.duration_s / result.empirical_nominal_interval_s) + 1
    ).clip(lower=1).astype(int)
    result["dropout_rate_empirical"] = (
        1.0 - result.actual_phy_points / result.expected_phy_points_empirical
    ).clip(lower=0.0)
    dry_snr = result.loc[(result.rain_rate_mm_h.fillna(0) == 0) & (result.actual_phy_points > 0), "mean_snr"]
    weak_threshold = float(dry_snr.quantile(0.1)) if len(dry_snr) else np.nan
    dry_reference = (
        result.loc[result.rain_rate_mm_h.fillna(0) == 0]
        .groupby(["month", "satellite_id"])
        .agg(dry_dropout=("dropout_rate_empirical", "median"), dry_snr=("mean_snr", "median"))
    )
    keys = pd.MultiIndex.from_frame(result[["month", "satellite_id"]])
    result["dropout_excess_vs_sat_month_dry"] = keys.map(dry_reference.dry_dropout) - 0
    result["dropout_excess_vs_sat_month_dry"] = result.dropout_rate_empirical - result.dropout_excess_vs_sat_month_dry
    result["snr_delta_vs_sat_month_dry"] = result.mean_snr - keys.map(dry_reference.dry_snr)
    causes = []
    for row in result.itertuples():
        rainy = (row.rain_rate_mm_h or 0) > 0.1
        excess = row.dropout_excess_vs_sat_month_dry if pd.notna(row.dropout_excess_vs_sat_month_dry) else 0
        if rainy and row.dropout_rate_empirical >= 0.2 and excess > 0.05:
            cause = "rain_associated_partial_dropout"
        elif not rainy and row.dropout_rate_empirical >= 0.2 and pd.notna(row.mean_snr) and row.mean_snr <= weak_threshold:
            cause = "dry_weak_signal_interference_candidate"
        elif row.dropout_rate_empirical >= 0.2:
            cause = "unexplained_partial_dropout"
        else:
            cause = "normal_or_low_dropout"
        causes.append(cause)
    result["diagnostic_cause"] = causes
    return result


def build_network_gap_transitions(
    pass_rows: pd.DataFrame,
    station: pd.DataFrame,
) -> pd.DataFrame:
    """Measure gaps in the union of all satellite communication sessions."""
    ordered = pass_rows.sort_values(["pass_start", "pass_end"]).reset_index(drop=True)
    rows = []
    coverage_end = None
    coverage_boundary_satellite = None
    previous = None
    for current in ordered.itertuples():
        if previous is not None:
            pairwise_gap_s = max(
                (current.pass_start - previous.pass_end).total_seconds(), 0.0
            )
            network_gap_s = max(
                (current.pass_start - coverage_end).total_seconds(), 0.0
            )
            gap_start = coverage_end if network_gap_s > 0 else pd.NaT
            gap_end = current.pass_start if network_gap_s > 0 else pd.NaT
            gap_rainfall_mm, gap_rain_rate = (None, None)
            if network_gap_s > 0:
                gap_rainfall_mm, gap_rain_rate = integrate_rain(
                    station, coverage_end, current.pass_start
                )
            rows.append({
                "month": current.pass_start.strftime("%Y-%m"),
                "previous_session_satellite_id": int(previous.satellite_id),
                "previous_session_end": previous.pass_end,
                "next_session_satellite_id": int(current.satellite_id),
                "next_session_start": current.pass_start,
                "same_satellite_transition": bool(
                    previous.satellite_id == current.satellite_id
                ),
                "pairwise_gap_s": pairwise_gap_s,
                "coverage_boundary_satellite_id": int(
                    coverage_boundary_satellite
                ),
                "coverage_boundary_end": coverage_end,
                "network_gap_start": gap_start,
                "network_gap_end": gap_end,
                "network_gap_s": network_gap_s,
                "continuous_handover": bool(network_gap_s == 0),
                "covered_by_overlapping_satellite": bool(
                    pairwise_gap_s > 0 and network_gap_s == 0
                ),
                "outage_gt_1h": bool(network_gap_s > 3600),
                "gap_rainfall_mm": gap_rainfall_mm,
                "gap_mean_rain_rate_mm_h": gap_rain_rate,
            })
        if coverage_end is None or current.pass_end > coverage_end:
            coverage_end = current.pass_end
            coverage_boundary_satellite = current.satellite_id
        previous = current
    return pd.DataFrame(rows)


def summarize_network_continuity(
    pass_rows: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for month, sessions in pass_rows.groupby("month", sort=True):
        sessions = sessions.sort_values("pass_start")
        intervals = []
        for start, end in zip(sessions.pass_start, sessions.pass_end):
            if not intervals or start > intervals[-1][1]:
                intervals.append([start, end])
            elif end > intervals[-1][1]:
                intervals[-1][1] = end
        union_s = sum((end - start).total_seconds() for start, end in intervals)
        span_s = max(
            (sessions.pass_end.max() - sessions.pass_start.min()).total_seconds(),
            1.0,
        )
        month_transitions = transitions[transitions.month == month]
        positive = month_transitions.loc[
            month_transitions.network_gap_s > 0, "network_gap_s"
        ]
        rainy_positive = month_transitions.loc[
            (month_transitions.network_gap_s > 0)
            & (month_transitions.gap_mean_rain_rate_mm_h.fillna(0) > 0.1)
        ]
        rows.append({
            "month": month,
            "network_transition_count": int(len(month_transitions)),
            "different_satellite_transition_rate": float(
                (~month_transitions.same_satellite_transition).mean()
            ),
            "continuous_handover_count": int(
                month_transitions.continuous_handover.sum()
            ),
            "continuous_handover_rate": float(
                month_transitions.continuous_handover.mean()
            ),
            "median_network_gap_s": float(
                month_transitions.network_gap_s.median()
            ),
            "positive_network_gap_count": int(len(positive)),
            "median_positive_network_gap_s": float(positive.median()),
            "p90_positive_network_gap_s": float(positive.quantile(0.9)),
            "p95_positive_network_gap_s": float(positive.quantile(0.95)),
            "max_network_gap_s": float(positive.max()),
            "outage_gt_1h_count": int((positive > 3600).sum()),
            "rain_aligned_positive_gap_count": int(
                month_transitions.loc[month_transitions.network_gap_s > 0]
                .gap_mean_rain_rate_mm_h.notna().sum()
            ),
            "rainy_positive_gap_count": int(len(rainy_positive)),
            "union_communication_hours": union_s / 3600.0,
            "observed_span_hours": span_s / 3600.0,
            "link_time_coverage_rate": union_s / span_s,
        })
    return pd.DataFrame(rows)


def summarize(
    pass_rows: pd.DataFrame,
    position_passes: pd.DataFrame,
    quality: pd.DataFrame,
    phy: pd.DataFrame,
    continuity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = pass_rows.sort_values("pass_start").copy()
    same_sat = ordered.sort_values(["satellite_id", "pass_start"]).copy()
    same_sat["same_sat_revisit_h"] = same_sat.groupby("satellite_id").pass_start.diff().dt.total_seconds() / 3600
    ordered["same_sat_revisit_h"] = same_sat["same_sat_revisit_h"].reindex(ordered.index)
    monthly = ordered.groupby("month").agg(
        communication_sessions=("satellite_id", "size"), communicated_satellites=("satellite_id", "nunique"),
        mean_dropout_rate=("dropout_rate", "mean"),
        median_dropout_rate=("dropout_rate", "median"), p90_dropout_rate=("dropout_rate", lambda s: s.quantile(.9)),
        mean_dropout_rate_empirical=("dropout_rate_empirical", "mean"),
        median_dropout_rate_empirical=("dropout_rate_empirical", "median"),
        p90_dropout_rate_empirical=("dropout_rate_empirical", lambda s: s.quantile(.9)),
        communication_hours=("duration_s", lambda s: s.sum() / 3600),
        median_same_sat_communication_revisit_h=("same_sat_revisit_h", "median"),
        median_sample_interval_s=("median_sample_interval_s", "median"),
        rainy_passes=("rain_rate_mm_h", lambda s: int((s.fillna(0) > 0).sum())),
    ).reset_index()
    monthly = monthly.merge(continuity, on="month", how="left")
    monthly["median_communication_gap_min"] = monthly.median_network_gap_s / 60.0
    visibility = position_passes.assign(month=position_passes.pass_start.dt.strftime("%Y-%m"))
    visibility_monthly = visibility.groupby("month").agg(
        visibility_opportunities=("satellite_id", "size"), visible_satellites=("satellite_id", "nunique"),
        summed_visibility_hours=("duration_s", lambda s: s.sum() / 3600),
    )
    visibility_same = visibility.sort_values(["satellite_id", "pass_start"]).copy()
    visibility_same["revisit_h"] = visibility_same.groupby("satellite_id").pass_start.diff().dt.total_seconds() / 3600
    visibility_revisit = visibility_same.groupby("month").revisit_h.median()
    monthly = monthly.merge(visibility_monthly, left_on="month", right_index=True, how="left")
    monthly["median_same_sat_visibility_revisit_h"] = monthly.month.map(visibility_revisit)
    days = ordered.groupby("month").pass_start.agg(lambda s: (s.max().date() - s.min().date()).days + 1)
    monthly["communication_sessions_per_day"] = monthly.apply(lambda r: r.communication_sessions / max(days.get(r.month, 1), 1), axis=1)
    raw_totals = quality.groupby("month").rows.sum()
    no_lock = quality[quality.reason == "no_satellite_lock"].set_index("month").rows
    invalid = quality[
        ~quality.reason.isin(["valid", "no_satellite_lock", "near_duplicate"])
    ].groupby("month").rows.sum()
    duplicates = quality[quality.reason == "near_duplicate"].set_index("month").rows
    monthly["raw_phy_rows"] = monthly.month.map(raw_totals).fillna(0).astype(int)
    monthly["no_satellite_lock_rows"] = monthly.month.map(no_lock).fillna(0).astype(int)
    monthly["invalid_payload_rows"] = monthly.month.map(invalid).fillna(0).astype(int)
    monthly["near_duplicate_rows"] = monthly.month.map(duplicates).fillna(0).astype(int)
    monthly["valid_phy_rows"] = monthly.month.map(phy.assign(month=phy.timestamp.dt.strftime("%Y-%m")).groupby("month").size()).fillna(0).astype(int)
    monthly["no_lock_rate"] = monthly.no_satellite_lock_rows / monthly.raw_phy_rows.clip(lower=1)
    monthly["invalid_payload_rate"] = monthly.invalid_payload_rows / monthly.raw_phy_rows.clip(lower=1)
    monthly["near_duplicate_rate"] = monthly.near_duplicate_rows / monthly.raw_phy_rows.clip(lower=1)

    rain_bins = pass_rows.dropna(subset=["rain_rate_mm_h"]).copy()
    rain_bins["rain_rate_bin"] = pd.cut(rain_bins.rain_rate_mm_h, RAIN_EDGES, labels=RAIN_LABELS, include_lowest=True, right=False)
    rain_summary = rain_bins.groupby("rain_rate_bin", observed=False).agg(
        sample_count=("satellite_id", "size"), mean_dropout_rate=("dropout_rate", "mean"),
        median_dropout_rate=("dropout_rate", "median"), p90_dropout_rate=("dropout_rate", lambda s: s.quantile(.9)),
        mean_dropout_rate_empirical=("dropout_rate_empirical", "mean"),
        median_dropout_rate_empirical=("dropout_rate_empirical", "median"),
        p90_dropout_rate_empirical=("dropout_rate_empirical", lambda s: s.quantile(.9)),
        mean_snr=("mean_snr", "mean"), mean_rssi=("mean_rssi", "mean"),
        mean_dropout_excess_vs_sat_month_dry=("dropout_excess_vs_sat_month_dry", "mean"),
        mean_snr_delta_vs_sat_month_dry=("snr_delta_vs_sat_month_dry", "mean"),
    ).reset_index()

    cause = pass_rows.groupby(["month", "diagnostic_cause"]).size().rename("passes").reset_index()
    satellite = same_sat.groupby(["month", "satellite_id"]).agg(
        pass_count=("satellite_id", "size"), mean_dropout_rate=("dropout_rate", "mean"),
        mean_dropout_rate_empirical=("dropout_rate_empirical", "mean"),
        median_revisit_h=("same_sat_revisit_h", "median"),
        mean_snr=("mean_snr", "mean"), rainy_passes=("rain_rate_mm_h", lambda s: int((s.fillna(0) > 0).sum())),
    ).reset_index()
    return monthly, rain_summary, cause, satellite


def summarize_conditions(pass_rows: pd.DataFrame) -> pd.DataFrame:
    frame = pass_rows.copy()
    frame["hour_bin"] = pd.cut(
        frame.pass_start.dt.hour,
        [-1, 5, 11, 17, 23],
        labels=["00-05", "06-11", "12-17", "18-23"],
    )
    frame["elevation_bin"] = pd.cut(
        frame.max_elevation_deg,
        [-np.inf, 10, 20, 40, 60, np.inf],
        labels=["<10", "10-20", "20-40", "40-60", ">=60"],
    )
    rows = []
    for dimension, column in [("hour", "hour_bin"), ("max_elevation_deg", "elevation_bin")]:
        summary = frame.groupby(column, observed=False).agg(
            sample_count=("satellite_id", "size"),
            mean_dropout_empirical=("dropout_rate_empirical", "mean"),
            median_dropout_empirical=("dropout_rate_empirical", "median"),
            mean_snr=("mean_snr", "mean"),
            rainy_sessions=("rain_rate_mm_h", lambda values: int((values.fillna(0) > 0.1).sum())),
        ).reset_index(names="category")
        summary.insert(0, "dimension", dimension)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records", date_format="iso"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-db", type=Path, default=Path("/home/wdz/BT/db_backups/satellite_data_20260527_100641_before_clean.db"))
    parser.add_argument("--raw-phy-csv", type=Path, default=Path("/home/wdz/satellite_data/工控机采集的原始备份数据/phy_data.csv"))
    parser.add_argument("--raw-position-csv", type=Path, default=Path("/home/wdz/satellite_data/工控机采集的原始备份数据/position_data.csv"))
    parser.add_argument("--rain-db", type=Path, default=Path("/home/wdz/satellite_data/satellite_data.db"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phy_csv_min, phy_csv_max = csv_time_bounds(args.raw_phy_csv, "localTime", args.chunksize)
    position_csv_min, position_csv_max = csv_time_bounds(args.raw_position_csv, "localTime", args.chunksize)
    cutover = db_time_bound(args.early_db, "phy_data", "localTime")
    position_cutover = db_time_bound(args.early_db, "position_data", "localTime")
    print(f"phy_csv_range={phy_csv_min}..{phy_csv_max} position_csv_range={position_csv_min}..{position_csv_max}")
    phy, quality, quality_minute, near_duplicates = load_phy(
        args.early_db, args.raw_phy_csv, cutover, args.chunksize
    )
    print(f"valid_phy={len(phy)} near_duplicates={near_duplicates}")
    position = load_position(args.early_db, args.raw_position_csv, position_cutover, args.chunksize)
    print(f"valid_above_horizon_position={len(position)}")
    position_passes = segment_position(position, gap_s=60.0, min_points=5)
    phy_sessions = segment_phy(phy, gap_s=60.0, min_points=2)
    station = load_station(args.rain_db)
    passes = build_pass_diagnostics(position_passes, phy_sessions, station)
    transitions = build_network_gap_transitions(passes, station)
    continuity = summarize_network_continuity(passes, transitions)
    monthly, rain, cause, satellite = summarize(
        passes, position_passes, quality, phy, continuity
    )
    quality_rain = summarize_quality_by_rain(quality_minute, station)
    conditions = summarize_conditions(passes)

    provenance = []
    for role, path in [("early_raw_database", args.early_db), ("raw_phy_csv", args.raw_phy_csv),
                       ("raw_position_csv", args.raw_position_csv), ("rain_gauge_database", args.rain_db)]:
        provenance.append({"role": role, "path": str(path), "bytes": path.stat().st_size,
                           "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(), "sha256": sha256(path)})
    provenance.append({"role": "source_cutover", "phy_timestamp": cutover.isoformat(),
                       "position_timestamp": position_cutover.isoformat(),
                       "csv_phy_range": [phy_csv_min.isoformat(), phy_csv_max.isoformat()],
                       "csv_position_range": [position_csv_min.isoformat(), position_csv_max.isoformat()],
                       "rule": "pre-clean backup DB through its table maximum; unsorted raw CSV strictly after that maximum"})
    provenance.append({"role": "cleaning_rules", "path": "/home/wdz/satellite_data/server.py",
                       "rule": "clean_phy_payload and clean_position_payload as inspected on analysis date"})

    passes.to_csv(args.output_dir / "pass_diagnostics.csv", index=False)
    transitions.to_csv(args.output_dir / "inter_satellite_gaps.csv", index=False)
    continuity.to_csv(args.output_dir / "network_continuity_summary.csv", index=False)
    monthly.to_csv(args.output_dir / "monthly_summary.csv", index=False)
    rain.to_csv(args.output_dir / "rain_rate_summary.csv", index=False)
    quality.to_csv(args.output_dir / "raw_quality_summary.csv", index=False)
    quality_rain.to_csv(args.output_dir / "raw_quality_by_rain.csv", index=False)
    cause.to_csv(args.output_dir / "diagnostic_cause_summary.csv", index=False)
    satellite.to_csv(args.output_dir / "satellite_monthly_summary.csv", index=False)
    conditions.to_csv(args.output_dir / "condition_summary.csv", index=False)
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "provenance": provenance,
        "method": {"nominal_phy_interval_s": 2, "position_gap_s": 60, "minimum_position_points": 5,
                   "rainfall": "rainfall/10 integrated over preceding 60 s", "near_duplicates_removed": near_duplicates},
        "monthly": json_records(monthly), "network_continuity": json_records(continuity),
        "rain_rate": json_records(rain),
        "quality": json_records(quality), "quality_by_rain": json_records(quality_rain),
        "causes": json_records(cause),
        "satellites": json_records(satellite), "conditions": json_records(conditions),
    }
    (args.output_dir / "dashboard_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(monthly.to_string(index=False))
    print(rain.to_string(index=False))


if __name__ == "__main__":
    main()
