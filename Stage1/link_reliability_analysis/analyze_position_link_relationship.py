#!/usr/bin/env python3
"""Analyze PHY availability against satellite geometry from a traceable raw DB."""
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
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
VERSION_RANGES = (
    ("0401", pd.Timestamp.min, pd.Timestamp("2026-04-29 18:21:19.033281")),
    ("0429", pd.Timestamp("2026-04-29 18:21:19.033281"), pd.Timestamp("2026-05-27 10:38:04.238518")),
    ("0611", pd.Timestamp("2026-05-27 10:38:04.238518"), pd.Timestamp("2026-07-08 23:43:54.540569")),
    ("0727", pd.Timestamp("2026-07-08 23:43:54.540569"), pd.Timestamp.max),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_version(frame: pd.DataFrame) -> None:
    frame["let_version"] = ""
    for version, start, end in VERSION_RANGES:
        frame.loc[frame.timestamp.ge(start) & frame.timestamp.lt(end), "let_version"] = version


def load_mapping(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"let_version": str})
    frame = frame.drop_duplicates(["let_version", "raw_satellite_id"], keep="first")
    frame["mapping_status"] = frame.status.map({
        "accepted": "mapped", "provisional": "pending", "unresolved": "pending"
    })
    frame.loc[frame.status.ne("accepted"), ["norad_id", "canonical_0727_satellite_id"]] = np.nan
    return frame[[
        "let_version", "raw_satellite_id", "norad_id", "physical_name",
        "canonical_0727_satellite_id", "mapping_status",
    ]]


def classify_phy(frame: pd.DataFrame) -> pd.Series:
    reason = pd.Series("valid", index=frame.index, dtype="object")
    rules = [
        (frame.timestamp.isna(), "invalid_localTime"),
        (frame.satellite_id.isna(), "missing_satelliteId"),
        (frame.satellite_id.eq(INVALID_SATELLITE_ID), "no_satellite_lock"),
        (frame.rssi.isna(), "missing_rssi"),
        (frame.snr.isna(), "missing_snr"),
        (frame.snr.isin([0, 255]), "invalid_snr"),
        (frame.freq_offset.isna(), "missing_freqOffset"),
        (frame.freq_offset.eq(0), "invalid_freqOffset"),
        (frame.td.isna(), "missing_td"),
        (frame.td.eq(0), "invalid_td"),
    ]
    available = pd.Series(True, index=frame.index)
    for mask, label in rules:
        selected = available & mask
        reason.loc[selected] = label
        available.loc[selected] = False
    return reason


def load_phy(db_path: Path, mapping: pd.DataFrame, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_parts, quality_parts = [], []
    query = """
      SELECT id,localTime,satelliteId,rssi,phyRssi,lastCniValue,snr,freqOffset,td
      FROM phy_data ORDER BY id
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        for raw in pd.read_sql_query(query, connection, chunksize=chunksize):
            frame = pd.DataFrame({
                "source_id": raw.id,
                "timestamp": pd.to_datetime(raw.localTime, errors="coerce", format="mixed"),
                "satellite_id": pd.to_numeric(raw.satelliteId, errors="coerce"),
                "rssi": pd.to_numeric(raw.rssi, errors="coerce"),
                "phy_rssi": pd.to_numeric(raw.phyRssi, errors="coerce"),
                "cni": pd.to_numeric(raw.lastCniValue, errors="coerce"),
                "snr": pd.to_numeric(raw.snr, errors="coerce"),
                "freq_offset": pd.to_numeric(raw.freqOffset, errors="coerce"),
                "td": pd.to_numeric(raw.td, errors="coerce"),
            })
            frame["quality_reason"] = classify_phy(frame)
            frame["month"] = frame.timestamp.dt.strftime("%Y-%m")
            quality_parts.append(frame.groupby(["month", "quality_reason"], dropna=False).size().rename("rows").reset_index())
            valid_parts.append(frame.loc[frame.quality_reason.eq("valid")].drop(columns=["quality_reason", "month"]))
    valid = pd.concat(valid_parts, ignore_index=True).sort_values("timestamp")
    feature_columns = ["satellite_id", "rssi", "phy_rssi", "cni", "snr", "freq_offset", "td"]
    hashes = pd.util.hash_pandas_object(valid[feature_columns], index=False)
    duplicate = valid.groupby(hashes, sort=False).timestamp.diff().dt.total_seconds().between(0, 1)
    duplicate_month = valid.loc[duplicate].timestamp.dt.strftime("%Y-%m").value_counts()
    valid = valid.loc[~duplicate].copy()
    add_version(valid)
    valid = valid.merge(
        mapping, left_on=["let_version", "satellite_id"],
        right_on=["let_version", "raw_satellite_id"], how="left",
    )
    valid["identity_key"] = np.where(
        valid.mapping_status.eq("mapped"),
        "N" + valid.norad_id.astype("Int64").astype(str),
        valid.let_version + ":" + valid.satellite_id.astype("Int64").astype(str),
    )
    quality = pd.concat(quality_parts).groupby(["month", "quality_reason"], as_index=False).rows.sum()
    if len(duplicate_month):
        valid_rows = quality.quality_reason.eq("valid")
        quality.loc[valid_rows, "rows"] -= (
            quality.loc[valid_rows, "month"].map(duplicate_month).fillna(0).astype(int)
        )
        quality = pd.concat([
            quality,
            duplicate_month.rename_axis("month").rename("rows").reset_index().assign(quality_reason="near_duplicate"),
        ], ignore_index=True).groupby(["month", "quality_reason"], as_index=False).rows.sum()
    return valid, quality


def geodetic_to_ecef(lon_deg: np.ndarray, lat_deg: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.column_stack((
        (n + height_m) * np.cos(lat) * np.cos(lon),
        (n + height_m) * np.cos(lat) * np.sin(lon),
        ((1.0 - WGS84_E2) * n + height_m) * np.sin(lat),
    ))


def add_geometry(frame: pd.DataFrame) -> None:
    satellite = geodetic_to_ecef(
        frame.longitude.to_numpy(float), frame.latitude.to_numpy(float), frame.sat_altitude.to_numpy(float)
    )
    receiver = geodetic_to_ecef(
        frame.receiver_longitude.to_numpy(float), frame.receiver_latitude.to_numpy(float), frame.receiver_altitude.to_numpy(float)
    )
    rho = satellite - receiver
    rx_lon = np.radians(frame.receiver_longitude.to_numpy(float))
    rx_lat = np.radians(frame.receiver_latitude.to_numpy(float))
    east = -np.sin(rx_lon) * rho[:, 0] + np.cos(rx_lon) * rho[:, 1]
    north = (-np.sin(rx_lat) * np.cos(rx_lon) * rho[:, 0]
             - np.sin(rx_lat) * np.sin(rx_lon) * rho[:, 1] + np.cos(rx_lat) * rho[:, 2])
    up = (np.cos(rx_lat) * np.cos(rx_lon) * rho[:, 0]
          + np.cos(rx_lat) * np.sin(rx_lon) * rho[:, 1] + np.sin(rx_lat) * rho[:, 2])
    frame["computed_ecef_x_m"] = satellite[:, 0]
    frame["computed_ecef_y_m"] = satellite[:, 1]
    frame["computed_ecef_z_m"] = satellite[:, 2]
    frame["slant_range_km"] = np.linalg.norm(rho, axis=1) / 1000.0
    frame["elevation_deg"] = np.degrees(np.arctan2(up, np.hypot(east, north)))
    frame["azimuth_deg"] = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    raw_ecef = frame[["ecef_x_m", "ecef_y_m", "ecef_z_m"]].to_numpy(float)
    frame["ecef_residual_km"] = np.linalg.norm(raw_ecef - satellite, axis=1) / 1000.0


def load_position(db_path: Path, mapping: pd.DataFrame, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_parts, quality_parts = [], []
    query = """
      SELECT id,localTime,satId,longitude,latitude,satAltitude,posLongitude,posLatitude,
             altitude,ecefPx,ecefPy,ecefPz FROM position_data ORDER BY id
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        for raw in pd.read_sql_query(query, connection, chunksize=chunksize):
            frame = pd.DataFrame({
                "source_id": raw.id,
                "timestamp": pd.to_datetime(raw.localTime, errors="coerce", format="mixed"),
                "satellite_id": pd.to_numeric(raw.satId, errors="coerce"),
                "longitude": pd.to_numeric(raw.longitude, errors="coerce"),
                "latitude": pd.to_numeric(raw.latitude, errors="coerce"),
                "sat_altitude": pd.to_numeric(raw.satAltitude, errors="coerce"),
                "receiver_longitude": pd.to_numeric(raw.posLongitude, errors="coerce"),
                "receiver_latitude": pd.to_numeric(raw.posLatitude, errors="coerce"),
                "receiver_altitude": pd.to_numeric(raw.altitude, errors="coerce"),
                "ecef_x_m": pd.to_numeric(raw.ecefPx, errors="coerce"),
                "ecef_y_m": pd.to_numeric(raw.ecefPy, errors="coerce"),
                "ecef_z_m": pd.to_numeric(raw.ecefPz, errors="coerce"),
            })
            required = ["longitude", "latitude", "sat_altitude", "receiver_longitude", "receiver_latitude", "receiver_altitude"]
            reason = pd.Series("valid", frame.index, dtype="object")
            available = pd.Series(True, frame.index)
            rules = [(frame.timestamp.isna(), "invalid_localTime"), (frame.satellite_id.isna(), "missing_satId"),
                     (frame.satellite_id.eq(INVALID_SATELLITE_ID), "no_satellite_lock")]
            for column in required + ["ecef_x_m", "ecef_y_m", "ecef_z_m"]:
                rules.extend([(frame[column].isna(), f"missing_{column}"), (frame[column].eq(0), f"zero_{column}")])
            ecef_radius = np.sqrt(frame.ecef_x_m ** 2 + frame.ecef_y_m ** 2 + frame.ecef_z_m ** 2)
            rules.extend([
                (~frame.longitude.between(-180, 180), "longitude_out_of_range"),
                (~frame.latitude.between(-90, 90), "latitude_out_of_range"),
                (~frame.sat_altitude.between(1e5, 3e6), "satAltitude_out_of_leo_range"),
                (~ecef_radius.between(6.4e6, 1e7), "ecef_radius_out_of_range"),
            ])
            for mask, label in rules:
                selected = available & mask
                reason.loc[selected] = label
                available.loc[selected] = False
            frame["quality_reason"] = reason
            frame["month"] = frame.timestamp.dt.strftime("%Y-%m")
            quality_parts.append(frame.groupby(["month", "quality_reason"], dropna=False).size().rename("rows").reset_index())
            valid_parts.append(frame.loc[reason.eq("valid")].drop(columns=["quality_reason", "month"]))
    valid = pd.concat(valid_parts, ignore_index=True).sort_values("timestamp")
    add_geometry(valid)
    valid["ecef_consistent"] = valid.ecef_residual_km.le(10.0)
    valid = valid.loc[valid.elevation_deg.ge(0)].copy()
    valid = valid.sort_values(["satellite_id", "timestamp"])
    duplicate = valid.groupby("satellite_id").timestamp.diff().dt.total_seconds().between(0, 1)
    valid = valid.loc[~duplicate].copy()
    add_version(valid)
    valid = valid.merge(
        mapping, left_on=["let_version", "satellite_id"],
        right_on=["let_version", "raw_satellite_id"], how="left",
    )
    valid["identity_key"] = np.where(
        valid.mapping_status.eq("mapped"),
        "N" + valid.norad_id.astype("Int64").astype(str),
        valid.let_version + ":" + valid.satellite_id.astype("Int64").astype(str),
    )
    quality = pd.concat(quality_parts).groupby(["month", "quality_reason"], as_index=False).rows.sum()
    return valid.sort_values("timestamp"), quality


def segment_position(position: pd.DataFrame, gap_s: float, min_points: int) -> pd.DataFrame:
    rows = []
    aggregations = {
        "pass_start": ("timestamp", "first"), "pass_end": ("timestamp", "last"),
        "position_points": ("timestamp", "size"), "min_elevation_deg": ("elevation_deg", "min"),
        "mean_elevation_deg": ("elevation_deg", "mean"), "max_elevation_deg": ("elevation_deg", "max"),
        "mean_slant_range_km": ("slant_range_km", "mean"), "min_slant_range_km": ("slant_range_km", "min"),
        "max_slant_range_km": ("slant_range_km", "max"), "mean_azimuth_deg": ("azimuth_deg", "mean"),
        "longitude_deg": ("longitude", "mean"), "latitude_deg": ("latitude", "mean"),
        "altitude_km": ("sat_altitude", lambda x: x.mean() / 1000.0),
        "ecef_x_km": ("computed_ecef_x_m", lambda x: x.mean() / 1000.0),
        "ecef_y_km": ("computed_ecef_y_m", lambda x: x.mean() / 1000.0),
        "ecef_z_km": ("computed_ecef_z_m", lambda x: x.mean() / 1000.0),
        "ecef_residual_km": ("ecef_residual_km", "median"),
    }
    for identity, group in position.groupby("identity_key", sort=False):
        group = group.sort_values("timestamp").copy()
        group["segment"] = group.timestamp.diff().dt.total_seconds().gt(gap_s).fillna(True).cumsum()
        segmented = group.groupby("segment", sort=False).agg(**aggregations).reset_index(drop=True)
        first = group.groupby("segment", sort=False).first().reset_index(drop=True)
        for column in ["let_version", "satellite_id", "norad_id", "physical_name", "canonical_0727_satellite_id", "mapping_status"]:
            segmented[column] = first[column].to_numpy()
        segmented["identity_key"] = identity
        rows.append(segmented)
    result = pd.concat(rows, ignore_index=True)
    result["duration_s"] = (result.pass_end - result.pass_start).dt.total_seconds()
    return result.loc[(result.position_points >= min_points) & result.duration_s.gt(0)].sort_values("pass_start")


def attach_phy(passes: pd.DataFrame, phy: pd.DataFrame, nominal_interval_s: float) -> pd.DataFrame:
    indexed = {}
    for identity, rows in phy.groupby("identity_key", sort=False):
        ordered = rows.sort_values("timestamp")
        indexed[identity] = (ordered.timestamp.to_numpy(dtype="datetime64[ns]"), ordered)
    output = []
    for row in passes.itertuples(index=False):
        match = indexed.get(row.identity_key)
        selected = None
        if match is not None:
            times, source = match
            left = np.searchsorted(times, np.datetime64(row.pass_start), side="left")
            right = np.searchsorted(times, np.datetime64(row.pass_end), side="right")
            selected = source.iloc[left:right]
        actual = 0 if selected is None else len(selected)
        expected = max(int(np.floor(row.duration_s / nominal_interval_s)) + 1, 1)
        item = row._asdict()
        item.update({
            "visibility_expected_phy_points": expected, "visibility_actual_phy_points": actual,
            "visibility_coverage_rate": min(actual / expected, 1.0),
            "mean_rssi": float(selected.rssi.mean()) if actual else np.nan,
            "mean_phy_rssi": float(selected.phy_rssi.mean()) if actual else np.nan,
            "mean_snr": float(selected.snr.mean()) if actual else np.nan,
            "max_internal_gap_s": float(selected.timestamp.diff().dt.total_seconds().max()) if actual > 1 else np.nan,
        })
        output.append(item)
    result = pd.DataFrame(output)
    result["month"] = result.pass_start.dt.strftime("%Y-%m")
    result["communication_observed"] = result.visibility_actual_phy_points.gt(0)
    result["visibility_without_phy"] = result.visibility_actual_phy_points.eq(0)
    return result


def segment_phy_sessions(phy: pd.DataFrame, gap_s: float = 60.0,
                         min_points: int = 2) -> pd.DataFrame:
    rows = []
    for identity, group in phy.groupby("identity_key", sort=False):
        group = group.sort_values("timestamp").copy()
        group["segment"] = group.timestamp.diff().dt.total_seconds().gt(gap_s).fillna(True).cumsum()
        sessions = group.groupby("segment", sort=False).agg(
            pass_start=("timestamp", "first"), pass_end=("timestamp", "last"),
            actual_phy_points=("timestamp", "size"), mean_rssi=("rssi", "mean"),
            mean_phy_rssi=("phy_rssi", "mean"), mean_snr=("snr", "mean"),
            max_internal_gap_s=("timestamp", lambda x: x.diff().dt.total_seconds().max()),
        ).reset_index(drop=True)
        first = group.groupby("segment", sort=False).first().reset_index(drop=True)
        for column in ["let_version", "satellite_id", "norad_id", "physical_name", "canonical_0727_satellite_id", "mapping_status"]:
            sessions[column] = first[column].to_numpy()
        sessions["identity_key"] = identity
        rows.append(sessions)
    result = pd.concat(rows, ignore_index=True)
    result["duration_s"] = (result.pass_end - result.pass_start).dt.total_seconds()
    return result.loc[(result.actual_phy_points >= min_points) & result.duration_s.gt(0)].sort_values("pass_start")


def attach_position_to_phy_sessions(sessions: pd.DataFrame, position: pd.DataFrame,
                                    nominal_interval_s: float) -> pd.DataFrame:
    indexed = {}
    for identity, rows in position.groupby("identity_key", sort=False):
        ordered = rows.sort_values("timestamp")
        indexed[identity] = (ordered.timestamp.to_numpy(dtype="datetime64[ns]"), ordered)
    output = []
    for row in sessions.itertuples(index=False):
        match = indexed.get(row.identity_key)
        selected = None
        if match is not None:
            times, source = match
            left = np.searchsorted(times, np.datetime64(row.pass_start - pd.Timedelta(seconds=5)), side="left")
            right = np.searchsorted(times, np.datetime64(row.pass_end + pd.Timedelta(seconds=5)), side="right")
            selected = source.iloc[left:right]
        item = row._asdict()
        expected = max(int(np.floor(row.duration_s / nominal_interval_s)) + 1, 1)
        item.update({
            "expected_phy_points": expected,
            "dropout_rate": max(1.0 - row.actual_phy_points / expected, 0.0),
            "position_points": 0 if selected is None else len(selected),
            "position_matched": bool(selected is not None and len(selected)),
        })
        geometry_columns = {
            "min_elevation_deg": ("elevation_deg", "min"), "mean_elevation_deg": ("elevation_deg", "mean"),
            "max_elevation_deg": ("elevation_deg", "max"), "mean_slant_range_km": ("slant_range_km", "mean"),
            "min_slant_range_km": ("slant_range_km", "min"), "max_slant_range_km": ("slant_range_km", "max"),
            "mean_azimuth_deg": ("azimuth_deg", "mean"), "longitude_deg": ("longitude", "mean"),
            "latitude_deg": ("latitude", "mean"), "altitude_km": ("sat_altitude", lambda x: x.mean() / 1000.0),
            "ecef_x_km": ("computed_ecef_x_m", lambda x: x.mean() / 1000.0),
            "ecef_y_km": ("computed_ecef_y_m", lambda x: x.mean() / 1000.0),
            "ecef_z_km": ("computed_ecef_z_m", lambda x: x.mean() / 1000.0),
            "ecef_residual_km": ("ecef_residual_km", "median"),
        }
        for target, (source_column, operation) in geometry_columns.items():
            if selected is None or selected.empty:
                item[target] = np.nan
            elif callable(operation):
                item[target] = float(operation(selected[source_column]))
            else:
                item[target] = float(getattr(selected[source_column], operation)())
        output.append(item)
    result = pd.DataFrame(output)
    result["month"] = result.pass_start.dt.strftime("%Y-%m")
    return result


def aggregate_bins(passes: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "elevation_deg": pd.cut(passes.max_elevation_deg, [-1, 5, 10, 20, 30, 45, 60, 90], right=False),
        "slant_range_km": pd.cut(passes.mean_slant_range_km, [0, 1000, 1500, 2000, 3000, 5000, np.inf], right=False),
        "altitude_km": pd.cut(passes.altitude_km, [0, 500, 800, 1100, 1400, 2000, np.inf], right=False),
    }
    rows = []
    for dimension, categories in specs.items():
        summary = passes.assign(category=categories.astype(str)).groupby("category", observed=False).agg(
            pass_count=("identity_key", "size"), satellites=("identity_key", "nunique"),
            mean_dropout_rate=("dropout_rate", "mean"), median_dropout_rate=("dropout_rate", "median"),
            p90_dropout_rate=("dropout_rate", lambda x: x.quantile(.9)), mean_rssi=("mean_rssi", "mean"),
            mean_snr=("mean_snr", "mean"),
        ).reset_index()
        summary.insert(0, "dimension", dimension)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def satellite_summary(passes: pd.DataFrame) -> pd.DataFrame:
    return passes.groupby("identity_key", dropna=False).agg(
        latest_let_id=("canonical_0727_satellite_id", "first"), physical_norad_id=("norad_id", "first"),
        physical_name=("physical_name", "first"), mapping_status=("mapping_status", "first"),
        source_let_versions=("let_version", lambda x: ",".join(sorted(set(x)))),
        pass_count=("identity_key", "size"), first_pass=("pass_start", "min"), last_pass=("pass_end", "max"),
        position_matched_passes=("position_matched", "sum"),
        mean_dropout_rate=("dropout_rate", "mean"), median_dropout_rate=("dropout_rate", "median"),
        p90_dropout_rate=("dropout_rate", lambda x: x.quantile(.9)), mean_rssi=("mean_rssi", "mean"),
        mean_phy_rssi=("mean_phy_rssi", "mean"), mean_snr=("mean_snr", "mean"),
        median_max_elevation_deg=("max_elevation_deg", "median"), median_slant_range_km=("mean_slant_range_km", "median"),
        median_altitude_km=("altitude_km", "median"),
    ).reset_index().sort_values(["pass_count", "identity_key"], ascending=[False, True])


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records", date_format="iso"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-db", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--position-gap-s", type=float, default=60.0)
    parser.add_argument("--minimum-position-points", type=int, default=5)
    parser.add_argument("--nominal-phy-interval-s", type=float, default=2.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping_csv)
    phy, phy_quality = load_phy(args.raw_db, mapping, args.chunksize)
    position, position_quality = load_position(args.raw_db, mapping, args.chunksize)
    position_passes = segment_position(position, args.position_gap_s, args.minimum_position_points)
    visibility = attach_phy(position_passes, phy, args.nominal_phy_interval_s)
    phy_sessions = segment_phy_sessions(phy)
    all_sessions = attach_position_to_phy_sessions(phy_sessions, position, args.nominal_phy_interval_s)
    passes = all_sessions.loc[all_sessions.position_matched].copy()
    geometry = aggregate_bins(passes)
    satellites = satellite_summary(passes)
    monthly = passes.groupby("month").agg(
        phy_sessions=("identity_key", "size"), identities=("identity_key", "nunique"),
        physical_satellites=("norad_id", "nunique"), mean_dropout_rate=("dropout_rate", "mean"),
        median_dropout_rate=("dropout_rate", "median"), mean_rssi=("mean_rssi", "mean"),
        mean_snr=("mean_snr", "mean"), mean_max_elevation_deg=("max_elevation_deg", "mean"),
        mean_slant_range_km=("mean_slant_range_km", "mean"),
        mean_altitude_km=("altitude_km", "mean"),
    ).reset_index()

    passes.to_csv(args.output_dir / "position_link_passes.csv", index=False)
    visibility.to_csv(args.output_dir / "position_visibility_opportunities.csv", index=False)
    satellites.to_csv(args.output_dir / "satellite_position_link_summary.csv", index=False)
    geometry.to_csv(args.output_dir / "geometry_dropout_summary.csv", index=False)
    monthly.to_csv(args.output_dir / "position_link_monthly_summary.csv", index=False)
    phy_quality.to_csv(args.output_dir / "phy_quality_summary.csv", index=False)
    position_quality.to_csv(args.output_dir / "position_quality_summary.csv", index=False)
    with sqlite3.connect(args.output_dir / "position_link_analysis.sqlite3") as connection:
        passes.to_sql("position_link_passes", connection, if_exists="replace", index=False)
        satellites.to_sql("satellite_summary", connection, if_exists="replace", index=False)
        geometry.to_sql("geometry_summary", connection, if_exists="replace", index=False)
        phy_quality.to_sql("phy_quality", connection, if_exists="replace", index=False)
        position_quality.to_sql("position_quality", connection, if_exists="replace", index=False)
        connection.executescript("""
          CREATE INDEX IF NOT EXISTS idx_position_link_identity_time ON position_link_passes(identity_key,pass_start);
          CREATE INDEX IF NOT EXISTS idx_position_link_norad_time ON position_link_passes(norad_id,pass_start);
        """)

    dashboard = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provenance": {
            "raw_database": str(args.raw_db.resolve()), "raw_database_sha256": sha256(args.raw_db),
            "mapping_csv": str(args.mapping_csv.resolve()), "mapping_sha256": sha256(args.mapping_csv),
            "source_database_recovered_with_sqlite_recover": True,
        },
        "method": {
            "pass_definition": "same identity, consecutive valid above-horizon positions with <=60 s gaps",
            "dropout_definition": "1 - actual same-identity PHY points / floor(position-window duration / 2 s + 1)",
            "position_quality": "server.py payload rules audited separately; geometry requires valid WGS-84 geodetic and ECEF fields",
        },
        "overview": {
            "valid_phy_rows": len(phy), "valid_position_rows_above_horizon": len(position),
            "position_opportunities": len(visibility),
            "visibility_opportunities_with_phy": int(visibility.communication_observed.sum()),
            "visibility_opportunities_without_phy": int(visibility.visibility_without_phy.sum()),
            "phy_sessions": len(all_sessions), "position_matched_phy_sessions": len(passes),
            "identified_physical_satellites": int(passes.norad_id.nunique()),
            "all_identity_keys": int(passes.identity_key.nunique()),
            "mean_dropout_rate": float(passes.dropout_rate.mean()),
        },
        "monthly": records(monthly), "geometry_summary": records(geometry), "satellites": records(satellites),
        "phy_quality": records(phy_quality), "position_quality": records(position_quality),
    }
    (args.output_dir / "position_link_dashboard.json").write_text(
        json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(dashboard["overview"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
