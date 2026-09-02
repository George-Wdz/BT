"""Build one multi-satellite sample for every valid rain-gauge minute."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LINK_COLUMNS = ["phyRssi", "rssi", "snr", "lastCniValue"]
GEO_COLUMNS = ["slant_range_km", "elevation_deg", "azimuth_sin", "azimuth_cos"]
WEATHER_COLUMNS = ["temperature", "humidity", "pressure"]
IMAGE_COLUMNS = ["prob_sunny", "prob_cloudy", "prob_rain", "image_available"]
TIME_COLUMNS = ["relative_time"]
BASE_FEATURE_COLUMNS = LINK_COLUMNS + GEO_COLUMNS + WEATHER_COLUMNS + IMAGE_COLUMNS + TIME_COLUMNS
INVALID_SATELLITE_ID = 4294967295
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class BuildConfig:
    db_path: str
    output_path: str
    terminal_id: str = "01-31-0005-0001"
    image_csv: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    window_seconds: float = 60.0
    rainfall_scale: float = 0.1
    min_phy_points: int = 10
    min_snr_db: float | None = None
    position_mode: str = "required"
    position_tolerance_seconds: float = 5.0
    weather_tolerance_seconds: float = 60.0
    image_tolerance_seconds: float = 600.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    split_strategy: str = "stratified_all"
    split_seed: int = 42
    holdout_periods: tuple[tuple[str, str], ...] = ()
    holdout_buffer_minutes: float = 60.0
    terminal_protocol: str = "legacy"
    shared_db_path: str | None = None
    shared_terminal_id: str = "01-31-0005-0001"
    adapter_path: str | None = None
    reference_checkpoint_path: str | None = None


def _read_sql(db_path: str, query: str, params: list[object]) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(query, conn, params=params)


def _time_predicates(column: str, cfg: BuildConfig) -> tuple[list[str], list[object]]:
    clauses = ["terminalId = ?"]
    params: list[object] = [cfg.terminal_id]
    if cfg.start_time:
        clauses.append(f"datetime({column}) >= datetime(?)")
        params.append(cfg.start_time)
    if cfg.end_time:
        clauses.append(f"datetime({column}) <= datetime(?)")
        params.append(cfg.end_time)
    return clauses, params


def _reference_feature_statistics(path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["transforms"]
    return (
        state["feature_mean"].cpu().numpy().astype(np.float64),
        state["feature_std"].cpu().numpy().astype(np.float64),
    )


def _load_new_terminal_phy(cfg: BuildConfig) -> pd.DataFrame:
    if not cfg.adapter_path or not cfg.reference_checkpoint_path:
        raise ValueError(
            "new terminal protocol requires adapter_path and reference_checkpoint_path"
        )
    clauses = [
        "b.terminalId = ?", "b.validMeasBb = 1", "r.validMeasRssi = 1",
        "b.trackNo IS NOT NULL", "b.phaseNo IS NOT NULL",
        "b.snr IS NOT NULL", "b.snr != 0",
        "r.chanRssi IS NOT NULL", "r.chanRssi != 0",
        "r.carrRssi IS NOT NULL", "r.carrRssi != 0",
    ]
    params: list[object] = [cfg.terminal_id]
    if cfg.start_time:
        clauses.append("b.localTime >= ?")
        params.append(cfg.start_time)
    if cfg.end_time:
        clauses.append("b.localTime <= ?")
        params.append(cfg.end_time)
    if cfg.min_snr_db is not None:
        clauses.append("b.snr >= ?")
        params.append(float(cfg.min_snr_db))
    frame = _read_sql(
        cfg.db_path,
        f"""
        SELECT b.localTime, b.trackNo, b.phaseNo, b.snr,
               r.chanRssi, r.carrRssi
        FROM phy_bb_data AS b
        JOIN phy_rssi_data AS r
          ON r.terminalId = b.terminalId AND r.localTime = b.localTime
        WHERE {' AND '.join(clauses)}
        ORDER BY b.localTime
        """,
        params,
    )
    if frame.empty:
        return pd.DataFrame(columns=["localTime", "satelliteId", *LINK_COLUMNS])
    adapter = json.loads(Path(cfg.adapter_path).read_text(encoding="utf-8"))
    source_columns = list(adapter["source_columns"])
    source = np.column_stack([
        pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        for column in source_columns
    ])
    source_mean = np.asarray(adapter["source_mean"], dtype=np.float64)
    source_scale = np.asarray(adapter["source_scale"], dtype=np.float64)
    target_mean, target_scale = _reference_feature_statistics(
        cfg.reference_checkpoint_path
    )
    standardized = np.clip(
        (source - source_mean.reshape(1, -1)) / source_scale.reshape(1, -1),
        -float(adapter.get("clip_z", 5.0)),
        float(adapter.get("clip_z", 5.0)),
    )
    mapped = target_mean[:4].reshape(1, -1) + standardized * target_scale[:4].reshape(1, -1)
    frame["satelliteId"] = (
        frame["trackNo"].astype("int64") * 256
        + frame["phaseNo"].astype("int64")
    )
    for index, column in enumerate(LINK_COLUMNS):
        frame[column] = mapped[:, index]
    return frame[["localTime", "satelliteId", *LINK_COLUMNS]]


def load_sources(cfg: BuildConfig) -> tuple[pd.DataFrame, ...]:
    if cfg.terminal_protocol == "new":
        phy = _load_new_terminal_phy(cfg)
        shared_db = cfg.shared_db_path
        if not shared_db:
            raise ValueError("new terminal protocol requires shared_db_path")
        shared_cfg = BuildConfig(
            db_path=shared_db,
            output_path=cfg.output_path,
            terminal_id=cfg.shared_terminal_id,
            start_time=cfg.start_time,
            end_time=cfg.end_time,
        )
        _, pos, weather, gauge = load_sources(shared_cfg)
        phy["localTime"] = pd.to_datetime(phy["localTime"], errors="coerce")
        phy.dropna(subset=["localTime"], inplace=True)
        return phy, pos, weather, gauge
    if cfg.terminal_protocol != "legacy":
        raise ValueError(f"unsupported terminal protocol: {cfg.terminal_protocol}")
    phy_where, phy_params = _time_predicates("localTime", cfg)
    phy_where.extend([
        "satelliteId != ?", "snr != 255",
        *[f"{column} IS NOT NULL" for column in LINK_COLUMNS],
    ])
    phy_params.append(INVALID_SATELLITE_ID)
    if cfg.min_snr_db is not None:
        phy_where.append("snr >= ?")
        phy_params.append(float(cfg.min_snr_db))
    phy = _read_sql(
        cfg.db_path,
        f"""SELECT localTime, satelliteId, {', '.join(LINK_COLUMNS)}
            FROM phy_data WHERE {' AND '.join(phy_where)} ORDER BY localTime""",
        phy_params,
    )

    pos_where, pos_params = _time_predicates("localTime", cfg)
    pos_where.extend([
        "satId != ?", "longitude IS NOT NULL", "longitude != 0",
        "latitude IS NOT NULL", "latitude != 0", "satAltitude IS NOT NULL",
        "satAltitude BETWEEN 100000 AND 3000000",
        "posLongitude IS NOT NULL", "posLongitude != 0",
        "posLatitude IS NOT NULL", "posLatitude != 0",
        "altitude IS NOT NULL", "altitude != 0",
        "ecefPx IS NOT NULL", "ecefPx != 0",
        "ecefPy IS NOT NULL", "ecefPy != 0",
        "ecefPz IS NOT NULL", "ecefPz != 0",
        "(ecefPx * ecefPx + ecefPy * ecefPy + ecefPz * ecefPz) "
        "BETWEEN 40960000000000 AND 100000000000000",
    ])
    pos_params.append(INVALID_SATELLITE_ID)
    pos = _read_sql(
        cfg.db_path,
        f"""SELECT localTime, satId, longitude, latitude, satAltitude,
                   posLongitude, posLatitude, altitude, ecefPx, ecefPy, ecefPz
            FROM position_data WHERE {' AND '.join(pos_where)} ORDER BY localTime""",
        pos_params,
    )

    weather_where, weather_params = _time_predicates("timestamp", cfg)
    weather = _read_sql(
        cfg.db_path,
        f"""SELECT timestamp, temperature, humidity, pressure
            FROM weather_data WHERE {' AND '.join(weather_where)} ORDER BY timestamp""",
        weather_params,
    )

    gauge_where, gauge_params = _time_predicates("datetime", cfg)
    gauge_where.extend(["rainfall IS NOT NULL", "rainfall >= 0"])
    gauge = _read_sql(
        cfg.db_path,
        f"""SELECT datetime AS timestamp, rainfall
            FROM weather_station WHERE {' AND '.join(gauge_where)} ORDER BY datetime""",
        gauge_params,
    )

    for frame, column in ((phy, "localTime"), (pos, "localTime"),
                          (weather, "timestamp"), (gauge, "timestamp")):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame.dropna(subset=[column], inplace=True)
    return phy, pos, weather, gauge


def add_geometry(position: pd.DataFrame) -> pd.DataFrame:
    out = position.copy()
    lon = np.deg2rad(out["posLongitude"].to_numpy(np.float64))
    lat = np.deg2rad(out["posLatitude"].to_numpy(np.float64))
    height = out["altitude"].to_numpy(np.float64)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    ground = np.column_stack([
        (prime_vertical + height) * cos_lat * np.cos(lon),
        (prime_vertical + height) * cos_lat * np.sin(lon),
        (prime_vertical * (1.0 - WGS84_E2) + height) * sin_lat,
    ])
    satellite = out[["ecefPx", "ecefPy", "ecefPz"]].to_numpy(np.float64)
    relative = satellite - ground
    slant = np.linalg.norm(relative, axis=1)
    east = relative[:, 0] * -np.sin(lon) + relative[:, 1] * np.cos(lon)
    north = (relative[:, 0] * -sin_lat * np.cos(lon)
             + relative[:, 1] * -sin_lat * np.sin(lon)
             + relative[:, 2] * cos_lat)
    up = (relative[:, 0] * cos_lat * np.cos(lon)
          + relative[:, 1] * cos_lat * np.sin(lon)
          + relative[:, 2] * sin_lat)
    azimuth = np.arctan2(east, north)
    out["slant_range_km"] = slant / 1000.0
    out["elevation_deg"] = np.rad2deg(np.arcsin(np.clip(up / np.maximum(slant, 1e-6), -1, 1)))
    out["azimuth_sin"] = np.sin(azimuth)
    out["azimuth_cos"] = np.cos(azimuth)
    radius = np.linalg.norm(satellite, axis=1)
    valid = (radius >= 6.4e6) & (radius <= 1.0e7) & np.isfinite(out[GEO_COLUMNS]).all(axis=1)
    return out.loc[valid].copy()


def match_position_same_satellite(
    phy: pd.DataFrame, position: pd.DataFrame, tolerance_seconds: float
) -> pd.DataFrame:
    """Nearest-time match constrained by satellite ID; cross-ID matches are impossible."""
    matched: list[pd.DataFrame] = []
    position_groups = {
        int(sat_id): group.sort_values("localTime")
        for sat_id, group in position.groupby("satId")
    }
    tolerance = pd.Timedelta(seconds=tolerance_seconds)
    for sat_id, link_group in phy.groupby("satelliteId"):
        sat_position = position_groups.get(int(sat_id))
        if sat_position is None:
            continue
        right = sat_position[["localTime", *GEO_COLUMNS]].rename(
            columns={"localTime": "position_time"}
        )
        merged = pd.merge_asof(
            link_group.sort_values("localTime"), right,
            left_on="localTime", right_on="position_time",
            direction="nearest", tolerance=tolerance,
        )
        merged = merged.dropna(subset=GEO_COLUMNS)
        merged["position_lag_seconds"] = (
            merged["localTime"] - merged["position_time"]
        ).abs().dt.total_seconds()
        matched.append(merged)
    if not matched:
        return pd.DataFrame(columns=[*phy.columns, "position_time", *GEO_COLUMNS])
    return pd.concat(matched, ignore_index=True).sort_values("localTime")


def match_position_with_fallback(
    phy: pd.DataFrame,
    position: pd.DataFrame,
    tolerance_seconds: float,
    fallback: np.ndarray,
) -> pd.DataFrame:
    """Same-satellite nearest match with checkpoint-mean geometry fallback."""
    matched: list[pd.DataFrame] = []
    position_groups = {
        int(sat_id): group.sort_values("localTime")
        for sat_id, group in position.groupby("satId")
    }
    tolerance = pd.Timedelta(seconds=tolerance_seconds)
    for sat_id, link_group in phy.groupby("satelliteId"):
        sat_position = position_groups.get(int(sat_id))
        if sat_position is None:
            merged = link_group.copy()
            merged["position_time"] = pd.NaT
            for column, value in zip(GEO_COLUMNS, fallback):
                merged[column] = float(value)
        else:
            right = sat_position[["localTime", *GEO_COLUMNS]].rename(
                columns={"localTime": "position_time"}
            )
            merged = pd.merge_asof(
                link_group.sort_values("localTime"), right,
                left_on="localTime", right_on="position_time",
                direction="nearest", tolerance=tolerance,
            )
            merged[GEO_COLUMNS] = merged[GEO_COLUMNS].fillna(
                dict(zip(GEO_COLUMNS, fallback.tolist()))
            )
        merged["position_lag_seconds"] = (
            merged["localTime"] - merged["position_time"]
        ).abs().dt.total_seconds()
        matched.append(merged)
    if not matched:
        return pd.DataFrame(columns=[*phy.columns, "position_time", *GEO_COLUMNS])
    return pd.concat(matched, ignore_index=True).sort_values("localTime")


def _clean_weather(weather: pd.DataFrame) -> pd.DataFrame:
    out = weather.copy()
    for column in WEATHER_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    hpa = out["pressure"].between(800, 1100)
    out.loc[hpa, "pressure"] /= 10.0
    valid = (out["temperature"].between(-10, 45)
             & out["humidity"].between(0, 100)
             & out["pressure"].between(95, 105))
    return out.loc[valid].sort_values("timestamp")


def attach_weather(phy: pd.DataFrame, weather: pd.DataFrame, tolerance_seconds: float) -> pd.DataFrame:
    return pd.merge_asof(
        phy.sort_values("localTime"), _clean_weather(weather),
        left_on="localTime", right_on="timestamp", direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    ).dropna(subset=WEATHER_COLUMNS)


def load_images(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    images = pd.read_csv(path)
    images["timestamp"] = pd.to_datetime(images["timestamp"], errors="coerce")
    for column in IMAGE_COLUMNS[:-1]:
        images[column] = pd.to_numeric(images[column], errors="coerce")
    images["image_available"] = 1.0
    return images.dropna(subset=["timestamp", *IMAGE_COLUMNS[:-1]]).sort_values("timestamp")


def build_samples(cfg: BuildConfig) -> tuple[list[dict], dict]:
    phy, position, weather, gauge = load_sources(cfg)
    if cfg.position_mode == "required":
        position = add_geometry(position)
        position_aligned = match_position_same_satellite(
            phy, position, cfg.position_tolerance_seconds
        )
        geometry_columns = GEO_COLUMNS
    elif cfg.position_mode == "fallback_mean":
        if not cfg.reference_checkpoint_path:
            raise ValueError("fallback_mean requires reference_checkpoint_path")
        position = add_geometry(position)
        feature_mean, _ = _reference_feature_statistics(cfg.reference_checkpoint_path)
        position_aligned = match_position_with_fallback(
            phy, position, cfg.position_tolerance_seconds, feature_mean[4:8]
        )
        geometry_columns = GEO_COLUMNS
    elif cfg.position_mode == "omit":
        position_aligned = phy.copy()
        geometry_columns = []
    else:
        raise ValueError(f"Unsupported position_mode: {cfg.position_mode}")
    aligned = attach_weather(position_aligned, weather, cfg.weather_tolerance_seconds)
    feature_columns = LINK_COLUMNS + geometry_columns + WEATHER_COLUMNS + IMAGE_COLUMNS + TIME_COLUMNS
    images = load_images(cfg.image_csv)

    anchors = gauge.sort_values("timestamp").reset_index(drop=True)
    anchor_ns = anchors["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    link_ns = aligned["localTime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    image_ns = None if images is None else images["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    window_ns = int(cfg.window_seconds * 1e9)
    image_tol_ns = int(cfg.image_tolerance_seconds * 1e9)
    samples: list[dict] = []

    for row_index, anchor in anchors.iterrows():
        end_ns = anchor_ns[row_index]
        start_ns = end_ns - window_ns
        left = int(np.searchsorted(link_ns, start_ns, side="right"))
        right = int(np.searchsorted(link_ns, end_ns, side="right"))
        window = aligned.iloc[left:right]
        if len(window) < cfg.min_phy_points:
            continue

        image_vector = np.zeros(4, dtype=np.float32)
        if images is not None and image_ns is not None and len(images):
            image_index = int(np.searchsorted(image_ns, end_ns))
            candidates = [idx for idx in (image_index - 1, image_index) if 0 <= idx < len(images)]
            nearest = min(candidates, key=lambda idx: abs(int(image_ns[idx]) - end_ns))
            if abs(int(image_ns[nearest]) - end_ns) <= image_tol_ns:
                image_vector = images.iloc[nearest][IMAGE_COLUMNS].to_numpy(np.float32)

        numeric = window[LINK_COLUMNS + geometry_columns + WEATHER_COLUMNS].to_numpy(np.float32)
        image_features = np.repeat(image_vector[None, :], len(window), axis=0)
        relative_time = ((window["localTime"].astype("int64").to_numpy() - end_ns)
                         / float(window_ns)).astype(np.float32)[:, None]
        samples.append({
            "features": np.concatenate([numeric, image_features, relative_time], axis=1),
            "satellite_ids": window["satelliteId"].to_numpy(np.int64),
            "timestamps_ns": window["localTime"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
            "minute_rainfall_mm": np.float32(float(anchor["rainfall"]) * cfg.rainfall_scale),
            "gauge_raw_rainfall": np.float32(anchor["rainfall"]),
            "anchor_time_ns": np.int64(end_ns),
            "window_start_ns": np.int64(start_ns),
            "point_count": np.int32(len(window)),
            "satellite_count": np.int16(window["satelliteId"].nunique()),
        })

    n_samples = len(samples)
    if cfg.split_strategy == "event_holdout":
        if not cfg.holdout_periods:
            raise ValueError("event_holdout requires at least one holdout period")
        timestamps = pd.to_datetime(
            [int(sample["anchor_time_ns"]) for sample in samples], unit="ns"
        )
        holdout = np.zeros(n_samples, dtype=bool)
        buffer = pd.Timedelta(minutes=cfg.holdout_buffer_minutes)
        for start, end in cfg.holdout_periods:
            start_ts = pd.Timestamp(start) - buffer
            end_ts = pd.Timestamp(end) + buffer
            holdout |= (timestamps >= start_ts) & (timestamps < end_ts)

        # The event windows are the only test set. Non-event samples are split
        # by rainfall class into train/validation, preserving the class ratio.
        splits = np.full(n_samples, "test", dtype="<U5")
        non_holdout = np.flatnonzero(~holdout)
        labels = np.asarray([
            samples[index]["minute_rainfall_mm"] > 0 for index in non_holdout
        ])
        rng = np.random.default_rng(cfg.split_seed)
        train_fraction = cfg.train_ratio / (cfg.train_ratio + cfg.val_ratio)
        for class_value in (False, True):
            indices = non_holdout[labels == class_value]
            rng.shuffle(indices)
            train_end = int(len(indices) * train_fraction)
            splits[indices[:train_end]] = "train"
            splits[indices[train_end:]] = "val"
    elif cfg.split_strategy == "time":
        train_end = int(n_samples * cfg.train_ratio)
        val_end = int(n_samples * (cfg.train_ratio + cfg.val_ratio))
        splits = np.array(["train"] * train_end + ["val"] * (val_end - train_end)
                          + ["test"] * (n_samples - val_end))
    elif cfg.split_strategy == "stratified_all":
        splits = np.full(n_samples, "test", dtype="<U5")
        rng = np.random.default_rng(cfg.split_seed)
        labels = np.asarray([sample["minute_rainfall_mm"] > 0 for sample in samples])
        for class_value in (False, True):
            indices = np.flatnonzero(labels == class_value)
            rng.shuffle(indices)
            train_end = int(len(indices) * cfg.train_ratio)
            val_end = int(len(indices) * (cfg.train_ratio + cfg.val_ratio))
            splits[indices[:train_end]] = "train"
            splits[indices[train_end:val_end]] = "val"
    else:
        raise ValueError(f"Unsupported split strategy: {cfg.split_strategy}")
    summary = {
        "sample_unit": "one gauge-anchored preceding-minute window",
        "target": "rainfall / 10 in mm; one scalar per window",
        "window_semantics": "(anchor_time - 60 s, anchor_time]",
        "position_alignment": (
            "same satellite ID plus nearest timestamp"
            if cfg.position_mode == "required"
            else (
                "same satellite ID plus nearest timestamp, checkpoint-mean fallback"
                if cfg.position_mode == "fallback_mean" else "omitted"
            )
        ),
        "base_feature_columns": feature_columns,
        "base_feature_dim": len(feature_columns),
        "dry_delta_added_at_training": True,
        "model_input_dim_with_dry_delta": len(feature_columns) + len(LINK_COLUMNS),
        "source_rows": {"phy": len(phy), "position": len(position), "weather": len(weather), "gauge": len(gauge)},
        "position_matched_phy_rows": int(
            position_aligned["position_time"].notna().sum()
            if "position_time" in position_aligned else len(position_aligned)
        ),
        "position_fallback_phy_rows": int(
            position_aligned["position_time"].isna().sum()
            if "position_time" in position_aligned else 0
        ),
        "position_lag_seconds": (
            {
                "median": float(position_aligned["position_lag_seconds"].median()),
                "p95": float(position_aligned["position_lag_seconds"].quantile(0.95)),
                "max": float(position_aligned["position_lag_seconds"].max()),
            }
            if cfg.position_mode in ("required", "fallback_mean") else None
        ),
        "aligned_phy_rows": len(aligned),
        "samples": n_samples,
        "rainy_samples": int(sum(sample["minute_rainfall_mm"] > 0 for sample in samples)),
        "split_counts": {name: int((splits == name).sum()) for name in ("train", "val", "test")},
        "rainy_split_counts": {
            name: int(sum(sample["minute_rainfall_mm"] > 0
                          for sample, split in zip(samples, splits) if split == name))
            for name in ("train", "val", "test")
        },
        "config": cfg.__dict__,
    }
    return samples, {"splits": splits, "summary": summary, "feature_columns": feature_columns}


def save_dataset(samples: list[dict], metadata: dict, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        samples=np.asarray(samples, dtype=object),
        splits=metadata["splits"],
        feature_columns=np.asarray(metadata["feature_columns"]),
        summary_json=np.asarray(json.dumps(metadata["summary"], ensure_ascii=False)),
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(metadata["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    index_rows = []
    for sample, split in zip(samples, metadata["splits"]):
        index_rows.append({
            "anchor_time": pd.to_datetime(int(sample["anchor_time_ns"]), unit="ns"),
            "window_start": pd.to_datetime(int(sample["window_start_ns"]), unit="ns"),
            "split": split,
            "minute_rainfall_mm": f"{float(sample['minute_rainfall_mm']):.2f}",
            "gauge_raw_rainfall": float(sample["gauge_raw_rainfall"]),
            "phy_point_count": int(sample["point_count"]),
            "satellite_count": int(sample["satellite_count"]),
        })
    pd.DataFrame(index_rows).to_csv(output.with_suffix(".index.csv"), index=False)
