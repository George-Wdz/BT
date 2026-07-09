"""
Stage1 数据预处理模块（Pass-based 版本）

核心思想：以卫星过境片段（pass）为基本数据单元，而非固定时间窗口。
每颗 LEO 卫星过境持续约 4 分钟，过境内链路数据物理上连贯。
"""
import json
from datetime import datetime

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from .db import (
    load_ground_weather,
    load_phy_data,
    load_position_data,
    load_weather_station,
)

PASS_GAP_THRESHOLD_S = 60.0
MIN_PASS_POINTS = 10

LINK_COLS = ["phyRssi", "rssi", "snr", "lastCniValue", "freqOffset", "td"]
POS_COLS = ["longitude", "latitude", "satAltitude",
            "posLongitude", "posLatitude", "altitude"]
POSITION_GEO_COLS = ["slant_range_km", "elevation_deg", "azimuth_sin", "azimuth_cos"]
TARGET_COLS = ["pass_rainfall_mm", "wind_speed", "wind_direction"]
IMAGE_WEATHER_COLS = ["prob_sunny", "prob_cloudy", "prob_rain", "image_available"]

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def merge_ground_weather(gw: pd.DataFrame, station: pd.DataFrame) -> pd.DataFrame:
    """Use weather_data first, then fill gaps from weather_station."""
    cols = ["temperature", "humidity", "pressure"]
    return gw[cols].combine_first(station[cols]).sort_index()


def load_image_weather_predictions(csv_path: str | None) -> pd.DataFrame | None:
    """Load image weather probabilities exported by vision/predict_weather_labels.py."""
    if not csv_path:
        return None
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"image weather label CSV not found: {path}")

    df = pd.read_csv(path)
    required = ["timestamp", "prob_sunny", "prob_cloudy", "prob_rain"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"image weather CSV missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in ["prob_sunny", "prob_cloudy", "prob_rain"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["prob_sunny", "prob_cloudy", "prob_rain"])
    if df.empty:
        raise ValueError(f"image weather CSV has no valid timestamp/probability rows: {path}")
    df["image_available"] = 1.0
    return df.set_index("timestamp")[IMAGE_WEATHER_COLS].sort_index()


def geodetic_to_ecef(lon_deg: pd.Series, lat_deg: pd.Series,
                     height_m: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WGS84 geodetic coordinates to ECEF meters."""
    lon = np.deg2rad(lon_deg.to_numpy(dtype=np.float64))
    lat = np.deg2rad(lat_deg.to_numpy(dtype=np.float64))
    h = height_m.to_numpy(dtype=np.float64)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h) * cos_lat * np.cos(lon)
    y = (n + h) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * sin_lat
    return x, y, z


def add_position_geometry(pos: pd.DataFrame) -> pd.DataFrame:
    """Append line-of-sight geometry from satellite/terminal positions."""
    required = [
        "posLongitude", "posLatitude", "altitude",
        "ecefPx", "ecefPy", "ecefPz",
    ]
    missing = [c for c in required if c not in pos.columns]
    if missing:
        raise ValueError(f"position geometry requires columns: {missing}")

    out = pos.copy()
    gx, gy, gz = geodetic_to_ecef(out["posLongitude"], out["posLatitude"], out["altitude"])
    sx = out["ecefPx"].to_numpy(dtype=np.float64)
    sy = out["ecefPy"].to_numpy(dtype=np.float64)
    sz = out["ecefPz"].to_numpy(dtype=np.float64)

    los_x = sx - gx
    los_y = sy - gy
    los_z = sz - gz
    slant_m = np.sqrt(los_x * los_x + los_y * los_y + los_z * los_z)

    lon = np.deg2rad(out["posLongitude"].to_numpy(dtype=np.float64))
    lat = np.deg2rad(out["posLatitude"].to_numpy(dtype=np.float64))
    up_x = np.cos(lat) * np.cos(lon)
    up_y = np.cos(lat) * np.sin(lon)
    up_z = np.sin(lat)
    east_x = -np.sin(lon)
    east_y = np.cos(lon)
    east_z = np.zeros_like(lon)
    north_x = -np.sin(lat) * np.cos(lon)
    north_y = -np.sin(lat) * np.sin(lon)
    north_z = np.cos(lat)

    east_m = los_x * east_x + los_y * east_y + los_z * east_z
    north_m = los_x * north_x + los_y * north_y + los_z * north_z
    up_m = los_x * up_x + los_y * up_y + los_z * up_z
    denom = np.maximum(slant_m, 1e-6)
    elevation_rad = np.arcsin(np.clip(up_m / denom, -1.0, 1.0))
    azimuth_rad = np.arctan2(east_m, north_m)

    out["slant_range_km"] = (slant_m / 1000.0).astype(np.float32)
    out["elevation_deg"] = np.rad2deg(elevation_rad).astype(np.float32)
    out["azimuth_sin"] = np.sin(azimuth_rad).astype(np.float32)
    out["azimuth_cos"] = np.cos(azimuth_rad).astype(np.float32)
    return out


def _cumulative_at(weather_station: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    """Estimate daily cumulative rainfall at an arbitrary pass boundary."""
    day = timestamp.date()
    day_rows = weather_station[weather_station.index.date == day]
    if len(day_rows) < 2:
        return None

    cum = pd.to_numeric(day_rows["rainfall_cumulative"], errors="coerce")
    valid = cum.notna()
    day_rows = day_rows.loc[valid]
    cum = cum.loc[valid]
    if len(day_rows) < 2 or timestamp < day_rows.index[0] or timestamp > day_rows.index[-1]:
        return None

    x = day_rows.index.view("int64").astype(np.float64)
    y = cum.to_numpy(dtype=np.float64)
    t = float(timestamp.value)
    value = float(np.interp(t, x, y))
    return value if np.isfinite(value) else None


def compute_pass_labels(weather_station: pd.DataFrame,
                        start: pd.Timestamp,
                        end: pd.Timestamp) -> tuple[np.ndarray | None, dict]:
    """Build pass-level labels from weather_station.

    The primary rainfall label is the cumulative-rainfall difference across
    the satellite pass. Instant rainfall is retained as metadata for analysis,
    but it is not used as the regression target.
    """
    if start.date() != end.date():
        return None, {"drop_reason": "cross_day_pass"}

    start_cum = _cumulative_at(weather_station, start)
    end_cum = _cumulative_at(weather_station, end)
    if start_cum is None or end_cum is None:
        return None, {"drop_reason": "missing_cumulative_boundary"}

    pass_rainfall = end_cum - start_cum
    if pass_rainfall < -1e-4:
        return None, {"drop_reason": "negative_cumulative_delta"}
    pass_rainfall = max(pass_rainfall, 0.0)

    ws_in_range = weather_station.loc[start:end]
    if len(ws_in_range) == 0:
        center = start + (end - start) / 2
        ws_in_range = weather_station.reindex(
            [center], method="nearest", tolerance=pd.Timedelta("5min")
        ).dropna(subset=["wind_speed", "wind_direction", "rainfall"])
    if len(ws_in_range) == 0:
        return None, {"drop_reason": "missing_weather_station_window"}

    wind_speed = pd.to_numeric(ws_in_range["wind_speed"], errors="coerce").mean()
    wind_direction = pd.to_numeric(ws_in_range["wind_direction"], errors="coerce").mean()
    rain_rate = pd.to_numeric(ws_in_range["rainfall"], errors="coerce")
    if not np.isfinite(wind_speed) or not np.isfinite(wind_direction):
        return None, {"drop_reason": "missing_wind_label"}

    labels = np.array([pass_rainfall, wind_speed, wind_direction], dtype=np.float32)
    meta = {
        "pass_start": start,
        "pass_end": end,
        "weather_rows": int(len(ws_in_range)),
        "rain_rate_mean": float(rain_rate.mean()) if rain_rate.notna().any() else np.nan,
        "rain_rate_max": float(rain_rate.max()) if rain_rate.notna().any() else np.nan,
        "rainy_ratio": float((rain_rate > 0).mean()) if rain_rate.notna().any() else np.nan,
    }
    return labels, meta


def segment_passes(phy: pd.DataFrame, pos: pd.DataFrame,
                   link_cols: list[str] | None = None,
                   pos_cols: list[str] | None = None,
                   gap_threshold: float = PASS_GAP_THRESHOLD_S,
                   min_points: int = MIN_PASS_POINTS) -> List[Dict]:
    """
    按卫星ID和时间间隔切分过境片段。
    一次过境 = 同一颗卫星连续观测（间隔<gap_threshold秒）的序列。
    """
    link_cols = link_cols or LINK_COLS
    pos_cols = pos_cols or POS_COLS
    passes = []
    pos_by_sat = {
        int(sat_id): sat_pos.sort_index()
        for sat_id, sat_pos in pos.groupby("satId")
        if pd.notna(sat_id)
    }
    for sat_id, sat_phy in phy.groupby("satelliteId"):
        sat_phy = sat_phy.sort_index()
        if len(sat_phy) < min_points:
            continue
        sat_pos = pos_by_sat.get(int(sat_id))
        if sat_pos is None or len(sat_pos) < min_points:
            continue

        times = sat_phy.index.to_series()
        gaps = times.diff().dt.total_seconds()
        # 标记过境切换点
        new_pass = (gaps > gap_threshold) | gaps.isna()
        pass_ids = new_pass.cumsum()

        for _, seg in sat_phy.groupby(pass_ids):
            if len(seg) < min_points:
                continue
            t_start, t_end = seg.index[0], seg.index[-1]
            # Match position only within the same satellite ID. A nearest
            # timestamp from another satellite would corrupt link geometry.
            seg_pos = sat_pos.reindex(seg.index, method="nearest",
                                      tolerance=pd.Timedelta("5s"))
            valid = seg_pos[pos_cols].notna().all(axis=1)
            if valid.sum() < min_points:
                continue
            idx = seg.index[valid]
            passes.append({
                "satellite_id": int(sat_id),
                "timestamps": idx.values,
                "link_features": seg.loc[idx, link_cols].values.astype(np.float32),
                "position_features": seg_pos.loc[idx, pos_cols].values.astype(np.float32),
                "feature_columns": {
                    "link": list(link_cols),
                    "position": list(pos_cols),
                },
            })

    print(f"Segmented {len(passes)} valid passes "
          f"from {phy['satelliteId'].nunique()} satellites")
    return passes


def attach_features_and_labels(passes: List[Dict],
                               ground_weather: pd.DataFrame,
                               weather_station: pd.DataFrame,
                               weather_cols: list[str] | None = None,
                               image_weather: pd.DataFrame | None = None,
                               image_tolerance: str = "10min") -> List[Dict]:
    """
    为每个过境片段附加：
    - 地面气象（每个时间步的温湿压，最近邻匹配）
    - 气象站标签：累计雨量差分 + 风速/风向均值
    """
    gw_cols = weather_cols or ["temperature", "humidity", "pressure"]
    image_tol = pd.Timedelta(image_tolerance)

    enriched = []
    drop_reasons: dict[str, int] = {}
    image_matched = 0
    for p in passes:
        idx = pd.DatetimeIndex(p["timestamps"])
        t_start, t_end = idx[0], idx[-1]

        # 地面气象：最近邻匹配（容忍 60 秒，因为采样频率较低）
        gw_aligned = ground_weather[gw_cols].reindex(
            idx, method="nearest", tolerance=pd.Timedelta("60s")
        )
        if gw_aligned.isna().any().any():
            gw_aligned = gw_aligned.ffill().bfill()
        if gw_aligned.isna().any().any():
            continue

        labels, label_meta = compute_pass_labels(weather_station, t_start, t_end)
        if labels is None:
            reason = label_meta.get("drop_reason", "missing_label")
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue

        image_features = None
        image_meta = {}
        if image_weather is not None:
            center = t_start + (t_end - t_start) / 2
            nearest_pos = image_weather.index.get_indexer(
                [center], method="nearest", tolerance=image_tol
            )[0]
            if nearest_pos < 0:
                vec = np.zeros(len(IMAGE_WEATHER_COLS), dtype=np.float32)
                image_meta = {
                    "image_available": 0,
                    "image_time_delta_s": np.nan,
                }
            else:
                matched_ts = image_weather.index[nearest_pos]
                row = image_weather.iloc[nearest_pos]
                vec = row[IMAGE_WEATHER_COLS].to_numpy(dtype=np.float32)
                image_matched += 1
                image_meta = {
                    "image_available": 1,
                    "image_time_delta_s": abs((matched_ts - center).total_seconds()),
                }
            image_features = np.repeat(vec.reshape(1, -1), len(idx), axis=0)

        enriched.append({
            **p,
            "ground_weather": gw_aligned.values.astype(np.float32),
            "feature_columns": {
                **p.get("feature_columns", {}),
                "ground_weather": list(gw_cols),
                **({"image_weather": list(IMAGE_WEATHER_COLS)}
                   if image_features is not None else {}),
            },
            **({"image_weather": image_features.astype(np.float32)}
               if image_features is not None else {}),
            "labels": labels,  # (3,) [pass_rainfall_mm, wind_speed, wind_direction]
            "label_meta": {**label_meta, **image_meta},
        })

    print(f"Attached features and labels: {len(enriched)} valid passes "
          f"(dropped {len(passes) - len(enriched)} due to missing data)")
    if drop_reasons:
        print(f"Label drop reasons: {drop_reasons}")
    if image_weather is not None:
        print(f"Image weather matched: {image_matched} / {len(enriched)} passes "
              f"(tolerance={image_tolerance})")
    return enriched


def pass_index_frame(dataset: List[Dict]) -> pd.DataFrame:
    """Build a compact, inspectable index for the persisted pass dataset."""
    rows = []
    for i, p in enumerate(dataset):
        idx = pd.DatetimeIndex(p["timestamps"])
        labels = p["labels"]
        meta = p.get("label_meta", {})
        rows.append({
            "pass_id": i,
            "satellite_id": p["satellite_id"],
            "pass_start": idx[0],
            "pass_end": idx[-1],
            "duration_s": (idx[-1] - idx[0]).total_seconds(),
            "points": len(idx),
            "weather_rows": meta.get("weather_rows"),
            "pass_rainfall_mm": float(labels[0]),
            "wind_speed": float(labels[1]),
            "wind_direction": float(labels[2]),
            "rain_rate_mean": meta.get("rain_rate_mean"),
            "rain_rate_max": meta.get("rain_rate_max"),
            "rainy_ratio": meta.get("rainy_ratio"),
            "image_available": meta.get("image_available"),
            "image_time_delta_s": meta.get("image_time_delta_s"),
        })
    return pd.DataFrame(rows)


def dataset_summary(dataset: List[Dict], db_path: str, source_ranges: dict,
                    feature_cols: dict | None = None) -> dict:
    """Summarize the DB snapshot materialized into pass_dataset.npz."""
    if not dataset:
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_db": db_path,
            "total_passes": 0,
            "source_ranges": source_ranges,
            "targets": TARGET_COLS,
            "features": feature_cols,
        }

    index = pass_index_frame(dataset)
    labels = np.stack([p["labels"] for p in dataset])
    lengths = index["points"].to_numpy()
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_db": db_path,
        "source_ranges": source_ranges,
        "features": feature_cols,
        "targets": TARGET_COLS,
        "label_policy": {
            "primary": "rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)",
            "auxiliary": ["mean wind_speed in pass window", "mean wind_direction in pass window"],
            "instant_rainfall": "stored only in index metadata as rain_rate_mean/rain_rate_max/rainy_ratio",
        },
        "total_passes": int(len(dataset)),
        "unique_satellites": int(index["satellite_id"].nunique()),
        "rainy_passes": int((labels[:, 0] > 0).sum()),
        "pass_length_points": {
            "min": int(lengths.min()),
            "median": int(np.median(lengths)),
            "p90": int(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
        },
        "pass_rainfall_mm": {
            "min": float(labels[:, 0].min()),
            "max": float(labels[:, 0].max()),
            "mean": float(labels[:, 0].mean()),
        },
        "time_range": {
            "pass_start": str(index["pass_start"].min()),
            "pass_end": str(index["pass_end"].max()),
        },
    }


def save_dataset_artifacts(dataset: List[Dict], db_path: str,
                           output_path: str, source_ranges: dict,
                           feature_cols: dict | None = None) -> None:
    """Persist training data plus lightweight audit files next to the npz."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sat_ids = sorted(set(p["satellite_id"] for p in dataset))
    np.savez_compressed(
        out,
        passes=np.array(dataset, dtype=object),
        satellite_ids=np.array(sat_ids, dtype=np.int64),
    )

    index_path = out.with_suffix(".index.csv")
    summary_path = out.with_suffix(".summary.json")
    pass_index_frame(dataset).to_csv(index_path, index=False)
    summary_path.write_text(
        json.dumps(dataset_summary(dataset, db_path, source_ranges, feature_cols),
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")
    print(f"Saved index to {index_path}")
    print(f"Saved summary to {summary_path}")


def build_pass_dataset(db_path: str, output_path: str = None,
                       feature_cols: dict | None = None,
                       strict_source_filters: bool = False,
                       image_weather_cfg: dict | None = None,
                       start_time: str | None = None,
                       end_time: str | None = None) -> List[Dict]:
    """构建 pass-based 数据集的完整流程"""
    feature_cols = feature_cols or {
        "link": LINK_COLS,
        "position": POS_COLS,
        "ground_weather": ["temperature", "humidity", "pressure"],
    }
    link_cols = feature_cols.get("link", LINK_COLS)
    pos_cols = feature_cols.get("position", POS_COLS)
    weather_cols = feature_cols.get("ground_weather", ["temperature", "humidity", "pressure"])
    image_weather_cfg = image_weather_cfg or {}
    use_image_weather = bool(image_weather_cfg.get("enabled", False))

    print("Loading phy_data from DB...")
    phy = load_phy_data(
        db_path,
        feature_cols=link_cols,
        strict_source_filters=strict_source_filters,
        start_time=start_time,
        end_time=end_time,
    ).set_index("localTime").sort_index()

    print("Loading position_data from DB...")
    pos = load_position_data(
        db_path,
        strict_source_filters=strict_source_filters,
        start_time=start_time,
        end_time=end_time,
    ).set_index("localTime").sort_index()
    if any(c in pos_cols for c in POSITION_GEO_COLS):
        print("Computing position geometry features...")
        pos = add_position_geometry(pos)
    # Keep satId for same-satellite alignment; only pos_cols enter features.
    pos = pos[["satId", *pos_cols]].sort_index()

    print("Loading weather_data from DB...")
    gw = load_ground_weather(db_path, start_time=start_time, end_time=end_time)

    print("Loading weather_station from DB...")
    ws = load_weather_station(db_path, start_time=start_time, end_time=end_time)

    print("Merging input weather...")
    gw_merged = merge_ground_weather(gw, ws)

    image_weather = None
    if use_image_weather:
        print("Loading image weather labels...")
        image_weather = load_image_weather_predictions(image_weather_cfg.get("csv_path"))

    print(f"\nData ranges:")
    print(f"  phy_data: {phy.index.min()} ~ {phy.index.max()} ({len(phy)} rows)")
    print(f"  position_data: {pos.index.min()} ~ {pos.index.max()} ({len(pos)} rows)")
    print(f"  weather_data: {gw.index.min()} ~ {gw.index.max()} ({len(gw)} rows)")
    print(f"  weather_station: {ws.index.min()} ~ {ws.index.max()} ({len(ws)} rows)")
    print(f"  input_weather: {gw_merged.index.min()} ~ {gw_merged.index.max()} ({len(gw_merged)} rows)")
    if image_weather is not None:
        print(f"  image_weather: {image_weather.index.min()} ~ {image_weather.index.max()} ({len(image_weather)} rows)")
    source_ranges = {
        "phy_data": {"start": str(phy.index.min()), "end": str(phy.index.max()), "rows": int(len(phy))},
        "position_data": {"start": str(pos.index.min()), "end": str(pos.index.max()), "rows": int(len(pos))},
        "weather_data": {"start": str(gw.index.min()), "end": str(gw.index.max()), "rows": int(len(gw))},
        "weather_station": {"start": str(ws.index.min()), "end": str(ws.index.max()), "rows": int(len(ws))},
        "input_weather": {"start": str(gw_merged.index.min()), "end": str(gw_merged.index.max()), "rows": int(len(gw_merged))},
    }
    if image_weather is not None:
        source_ranges["image_weather"] = {
            "start": str(image_weather.index.min()),
            "end": str(image_weather.index.max()),
            "rows": int(len(image_weather)),
            "csv_path": image_weather_cfg.get("csv_path"),
            "tolerance": image_weather_cfg.get("tolerance", "10min"),
        }

    print("\nSegmenting passes...")
    passes = segment_passes(phy, pos, link_cols=link_cols, pos_cols=pos_cols)

    print("\nAttaching features and labels...")
    dataset = attach_features_and_labels(
        passes,
        gw_merged,
        ws,
        weather_cols=weather_cols,
        image_weather=image_weather,
        image_tolerance=image_weather_cfg.get("tolerance", "10min"),
    )

    # 统计
    if dataset:
        lens = [len(p["link_features"]) for p in dataset]
        sat_ids = sorted(set(p["satellite_id"] for p in dataset))
        labels = np.stack([p["labels"] for p in dataset])
        print(f"\n=== Dataset Stats ===")
        print(f"Total passes: {len(dataset)}")
        print(f"Unique satellites: {len(sat_ids)}")
        print(f"Pass length: min={min(lens)}, median={int(np.median(lens))}, "
              f"p90={int(np.percentile(lens, 90))}, max={max(lens)}")
        print(f"Pass rainfall: min={labels[:, 0].min():.2f}, "
              f"max={labels[:, 0].max():.2f}, mean={labels[:, 0].mean():.4f}")
        print(f"Wind speed: min={labels[:, 1].min():.2f}, "
              f"max={labels[:, 1].max():.2f}, mean={labels[:, 1].mean():.2f}")
        print(f"Rainy passes (pass_rainfall_mm>0): "
              f"{(labels[:, 0] > 0).sum()} / {len(dataset)}")

    if output_path:
        save_dataset_artifacts(dataset, db_path, output_path, source_ranges, feature_cols)

    return dataset
