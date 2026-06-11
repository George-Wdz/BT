"""Build Stage2-ready weather tables from DB weather and Stage1 pass outputs.

The output follows GPT4TS Dataset_Custom convention:
    date, feature_1, ..., feature_n, target

Rows are regular time buckets. The target is fixed-window rainfall computed
from weather_station.rainfall_cumulative, while Stage1 pass-level rainfall is
aggregated into bucket-level features by pass end time.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DB = "/home/wdz/satellite_data/satellite_data.db"
DEFAULT_PASS_INDEX = "/home/wdz/BT/Stage1/model/data/pass_dataset.index.csv"
DEFAULT_OUTPUT = (
    "/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/"
    "datasets/weather/stage1_5_weather_10min.csv"
)


def connect_readonly(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_weather_station(db_path: str) -> pd.DataFrame:
    query = """
        SELECT datetime,
               temperature,
               humidity,
               pressure,
               wind_speed,
               wind_direction,
               rainfall,
               rainfall_cumulative
        FROM weather_station
        ORDER BY datetime
    """
    with connect_readonly(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601")
    return df.set_index("datetime").sort_index()


def cumulative_at(weather: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    """Interpolate daily cumulative rainfall at a bucket boundary."""
    day_rows = weather[weather.index.date == timestamp.date()]
    if len(day_rows) < 2:
        return None

    cumulative = pd.to_numeric(day_rows["rainfall_cumulative"], errors="coerce")
    valid = cumulative.notna()
    day_rows = day_rows.loc[valid]
    cumulative = cumulative.loc[valid]
    if len(day_rows) < 2 or timestamp < day_rows.index[0] or timestamp > day_rows.index[-1]:
        return None

    x = day_rows.index.view("int64").astype(np.float64)
    y = cumulative.to_numpy(dtype=np.float64)
    value = float(np.interp(float(timestamp.value), x, y))
    return value if np.isfinite(value) else None


def window_rainfall(weather: pd.DataFrame, end_time: pd.Timestamp,
                    freq: pd.Timedelta) -> float:
    start_time = end_time - freq
    if start_time.date() != end_time.date():
        return np.nan
    start_value = cumulative_at(weather, start_time)
    end_value = cumulative_at(weather, end_time)
    if start_value is None or end_value is None:
        return np.nan
    delta = end_value - start_value
    return max(float(delta), 0.0) if delta >= -1e-4 else np.nan


def circular_wind_mean(degrees: pd.Series, speeds: pd.Series) -> float:
    deg = pd.to_numeric(degrees, errors="coerce")
    speed = pd.to_numeric(speeds, errors="coerce").fillna(0.0)
    valid = deg.notna() & (speed > 0)
    if not valid.any():
        return 0.0
    radians = np.deg2rad(deg[valid].to_numpy(dtype=np.float64))
    weights = speed[valid].to_numpy(dtype=np.float64)
    sin_mean = np.average(np.sin(radians), weights=weights)
    cos_mean = np.average(np.cos(radians), weights=weights)
    return float((np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0)


def aggregate_weather(weather: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate weather_station into regular right-labeled buckets."""
    freq_delta = pd.Timedelta(freq)
    grouped = weather.resample(freq, label="right", closed="right")
    out = grouped.agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        pressure=("pressure", "mean"),
        wind_speed=("wind_speed", "mean"),
        rain_rate_mean=("rainfall", "mean"),
        rain_rate_max=("rainfall", "max"),
        weather_rows=("rainfall", "size"),
    )

    wind_direction = grouped.apply(
        lambda g: circular_wind_mean(g["wind_direction"], g["wind_speed"])
    )
    out["wind_direction"] = wind_direction
    radians = np.deg2rad(out["wind_direction"].astype(float))
    out["wind_dir_sin"] = np.sin(radians)
    out["wind_dir_cos"] = np.cos(radians)
    out["wind_east"] = out["wind_speed"] * out["wind_dir_sin"]
    out["wind_north"] = out["wind_speed"] * out["wind_dir_cos"]
    out["rain_window_mm"] = [
        window_rainfall(weather, ts, freq_delta) for ts in out.index
    ]
    out = out.dropna(subset=[
        "temperature", "humidity", "pressure", "wind_speed", "rain_window_mm"
    ])
    out.index.name = "date"
    return out


def load_pass_index(path: str | None, rain_col: str) -> pd.DataFrame | None:
    if not path:
        return None
    index_path = Path(path)
    if not index_path.exists():
        print(f"Stage1 pass index not found, stage1_* features will be zero: {index_path}")
        return None

    df = pd.read_csv(index_path, parse_dates=["pass_start", "pass_end"])
    if rain_col not in df.columns:
        raise ValueError(f"Rain column '{rain_col}' not found in {index_path}")
    return df


def aggregate_stage1_passes(pass_index: pd.DataFrame | None,
                            bucket_index: pd.DatetimeIndex,
                            freq: str,
                            rain_col: str) -> pd.DataFrame:
    columns = [
        "stage1_rain_sum", "stage1_rain_mean", "stage1_rain_max",
        "stage1_rain_rate_mean", "stage1_rain_rate_max",
        "stage1_pass_count", "stage1_has_pass",
    ]
    if pass_index is None or pass_index.empty:
        return pd.DataFrame(0.0, index=bucket_index, columns=columns)

    freq_delta = pd.Timedelta(freq)
    passes = pass_index.copy()
    # Rows are labeled by bucket end. A pass ending at 11:34 belongs to 11:40
    # for 10min buckets, matching online availability.
    passes["date"] = passes["pass_end"].dt.ceil(freq_delta)
    grouped = passes.groupby("date")
    out = grouped.agg(
        stage1_rain_sum=(rain_col, "sum"),
        stage1_rain_mean=(rain_col, "mean"),
        stage1_rain_max=(rain_col, "max"),
        stage1_rain_rate_mean=("rain_rate_mean", "mean"),
        stage1_rain_rate_max=("rain_rate_max", "max"),
        stage1_pass_count=(rain_col, "size"),
    )
    out = out.reindex(bucket_index)
    out["stage1_has_pass"] = out["stage1_pass_count"].fillna(0).gt(0).astype(float)
    out = out.fillna(0.0)
    return out[columns]


def build_table(db_path: str, freq: str, pass_index_path: str | None,
                stage1_rain_col: str, target_name: str,
                start: str | None = None, end: str | None = None) -> pd.DataFrame:
    weather = load_weather_station(db_path)
    if start:
        weather = weather[weather.index >= pd.Timestamp(start)]
    if end:
        weather = weather[weather.index < pd.Timestamp(end)]
    if weather.empty:
        raise ValueError("No weather_station rows in selected time range.")

    weather_buckets = aggregate_weather(weather, freq)
    pass_index = load_pass_index(pass_index_path, stage1_rain_col)
    stage1_features = aggregate_stage1_passes(
        pass_index, weather_buckets.index, freq, stage1_rain_col
    )
    table = weather_buckets.join(stage1_features, how="left").fillna(0.0)
    table = table.rename(columns={"rain_window_mm": target_name})

    feature_cols = [
        "temperature", "humidity", "pressure",
        "wind_speed", "wind_direction", "wind_dir_sin", "wind_dir_cos",
        "wind_east", "wind_north",
        "rain_rate_mean", "rain_rate_max", "weather_rows",
        "stage1_rain_sum", "stage1_rain_mean", "stage1_rain_max",
        "stage1_rain_rate_mean", "stage1_rain_rate_max",
        "stage1_pass_count", "stage1_has_pass",
    ]
    table = table.reset_index()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return table[["date", *feature_cols, target_name]]


def write_summary(table: pd.DataFrame, output_path: Path,
                  db_path: str, pass_index_path: str | None,
                  freq: str, target_name: str) -> None:
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_db": db_path,
        "pass_index": pass_index_path,
        "freq": freq,
        "target": target_name,
        "rows": int(len(table)),
        "date_start": table["date"].min() if len(table) else None,
        "date_end": table["date"].max() if len(table) else None,
        "target_positive_rows": int((table[target_name] > 0).sum()) if len(table) else 0,
        "target_sum_mm": float(table[target_name].sum()) if len(table) else 0.0,
        "stage1_pass_rows": int((table["stage1_pass_count"] > 0).sum()) if len(table) else 0,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=DEFAULT_DB)
    parser.add_argument("--pass-index", default=DEFAULT_PASS_INDEX,
                        help="Stage1 pass_dataset.index.csv. If missing, stage1_* features are zero.")
    parser.add_argument("--stage1-rain-col", default="pass_rainfall_mm",
                        help="Column in pass index to aggregate as Stage1 rainfall.")
    parser.add_argument("--freq", default="10min",
                        help="Pandas frequency, e.g. 10min, 30min, 1h.")
    parser.add_argument("--target-name", default=None,
                        help="Default: rain_<freq>_mm, e.g. rain_10min_mm.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_name = args.target_name or f"rain_{args.freq}_mm".replace(" ", "")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = build_table(
        db_path=args.db_path,
        freq=args.freq,
        pass_index_path=args.pass_index,
        stage1_rain_col=args.stage1_rain_col,
        target_name=target_name,
        start=args.start,
        end=args.end,
    )
    table.to_csv(output_path, index=False)
    print(f"Saved table: {output_path} rows={len(table)} target={target_name}")
    write_summary(table, output_path, args.db_path, args.pass_index, args.freq, target_name)


if __name__ == "__main__":
    main()
