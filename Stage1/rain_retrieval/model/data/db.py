"""SQLite read helpers for Stage1 datasets."""
from __future__ import annotations

import sqlite3

import pandas as pd


INVALID_SATELLITE_ID = 4294967295
DB_URI_SUFFIX = "?mode=ro"


def connect_readonly(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}{DB_URI_SUFFIX}", uri=True)


def load_phy_data(
    db_path: str,
    feature_cols: list[str] | None = None,
    strict_source_filters: bool = False,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    """Load valid link samples from phy_data.

    The source DB already owns duplicate handling via unique indexes. The
    predicates below only keep physically usable rows for model features.
    """
    feature_cols = feature_cols or [
        "phyRssi", "rssi", "snr", "lastCniValue", "freqOffset", "td"
    ]
    allowed_cols = {
        "phyRssi", "rssi", "snr", "lastCniValue", "freqOffset", "td", "ncr"
    }
    unknown = sorted(set(feature_cols) - allowed_cols)
    if unknown:
        raise ValueError(f"Unknown phy_data feature columns: {unknown}")

    select_cols = ", ".join(feature_cols)
    predicates = []
    params = []
    if strict_source_filters:
        predicates.append("satelliteId != ?")
        params.append(INVALID_SATELLITE_ID)
    if strict_source_filters and "snr" in feature_cols:
        predicates.append("snr != 255")
    for col in feature_cols:
        predicates.append(f"{col} IS NOT NULL")
    if strict_source_filters and "freqOffset" in feature_cols:
        predicates.append("freqOffset != 0")
    if strict_source_filters and "td" in feature_cols:
        predicates.append("td != 0")
    if start_time:
        predicates.append("datetime(localTime) >= datetime(?)")
        params.append(start_time)
    if end_time:
        predicates.append("datetime(localTime) <= datetime(?)")
        params.append(end_time)

    where_sql = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    query = f"""
        SELECT localTime, satelliteId, earthStationId, {select_cols}
        FROM phy_data
        {where_sql}
        ORDER BY localTime
    """
    with connect_readonly(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    df["earthStationId"] = 0
    df["localTime"] = pd.to_datetime(df["localTime"], format="ISO8601")
    return df


def load_position_data(
    db_path: str,
    strict_source_filters: bool = False,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    predicates = []
    params = []
    where_sql = """
        SELECT localTime, satId,
               longitude, latitude, satAltitude,
               posLongitude, posLatitude, altitude,
               ecefPx, ecefPy, ecefPz
        FROM position_data
    """
    if strict_source_filters:
        predicates.extend([
            "satAltitude IS NOT NULL",
            "satAltitude != 0",
            "longitude IS NOT NULL",
            "latitude IS NOT NULL",
            "posLongitude IS NOT NULL",
            "posLatitude IS NOT NULL",
        ])
    if start_time:
        predicates.append("datetime(localTime) >= datetime(?)")
        params.append(start_time)
    if end_time:
        predicates.append("datetime(localTime) <= datetime(?)")
        params.append(end_time)
    if predicates:
        where_sql += " WHERE " + " AND ".join(predicates)
    query = where_sql + " ORDER BY localTime"
    with connect_readonly(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    df["localTime"] = pd.to_datetime(df["localTime"], format="ISO8601")
    return df


def load_ground_weather(
    db_path: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    predicates = []
    params = []
    if start_time:
        predicates.append("datetime(timestamp) >= datetime(?)")
        params.append(start_time)
    if end_time:
        predicates.append("datetime(timestamp) <= datetime(?)")
        params.append(end_time)
    where_sql = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    query = """
        SELECT timestamp, temperature, humidity, pressure
        FROM weather_data
        {where_sql}
        ORDER BY timestamp
    """.format(where_sql=where_sql)
    with connect_readonly(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.set_index("timestamp").sort_index()


def load_weather_station(
    db_path: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    predicates = []
    params = []
    if start_time:
        predicates.append("datetime(datetime) >= datetime(?)")
        params.append(start_time)
    if end_time:
        predicates.append("datetime(datetime) <= datetime(?)")
        params.append(end_time)
    where_sql = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    query = """
        SELECT datetime AS timestamp,
               temperature,
               humidity,
               pressure,
               wind_speed,
               wind_direction,
               rainfall,
               rainfall_cumulative
        FROM weather_station
        {where_sql}
        ORDER BY datetime
    """.format(where_sql=where_sql)
    with connect_readonly(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.set_index("timestamp").sort_index()
