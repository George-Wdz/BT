"""Terminal readers used by minute-level training and online inference."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_flow import GEO_COLUMNS, IMAGE_COLUMNS, LINK_COLUMNS, add_geometry


def protocol_satellite_id(track_no: pd.Series, phase_no: pd.Series) -> pd.Series:
    return (track_no.astype("int64") * 256 + phase_no.astype("int64")).astype("int64")


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class RuntimeConfig:
    config_path: Path
    terminal_id: str
    db_path: Path
    adapter_path: Path | None
    min_pass_points: int = 3
    no_rain_threshold: float = 0.0
    pass_gap_threshold_s: float = 60.0
    max_passes: int = 2000

    @classmethod
    def load(cls, config_path: str | Path) -> "RuntimeConfig":
        path = Path(config_path).expanduser().resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            config_path=path,
            terminal_id=str(raw["terminal_id"]),
            db_path=_resolve(path.parent, raw["db_path"]),
            adapter_path=_resolve(path.parent, raw.get("adapter_path")),
            min_pass_points=int(raw.get("min_phy_points", raw.get("min_pass_points", 3))),
            no_rain_threshold=float(raw.get("no_rain_threshold", 0.0)),
            pass_gap_threshold_s=float(raw.get("pass_gap_threshold_s", 60.0)),
            max_passes=int(raw.get("max_passes", 2000)),
        )


class ZScoreDomainAdapter:
    def __init__(self, path: Path, target_mean: np.ndarray, target_scale: np.ndarray):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.terminal_id = str(raw["terminal_id"])
        self.source_columns = list(raw["source_columns"])
        self.source_mean = np.asarray(raw["source_mean"], dtype=np.float64)
        self.source_scale = np.asarray(raw["source_scale"], dtype=np.float64)
        self.clip_z = float(raw.get("clip_z", 5.0))
        self.target_mean = np.asarray(target_mean[:4], dtype=np.float64)
        self.target_scale = np.asarray(target_scale[:4], dtype=np.float64)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        source = np.column_stack([
            pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            for column in self.source_columns
        ])
        standardized = (source - self.source_mean) / self.source_scale
        standardized = np.clip(standardized, -self.clip_z, self.clip_z)
        return (self.target_mean + standardized * self.target_scale).astype(np.float32)


class _ImageLabelReader:
    def _configure_images(self, image_csv: Path | None) -> None:
        self.image_csv = image_csv
        self._image_cache: tuple[int, pd.DataFrame] | None = None

    def _images(self) -> pd.DataFrame | None:
        if self.image_csv is None or not self.image_csv.is_file():
            return None
        modified_ns = self.image_csv.stat().st_mtime_ns
        if self._image_cache is not None and self._image_cache[0] == modified_ns:
            return self._image_cache[1]
        frame = pd.read_csv(self.image_csv)
        required = ["timestamp", "prob_sunny", "prob_cloudy", "prob_rain"]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"image label CSV is missing columns: {missing}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        for column in required[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=required).sort_values("timestamp")
        if "image_available" not in frame:
            frame["image_available"] = 1.0
        else:
            frame["image_available"] = pd.to_numeric(
                frame["image_available"], errors="coerce"
            ).fillna(0.0)
        result = frame.set_index("timestamp")[IMAGE_COLUMNS]
        self._image_cache = (modified_ns, result)
        return result


class LegacyTerminalReader(_ImageLabelReader):
    def __init__(self, runtime: RuntimeConfig, image_csv: Path | None = None):
        self.runtime = runtime
        self.adapter = None
        self._configure_images(image_csv)

    def latest_link_time(self) -> pd.Timestamp | None:
        with sqlite3.connect(f"file:{self.runtime.db_path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT MAX(localTime) FROM phy_data WHERE terminalId=?",
                [self.runtime.terminal_id],
            ).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

    def _read_link(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        query = f"""
            SELECT localTime, satelliteId, earthStationId, {', '.join(LINK_COLUMNS)}
            FROM phy_data
            WHERE terminalId=? AND localTime>=? AND localTime<=?
              AND satelliteId IS NOT NULL AND satelliteId != 4294967295
              AND snr IS NOT NULL AND snr != 255
              AND phyRssi IS NOT NULL AND rssi IS NOT NULL
              AND lastCniValue IS NOT NULL
            ORDER BY localTime
        """
        with sqlite3.connect(f"file:{self.runtime.db_path}?mode=ro", uri=True) as connection:
            frame = pd.read_sql_query(
                query, connection,
                params=[self.runtime.terminal_id, start.isoformat(), end.isoformat()],
            )
        if frame.empty:
            return frame
        frame["localTime"] = pd.to_datetime(frame["localTime"], errors="coerce")
        return frame.dropna(subset=["localTime"]).set_index("localTime").sort_index()

    def _read_position(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        query = """
            SELECT localTime, satId, longitude, latitude, satAltitude,
                   posLongitude, posLatitude, altitude, ecefPx, ecefPy, ecefPz
            FROM position_data
            WHERE terminalId=? AND localTime>=? AND localTime<=?
              AND satId IS NOT NULL AND satId != 4294967295
              AND posLongitude IS NOT NULL AND posLongitude != 0
              AND posLatitude IS NOT NULL AND posLatitude != 0
              AND altitude IS NOT NULL AND altitude != 0
              AND ecefPx IS NOT NULL AND ecefPx != 0
              AND ecefPy IS NOT NULL AND ecefPy != 0
              AND ecefPz IS NOT NULL AND ecefPz != 0
            ORDER BY localTime
        """
        with sqlite3.connect(f"file:{self.runtime.db_path}?mode=ro", uri=True) as connection:
            frame = pd.read_sql_query(
                query, connection,
                params=[self.runtime.terminal_id, start.isoformat(), end.isoformat()],
            )
        if frame.empty:
            return pd.DataFrame(columns=["satId", *GEO_COLUMNS])
        frame["localTime"] = pd.to_datetime(frame["localTime"], errors="coerce")
        frame = add_geometry(frame.dropna(subset=["localTime"]))
        return frame.set_index("localTime")[["satId", *GEO_COLUMNS]].sort_index()


class NewTerminalReader(_ImageLabelReader):
    def __init__(self, runtime: RuntimeConfig, target_mean: np.ndarray,
                 target_scale: np.ndarray, image_csv: Path | None = None):
        if runtime.adapter_path is None:
            raise ValueError(f"adapter_path is required for {runtime.terminal_id}")
        self.runtime = runtime
        self.adapter = ZScoreDomainAdapter(runtime.adapter_path, target_mean, target_scale)
        self._configure_images(image_csv)
        if self.adapter.terminal_id != runtime.terminal_id:
            raise ValueError("terminal adapter does not match runtime config")

    def latest_link_time(self) -> pd.Timestamp | None:
        with sqlite3.connect(f"file:{self.runtime.db_path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT MAX(localTime) FROM phy_bb_data WHERE terminalId=?",
                [self.runtime.terminal_id],
            ).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

    def _read_link(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        query = """
            SELECT b.localTime, b.trackNo, b.phaseNo, b.snr,
                   b.rsrp, b.nipower, b.freqOffset, b.timeOffset,
                   r.gain, r.chanRssi, r.carrRssi
            FROM phy_bb_data AS b
            JOIN phy_rssi_data AS r
              ON r.terminalId=b.terminalId AND r.localTime=b.localTime
            WHERE b.terminalId=? AND b.localTime>=? AND b.localTime<=?
              AND b.validMeasBb=1 AND r.validMeasRssi=1
              AND b.trackNo IS NOT NULL AND b.phaseNo IS NOT NULL
              AND b.snr IS NOT NULL AND b.snr != 0
              AND r.chanRssi IS NOT NULL AND r.chanRssi != 0
              AND r.carrRssi IS NOT NULL AND r.carrRssi != 0
            ORDER BY b.localTime
        """
        with sqlite3.connect(f"file:{self.runtime.db_path}?mode=ro", uri=True) as connection:
            frame = pd.read_sql_query(
                query, connection,
                params=[self.runtime.terminal_id, start.isoformat(), end.isoformat()],
            )
        if frame.empty:
            return frame
        frame["localTime"] = pd.to_datetime(frame["localTime"], errors="coerce")
        frame = frame.dropna(subset=["localTime"])
        frame["satelliteId"] = protocol_satellite_id(frame["trackNo"], frame["phaseNo"])
        mapped = self.adapter.transform(frame)
        for index, column in enumerate(LINK_COLUMNS):
            frame[column] = mapped[:, index]
        frame["earthStationId"] = 0
        return frame.set_index("localTime")[
            ["satelliteId", "earthStationId", *LINK_COLUMNS]
        ].sort_index()
