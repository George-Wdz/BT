"""Shared position-quality and orbit-coordinate helpers."""

from __future__ import annotations

import math
from datetime import datetime


WGS84_A_M = 6378137.0
WGS84_E2 = 6.69437999014e-3

VERSION_BOUNDARIES = (
    (datetime.fromisoformat("2026-07-08T23:43:54.540569"), "0727"),
    (datetime.fromisoformat("2026-05-27T10:38:04.238518"), "0611"),
    (datetime.fromisoformat("2026-04-29T18:21:19.033281"), "0429"),
)


def version_for_local_time(value: str | datetime) -> str:
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    timestamp = timestamp.replace(tzinfo=None)
    for boundary, version in VERSION_BOUNDARIES:
        if timestamp >= boundary:
            return version
    return "0401"


def position_quality_reason(row: dict) -> str | None:
    fields = (
        "satAltitude", "longitude", "latitude", "altitude",
        "posLongitude", "posLatitude", "ecefPx", "ecefPy", "ecefPz",
    )
    values: dict[str, float] = {}
    for field in fields:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            return f"missing_{field}"
        if not math.isfinite(value):
            return f"nonfinite_{field}"
        if value == 0.0:
            return f"zero_{field}"
        values[field] = value

    radius = math.sqrt(sum(values[field] ** 2 for field in ("ecefPx", "ecefPy", "ecefPz")))
    if not 6.4e6 <= radius <= 1.0e7:
        return "ecef_radius_out_of_range"
    if not 1.0e5 <= values["satAltitude"] <= 3.0e6:
        return "satAltitude_out_of_leo_range"
    return None


def ecef_to_geodetic(ecef_m: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert WGS-84 ECEF metres to longitude/latitude degrees and height."""
    x, y, z = ecef_m
    longitude = math.atan2(y, x)
    p = math.hypot(x, y)
    latitude = math.atan2(z, p * (1.0 - WGS84_E2))
    height = 0.0
    for _ in range(10):
        sin_lat = math.sin(latitude)
        prime_vertical = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = p / max(math.cos(latitude), 1e-12) - prime_vertical
        next_latitude = math.atan2(
            z,
            p * (1.0 - WGS84_E2 * prime_vertical / (prime_vertical + height)),
        )
        if abs(next_latitude - latitude) < 1e-12:
            latitude = next_latitude
            break
        latitude = next_latitude
    return math.degrees(longitude), math.degrees(latitude), height
