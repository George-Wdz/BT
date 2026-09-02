#!/usr/bin/env python3
"""Strict parser for the observed fixed-size LET table container.

Only the container fields verified from the supplied files are decoded.  The
17-byte orbit payload is intentionally kept opaque until its vendor format is
available or an independent ECEF validation succeeds.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path


HEADER_SIZE = 2
RECORD_SIZE = 173
MAX_RECORDS = 1296
POINT_SIZE = 21
MAX_POINTS = 8


@dataclass(frozen=True)
class LetPoint:
    epoch_bdt_seconds: int
    orbit_payload: bytes

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.orbit_payload).hexdigest()


@dataclass(frozen=True)
class LetRecord:
    satellite_id: int
    declared_point_count: int
    points: tuple[LetPoint, ...]


@dataclass(frozen=True)
class LetTable:
    path: Path
    declared_record_count: int
    records: tuple[LetRecord, ...]

    @property
    def by_id(self) -> dict[int, LetRecord]:
        return {record.satellite_id: record for record in self.records}


def parse_let_table(path: str | Path) -> LetTable:
    path = Path(path)
    data = path.read_bytes()
    expected_size = HEADER_SIZE + MAX_RECORDS * RECORD_SIZE
    if len(data) != expected_size:
        raise ValueError(
            f"unexpected LET size for {path}: {len(data)} bytes; "
            f"expected {expected_size}"
        )

    record_count = struct.unpack_from("<H", data, 0)[0]
    if not 0 <= record_count <= MAX_RECORDS:
        raise ValueError(f"invalid LET record count: {record_count}")

    records: list[LetRecord] = []
    seen_ids: set[int] = set()
    for index in range(record_count):
        offset = HEADER_SIZE + index * RECORD_SIZE
        satellite_id = struct.unpack_from("<I", data, offset)[0]
        point_count = data[offset + 4]
        if point_count > MAX_POINTS:
            raise ValueError(
                f"record {index} satellite {satellite_id} declares "
                f"{point_count} points; maximum is {MAX_POINTS}"
            )
        if satellite_id in seen_ids:
            raise ValueError(f"duplicate satellite ID {satellite_id} in {path}")
        seen_ids.add(satellite_id)

        points: list[LetPoint] = []
        points_offset = offset + 5
        for point_index in range(point_count):
            start = points_offset + point_index * POINT_SIZE
            raw_point = data[start : start + POINT_SIZE]
            epoch = struct.unpack_from("<I", raw_point, 0)[0]
            payload = raw_point[4:]
            if epoch == 0 and not any(payload):
                continue
            points.append(LetPoint(epoch, payload))

        records.append(LetRecord(satellite_id, point_count, tuple(points)))

    return LetTable(path, record_count, tuple(records))
