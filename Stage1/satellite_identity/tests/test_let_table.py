from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from satellite_identity.let_table import (
    HEADER_SIZE,
    MAX_RECORDS,
    RECORD_SIZE,
    parse_let_table,
)


class LetTableParserTest(unittest.TestCase):
    def make_table(self, point_count: int = 1) -> Path:
        data = bytearray(HEADER_SIZE + MAX_RECORDS * RECORD_SIZE)
        struct.pack_into("<H", data, 0, 1)
        struct.pack_into("<I", data, HEADER_SIZE, 513)
        data[HEADER_SIZE + 4] = point_count
        if point_count <= 8:
            struct.pack_into("<I", data, HEADER_SIZE + 5, 641294940)
            data[HEADER_SIZE + 9 : HEADER_SIZE + 26] = bytes(range(17))
        handle = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        handle.write(data)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_parses_verified_container_fields_without_decoding_payload(self) -> None:
        table = parse_let_table(self.make_table())
        self.assertEqual(table.declared_record_count, 1)
        self.assertEqual(table.records[0].satellite_id, 513)
        self.assertEqual(table.records[0].points[0].epoch_bdt_seconds, 641294940)
        self.assertEqual(table.records[0].points[0].orbit_payload, bytes(range(17)))

    def test_rejects_impossible_point_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum is 8"):
            parse_let_table(self.make_table(point_count=9))


if __name__ == "__main__":
    unittest.main()
