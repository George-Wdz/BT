from __future__ import annotations

import unittest
from datetime import date

from satellite_identity.download_gp_history import (
    HistoryWindow,
    build_history_url,
    launch_date,
    validate_history_csv,
    parse_history_window,
    split_history_window,
    merge_history_payloads,
)


class DownloadGpHistoryTest(unittest.TestCase):
    def test_parses_custom_history_window(self):
        window = parse_history_window("0727=2026-07-09:2026-08-19")
        self.assertEqual(window.let_version, "0727")
        self.assertEqual(window.start.isoformat(), "2026-07-09")
        self.assertEqual(window.stop.isoformat(), "2026-08-19")

    def test_splits_and_merges_history_chunks(self):
        window = parse_history_window("0611=2026-05-27:2026-06-11")
        chunks = split_history_window(window, 7)
        self.assertEqual(
            [(item.start.isoformat(), item.stop.isoformat()) for item in chunks],
            [
                ("2026-05-27", "2026-06-02"),
                ("2026-06-03", "2026-06-09"),
                ("2026-06-10", "2026-06-11"),
            ],
        )
        first = b"NORAD_CAT_ID,EPOCH,GP_ID\n60379,2026-06-01T00:00:00,1\n"
        second = b"NORAD_CAT_ID,EPOCH,GP_ID\n60379,2026-06-01T00:00:00,1\n60379,2026-06-02T00:00:00,2\n"
        merged = merge_history_payloads([first, second])
        self.assertEqual(validate_history_csv(merged, {60379}), (2, 1))

    def test_parses_cospar_launch_date(self) -> None:
        self.assertEqual(launch_date("2024-140A"), date(2024, 5, 19))

    def test_builds_gp_history_query(self) -> None:
        window = HistoryWindow("test", date(2026, 3, 18), date(2026, 3, 24))
        url = build_history_url([60379, 60380], window)
        self.assertIn("NORAD_CAT_ID/60379,60380", url)
        self.assertIn("EPOCH/2026-03-18--2026-03-25", url)
        self.assertTrue(url.endswith("format/csv"))

    def test_validates_expected_csv(self) -> None:
        payload = (
            b"NORAD_CAT_ID,EPOCH,OBJECT_NAME\n"
            b"60379,2026-03-20T00:00:00.000000,QIANFAN-1\n"
        )
        self.assertEqual(validate_history_csv(payload, {60379}), (1, 1))

    def test_rejects_non_csv_login_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid Space-Track CSV"):
            validate_history_csv(b'{"error":"login required"}', {60379})


if __name__ == "__main__":
    unittest.main()
