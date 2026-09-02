from datetime import datetime, timezone
import unittest

from satellite_identity.match_historical_gp import local_time_to_utc


class MatchHistoricalGpTest(unittest.TestCase):
    def test_converts_receiver_local_time_to_utc(self) -> None:
        self.assertEqual(
            local_time_to_utc("2026-08-25T12:34:56.123456"),
            datetime(2026, 8, 25, 4, 34, 56, 123456, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
