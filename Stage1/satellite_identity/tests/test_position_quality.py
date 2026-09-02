import unittest

from satellite_identity.position_quality import (
    ecef_to_geodetic,
    position_quality_reason,
    version_for_local_time,
)


class PositionQualityTest(unittest.TestCase):
    def test_rejects_incomplete_equatorial_placeholder(self):
        row = {
            "satAltitude": 121863.0,
            "longitude": -105.6431,
            "latitude": 0.0,
            "altitude": 73.0,
            "posLongitude": 121.416,
            "posLatitude": 31.2185,
            "ecefPx": -1752692.7219,
            "ecefPy": -6259238.6296,
            "ecefPz": 0.0,
        }
        self.assertEqual(position_quality_reason(row), "zero_latitude")

    def test_converts_equatorial_ecef(self):
        longitude, latitude, altitude = ecef_to_geodetic((6_878_137.0, 0.0, 0.0))
        self.assertAlmostEqual(longitude, 0.0, places=8)
        self.assertAlmostEqual(latitude, 0.0, places=8)
        self.assertAlmostEqual(altitude, 500_000.0, places=4)

    def test_selects_effective_catalog(self):
        self.assertEqual(version_for_local_time("2026-04-20T00:00:00"), "0401")
        self.assertEqual(version_for_local_time("2026-06-01T00:00:00"), "0611")
        self.assertEqual(version_for_local_time("2026-08-01T00:00:00"), "0727")


if __name__ == "__main__":
    unittest.main()
