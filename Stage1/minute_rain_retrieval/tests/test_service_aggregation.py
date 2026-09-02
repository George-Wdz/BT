import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import MinuteThreeTerminalRunner


def _row(terminal_id: str, amount: float) -> dict:
    return {
        "terminal_id": terminal_id,
        "satellite_id": 1,
        "pass_start": "2026-07-07T00:00:00",
        "pass_end": "2026-07-07T00:01:00",
        "reported_rainfall_mm": amount,
        "observed_rainfall_mm": 0.02,
        "observed_available": 1,
        "inferred_at": "2026-08-17T00:00:00",
    }


def test_minute_cumulative_sums_one_prediction_per_anchor():
    rows = [_row("01-31-0005-0001", 0.02), _row("01-31-0005-0002", 0.03)]
    series, consistency = MinuteThreeTerminalRunner._minute_model_time_series(
        rows,
        pd.Timestamp("2026-07-07T00:00:00"),
        pd.Timestamp("2026-07-07T00:03:00"),
        1,
    )
    assert series["01-31-0005-0001"]["coverage_cumulative_mm"][-1][1] == 0.02
    assert series["01-31-0005-0002"]["coverage_cumulative_mm"][-1][1] == 0.03
    assert (
        series["01-31-0005-0001"]["observed_coverage_cumulative_mm"][-1][1]
        == 0.02
    )
    assert (
        series["01-31-0005-0002"]["observed_coverage_cumulative_mm"][-1][1]
        == 0.02
    )
    assert series["consensus"]["coverage_cumulative_mm"][-1][1] == 0.025
    assert series["consensus"]["observed_coverage_cumulative_mm"][-1][1] == 0.02
    assert consistency["consensus_bins"] == 1


def test_minute_consistency_pairs_identical_anchor_times():
    rows = [_row("01-31-0005-0001", 0.02), _row("01-31-0005-0002", 0.03)]
    groups, summary = MinuteThreeTerminalRunner._minute_consistency_groups(rows)
    assert len(groups) == 1
    assert groups[0]["terminal_count"] == 2
    assert groups[0]["rate_range_mm_h"] == 0.6
    assert summary["rain_decision_agreement"] == 1.0
