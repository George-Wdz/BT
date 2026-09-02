import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_flow import GEO_COLUMNS, match_position_same_satellite
from model import MinuteRainTransformer
from dataset import MinuteRainDataset
from train import DynamicDryDownsampleSampler


def test_position_match_requires_same_satellite_id():
    phy = pd.DataFrame({
        "localTime": pd.to_datetime(["2026-01-01 00:00:01", "2026-01-01 00:00:01"]),
        "satelliteId": [1, 2], "phyRssi": [-70, -80], "rssi": [-71, -81],
        "snr": [10, 9], "lastCniValue": [8, 7],
    })
    position = pd.DataFrame({
        "localTime": pd.to_datetime(["2026-01-01 00:00:01", "2026-01-01 00:00:03"]),
        "satId": [1, 2], "slant_range_km": [100, 200], "elevation_deg": [50, 40],
        "azimuth_sin": [0.1, 0.2], "azimuth_cos": [0.9, 0.8],
    })
    result = match_position_same_satellite(phy, position, 5)
    assert result.loc[result.satelliteId == 1, GEO_COLUMNS[0]].item() == 100
    assert result.loc[result.satelliteId == 2, GEO_COLUMNS[0]].item() == 200


def test_model_returns_one_value_per_minute_not_per_phy_point():
    model = MinuteRainTransformer(input_dim=20, num_satellites=3, d_model=32,
                                  num_heads=4, num_layers=1, d_ff=64, max_points=16)
    model.eval()
    with torch.no_grad():
        output = model(
            torch.randn(2, 11, 20),
            torch.ones(2, 11, dtype=torch.long),
            torch.tensor([[True] * 7 + [False] * 4, [True] * 11]),
        )
    assert output["prediction"].shape == (2,)
    assert output["rain_logit"].shape == (2,)


def test_hard_snr_mask_excludes_only_low_quality_points():
    model = MinuteRainTransformer(
        input_dim=20, num_satellites=3, d_model=32, num_heads=4,
        num_layers=1, d_ff=64, max_points=8,
        snr_quality_mode="hard_mask", snr_threshold_db=-10,
    )
    model.eval()
    valid = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        output = model(
            torch.randn(1, 4, 20), torch.ones(1, 4, dtype=torch.long), valid,
            torch.tensor([[-20.0, -10.0, 4.0, 8.0]]),
        )
    assert output["attention_valid_mask"].tolist() == [[False, True, True, False]]


def test_soft_snr_gate_keeps_context_tokens_and_returns_continuous_weights():
    model = MinuteRainTransformer(
        input_dim=20, num_satellites=3, d_model=32, num_heads=4,
        num_layers=1, d_ff=64, max_points=8,
        snr_quality_mode="soft_gate", snr_threshold_db=-10,
        snr_gate_temperature_db=2,
    )
    model.eval()
    valid = torch.tensor([[True, True, True]])
    with torch.no_grad():
        output = model(
            torch.randn(1, 3, 20), torch.ones(1, 3, dtype=torch.long), valid,
            torch.tensor([[-20.0, -10.0, 0.0]]),
        )
    weights = output["quality_weight"][0]
    assert weights[0] < weights[1] < weights[2]
    assert torch.allclose(weights[1], torch.tensor(0.5))
    assert output["attention_valid_mask"].tolist() == valid.tolist()


def test_dry_downsampling_keeps_all_rainy_samples():
    samples = [
        {"minute_rainfall_mm": value} for value in [0.02, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    dataset = object.__new__(MinuteRainDataset)
    dataset.samples = samples
    sampler = DynamicDryDownsampleSampler(dataset, rain_threshold=0.005,
                                           max_dry_to_rain_ratio=1.5, seed=1)
    indices = list(iter(sampler))
    assert {0, 1}.issubset(indices)
    assert len(indices) == 5
    rain_only = DynamicDryDownsampleSampler(
        dataset, rain_threshold=0.005, max_dry_to_rain_ratio=0, seed=1
    )
    assert set(iter(rain_only)) == {0, 1}
