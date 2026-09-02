#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from data_flow import BuildConfig, build_samples, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gauge-anchored minute rainfall samples")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--terminal-id", default="01-31-0005-0001")
    parser.add_argument("--image-csv")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--rainfall-scale", type=float, default=0.1)
    parser.add_argument("--min-phy-points", type=int, default=10)
    parser.add_argument(
        "--min-snr-db", type=float,
        help="Discard PHY points below this SNR before counting valid points per minute.",
    )
    parser.add_argument(
        "--position-mode", choices=("required", "fallback_mean", "omit"),
        default="required",
    )
    parser.add_argument("--position-tolerance-seconds", type=float, default=5.0)
    parser.add_argument("--weather-tolerance-seconds", type=float, default=60.0)
    parser.add_argument("--image-tolerance-seconds", type=float, default=600.0)
    parser.add_argument("--split-strategy", choices=("stratified_all", "time", "event_holdout"),
                        default="stratified_all")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--holdout-period", nargs=2, action="append", metavar=("START", "END"),
        default=[], help="Event period to reserve for test; repeat for multiple periods.",
    )
    parser.add_argument(
        "--holdout-buffer-minutes", type=float, default=60.0,
        help="Expand each holdout period on both sides to prevent adjacent-window leakage.",
    )
    parser.add_argument(
        "--terminal-protocol", choices=("legacy", "new"), default="legacy"
    )
    parser.add_argument("--shared-db-path")
    parser.add_argument("--shared-terminal-id", default="01-31-0005-0001")
    parser.add_argument("--adapter-path")
    parser.add_argument("--reference-checkpoint-path")
    args = parser.parse_args()
    args.holdout_periods = tuple(tuple(period) for period in args.holdout_period)
    del args.holdout_period
    cfg = BuildConfig(**vars(args))
    samples, metadata = build_samples(cfg)
    if not samples:
        raise RuntimeError("No valid minute samples were built; check source ranges and tolerances")
    save_dataset(samples, metadata, args.output_path)
    print(json.dumps(metadata["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
