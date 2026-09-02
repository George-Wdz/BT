#!/usr/bin/env python3
"""Build an operational canonical map including authorized LET continuity."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CONTINUITY_EVIDENCE = "numeric_continuity+historical_tle+let_orbit"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-mapping", type=Path,
        default=module_dir / "analysis/latest/historical_physical_mapping.csv",
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=module_dir / "analysis/latest/operational_canonical_0727_mapping.csv",
    )
    parser.add_argument(
        "--base-mapping", type=Path,
        help="retain previously validated operational rows when targets do not conflict",
    )
    args = parser.parse_args()

    with args.physical_mapping.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    accepted_latest = defaultdict(list)
    for row in rows:
        if row["let_version"] == "0727" and row["status"] == "accepted" and row["norad_id"]:
            accepted_latest[int(row["norad_id"])].append(row)

    output = []
    for row in rows:
        if not row["norad_id"]:
            continue
        latest = accepted_latest.get(int(row["norad_id"]), [])
        if len(latest) != 1:
            continue
        if row["status"] == "accepted":
            policy = "strict_multi_evidence"
        elif row["status"] == "provisional" and row["evidence"] == CONTINUITY_EVIDENCE:
            policy = "authorized_same_numeric_id_continuity_with_let_orbit"
        else:
            continue
        output.append({
            "source_let_version": row["let_version"],
            "raw_satellite_id": row["raw_satellite_id"],
            "norad_id": row["norad_id"],
            "physical_name": row["physical_name"],
            "object_id": row["object_id"],
            "canonical_0727_satellite_id": latest[0]["raw_satellite_id"],
            "policy": policy,
            "evidence": row["evidence"],
        })
    if args.base_mapping:
        by_key = {
            (row["source_let_version"], int(row["raw_satellite_id"])): row
            for row in output
        }
        with args.base_mapping.open(encoding="utf-8", newline="") as handle:
            for base in csv.DictReader(handle):
                key = (base["source_let_version"], int(base["raw_satellite_id"]))
                existing = by_key.get(key)
                if existing is not None:
                    if (
                        existing["canonical_0727_satellite_id"]
                        != base["canonical_0727_satellite_id"]
                    ):
                        raise ValueError(f"conflicting operational mapping for {key}")
                    continue
                retained = dict(base)
                retained["policy"] = f"retained_previous:{base.get('policy', 'validated')}"
                by_key[key] = retained
        output = list(by_key.values())
    if not output:
        raise ValueError("no operational mappings were produced")
    output.sort(key=lambda row: (row["source_let_version"], int(row["raw_satellite_id"])))
    write_csv(args.output_path, output)
    summary = {
        "physical_mapping": str(args.physical_mapping.resolve()),
        "base_mapping": str(args.base_mapping.resolve()) if args.base_mapping else None,
        "output_mapping": str(args.output_path.resolve()),
        "rows": len(output),
        "changed_rows": sum(
            row["raw_satellite_id"] != row["canonical_0727_satellite_id"]
            for row in output
        ),
        "policy_counts": dict(Counter(row["policy"] for row in output)),
        "version_counts": dict(Counter(row["source_let_version"] for row in output)),
    }
    summary_path = args.output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
