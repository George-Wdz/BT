#!/usr/bin/env python3
"""Collect link-reliability CSV artifacts into one review-friendly workbook."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SHEETS = {
    "monthly_summary": "monthly_summary.csv",
    "network_continuity": "network_continuity_summary.csv",
    "inter_satellite_gaps": "inter_satellite_gaps.csv",
    "rain_rate_summary": "rain_rate_summary.csv",
    "quality_by_month": "raw_quality_summary.csv",
    "quality_by_rain": "raw_quality_by_rain.csv",
    "diagnostic_causes": "diagnostic_cause_summary.csv",
    "satellite_monthly": "satellite_monthly_summary.csv",
    "conditions": "condition_summary.csv",
    "pass_diagnostics": "pass_diagnostics.csv",
}


def main() -> None:
    output = ARTIFACTS / "link_reliability_analysis_summary.xlsx"
    provenance = json.loads((ARTIFACTS / "provenance.json").read_text(encoding="utf-8"))
    provenance_rows = []
    for item in provenance:
        provenance_rows.extend(
            {"role": item.get("role"), "field": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value}
            for key, value in item.items()
            if key != "role"
        )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(provenance_rows).to_excel(writer, sheet_name="provenance", index=False)
        for sheet, filename in SHEETS.items():
            pd.read_csv(ARTIFACTS / filename).to_excel(writer, sheet_name=sheet, index=False)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
