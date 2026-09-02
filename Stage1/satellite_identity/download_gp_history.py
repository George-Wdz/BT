#!/usr/bin/env python3
"""Download the historical Qianfan GP windows needed for LET ID matching.

Space-Track credentials are requested interactively and are never written to
disk.  The current Qianfan catalog is cached locally to avoid repeated
CelesTrak downloads and to provide the NORAD candidate list.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from http.cookiejar import CookieJar
from pathlib import Path


CELESTRAK_QIANFAN_JSON = (
    "https://celestrak.org/NORAD/elements/"
    "gp.php?GROUP=QIANFAN&FORMAT=JSON"
)
SPACE_TRACK_LOGIN = "https://www.space-track.org/ajaxauth/login"
SPACE_TRACK_QUERY_ROOT = (
    "https://www.space-track.org/basicspacedata/query/class/gp_history"
)
USER_AGENT = "BT-satellite-identity-research/1.0"


@dataclass(frozen=True)
class HistoryWindow:
    let_version: str
    start: date
    stop: date


HISTORY_WINDOWS = (
    HistoryWindow("0401", date(2026, 3, 18), date(2026, 3, 24)),
    HistoryWindow("0429", date(2026, 5, 17), date(2026, 5, 21)),
    HistoryWindow("0611", date(2026, 6, 26), date(2026, 7, 3)),
    HistoryWindow("0727", date(2026, 8, 11), date(2026, 8, 18)),
)


def parse_history_window(value: str) -> HistoryWindow:
    """Parse VERSION=YYYY-MM-DD:YYYY-MM-DD for targeted recovery downloads."""
    try:
        version, date_range = value.split("=", 1)
        start_text, stop_text = date_range.split(":", 1)
        start, stop = date.fromisoformat(start_text), date.fromisoformat(stop_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "window must be VERSION=YYYY-MM-DD:YYYY-MM-DD"
        ) from error
    if start > stop:
        raise argparse.ArgumentTypeError("window start must not be after stop")
    return HistoryWindow(version.strip(), start, stop)


def split_history_window(window: HistoryWindow, chunk_days: int) -> list[HistoryWindow]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    chunks = []
    start = window.start
    while start <= window.stop:
        stop = min(start + timedelta(days=chunk_days - 1), window.stop)
        chunks.append(HistoryWindow(window.let_version, start, stop))
        start = stop + timedelta(days=1)
    return chunks


def launch_date(object_id: str) -> date:
    """Return the launch date encoded by a COSPAR object ID such as 2024-140A."""
    year = int(object_id[0:4])
    day_of_year = int(object_id[5:8])
    return date(year, 1, 1) + timedelta(days=day_of_year - 1)


def request_bytes(
    opener: urllib.request.OpenerDirector, url: str, data: bytes | None = None
) -> bytes:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": USER_AGENT},
    )
    with opener.open(request, timeout=120) as response:
        return response.read()


def load_or_download_catalog(catalog_path: Path, refresh: bool) -> list[dict]:
    if refresh or not catalog_path.exists():
        print("Downloading current Qianfan catalog from CelesTrak...", flush=True)
        # curl is used here because urllib connections to CelesTrak can stall on
        # this server even though the same HTTPS endpoint is reachable by curl.
        result = subprocess.run(
            [
                "curl", "-fLsS", "--connect-timeout", "10", "--max-time", "60",
                CELESTRAK_QIANFAN_JSON,
            ],
            check=True,
            capture_output=True,
            timeout=70,
        )
        payload = result.stdout
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_bytes(payload)
        print(f"Downloaded current Qianfan catalog: {catalog_path}", flush=True)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    required = {"NORAD_CAT_ID", "OBJECT_ID", "OBJECT_NAME"}
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("CelesTrak catalog is empty or malformed")
    if not required.issubset(catalog[0]):
        raise ValueError("CelesTrak catalog lacks required OMM fields")
    return catalog


def login(identity: str, password: str) -> urllib.request.OpenerDirector:
    cookies = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    form = urllib.parse.urlencode(
        {"identity": identity, "password": password}
    ).encode("utf-8")
    payload = request_bytes(opener, SPACE_TRACK_LOGIN, form)
    response_text = payload.decode("utf-8", errors="replace").lower()
    if "login failed" in response_text or '"login":"failed"' in response_text:
        raise RuntimeError("Space-Track login failed")
    if not list(cookies):
        raise RuntimeError("Space-Track login returned no session cookie")
    return opener


def build_history_url(norad_ids: list[int], window: HistoryWindow) -> str:
    id_list = ",".join(str(value) for value in norad_ids)
    # Space-Track's upper EPOCH bound is exclusive. CLI windows are inclusive,
    # so query through midnight after the requested final day.
    stop_exclusive = window.stop + timedelta(days=1)
    epoch_range = f"{window.start.isoformat()}--{stop_exclusive.isoformat()}"
    return (
        f"{SPACE_TRACK_QUERY_ROOT}/NORAD_CAT_ID/{id_list}"
        f"/EPOCH/{epoch_range}/orderby/NORAD_CAT_ID,EPOCH/format/csv"
    )


def validate_history_csv(payload: bytes, expected_ids: set[int]) -> tuple[int, int]:
    if not payload.strip():
        return 0, 0
    text = payload.decode("utf-8-sig", errors="strict")
    reader = csv.DictReader(io.StringIO(text))
    required = {"NORAD_CAT_ID", "EPOCH"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        preview = text[:300].replace("\n", " ")
        raise ValueError(f"invalid Space-Track CSV response: {preview}")
    rows = list(reader)
    returned_ids = {int(row["NORAD_CAT_ID"]) for row in rows}
    unexpected = returned_ids - expected_ids
    if unexpected:
        raise ValueError(f"Space-Track returned unexpected NORAD IDs: {unexpected}")
    return len(rows), len(returned_ids)


def merge_history_payloads(payloads: list[bytes]) -> bytes:
    fieldnames = None
    rows: dict[tuple[str, str, str], dict] = {}
    for payload in payloads:
        if not payload.strip():
            continue
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
        if reader.fieldnames is None:
            raise ValueError("historical GP chunk has no CSV header")
        if fieldnames is None:
            fieldnames = reader.fieldnames
        elif reader.fieldnames != fieldnames:
            raise ValueError("historical GP chunks have inconsistent columns")
        for row in reader:
            key = (row["NORAD_CAT_ID"], row["EPOCH"], row.get("GP_ID", ""))
            rows[key] = row
    if fieldnames is None:
        raise ValueError("no historical GP payloads to merge")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows[key] for key in sorted(rows, key=lambda item: (int(item[0]), item[1], item[2])))
    return output.getvalue().encode("utf-8")


def credential(prompt: str, env_name: str, secret: bool = False) -> str:
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    return (getpass.getpass(prompt) if secret else input(prompt)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis" / "history_gp",
    )
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="download a fresh CelesTrak catalog instead of using the local cache",
    )
    parser.add_argument(
        "--window", action="append", type=parse_history_window,
        help=(
            "override default windows; repeat as needed, for example "
            "--window 0727=2026-07-09:2026-08-19"
        ),
    )
    parser.add_argument(
        "--chunk-days", type=int, default=7,
        help="split each requested history window into this many days per request",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    chunk_dir = output_dir / ".chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "qianfan_current_catalog.json"
    catalog = load_or_download_catalog(catalog_path, args.refresh_catalog)

    identity = credential("Space-Track account email: ", "SPACETRACK_IDENTITY")
    password = credential(
        "Space-Track password (input hidden): ", "SPACETRACK_PASSWORD", secret=True
    )
    if not identity or not password:
        raise ValueError("Space-Track identity and password are required")

    print("Logging in to Space-Track...", flush=True)
    opener = login(identity, password)
    summaries: list[dict] = []
    windows = tuple(args.window) if args.window else HISTORY_WINDOWS
    if len({window.let_version for window in windows}) != len(windows):
        raise ValueError("each LET version may appear only once per download run")
    for window in windows:
        candidates = [
            row
            for row in catalog
            if launch_date(str(row["OBJECT_ID"])) <= window.stop
        ]
        norad_ids = sorted(int(row["NORAD_CAT_ID"]) for row in candidates)
        chunks = split_history_window(window, args.chunk_days)
        payloads = []
        chunk_summaries = []
        for index, chunk in enumerate(chunks, start=1):
            url = build_history_url(norad_ids, chunk)
            chunk_path = chunk_dir / (
                f"qianfan_gp_history_{window.let_version}_"
                f"{chunk.start:%Y%m%d}_{chunk.stop:%Y%m%d}.csv"
            )
            print(
                f"Downloading LET {window.let_version} chunk {index}/{len(chunks)}: "
                f"{chunk.start} to {chunk.stop}, candidates={len(norad_ids)}",
                flush=True,
            )
            if chunk_path.exists():
                chunk_payload = chunk_path.read_bytes()
                print(f"Using cached chunk: {chunk_path}", flush=True)
            else:
                chunk_payload = request_bytes(opener, url)
            chunk_rows, chunk_satellites = validate_history_csv(
                chunk_payload, set(norad_ids)
            )
            if chunk_rows == 0:
                raise ValueError(
                    f"Space-Track returned no GP rows for {window.let_version} "
                    f"{chunk.start} to {chunk.stop}; retry this chunk later"
                )
            if not chunk_path.exists():
                chunk_path.write_bytes(chunk_payload)
            payloads.append(chunk_payload)
            chunk_summaries.append({
                "start": chunk.start.isoformat(), "stop": chunk.stop.isoformat(),
                "element_sets": chunk_rows, "returned_satellites": chunk_satellites,
                "cache_path": str(chunk_path),
            })
        payload = merge_history_payloads(payloads)
        row_count, satellite_count = validate_history_csv(payload, set(norad_ids))
        output_path = output_dir / f"qianfan_gp_history_{window.let_version}.csv"
        output_path.write_bytes(payload)
        summaries.append(
            {
                "let_version": window.let_version,
                "start": window.start.isoformat(),
                "stop": window.stop.isoformat(),
                "candidate_satellites": len(norad_ids),
                "returned_satellites": satellite_count,
                "element_sets": row_count,
                "chunks": chunk_summaries,
                "path": str(output_path),
            }
        )
        print(f"Saved {row_count} element sets: {output_path}", flush=True)

    summary_path = output_dir / "download_summary.json"
    summary_path.write_text(
        json.dumps({"windows": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Completed. Summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"Network request failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
