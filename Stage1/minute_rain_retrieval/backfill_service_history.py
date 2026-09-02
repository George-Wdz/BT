#!/usr/bin/env python3
"""Backfill minute-rain history through the running inference service."""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def completed_dates(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "ok" and record.get("date"):
            completed.add(str(record["date"]))
    return completed


def request_day(
    base_url: str,
    current_date: date,
    max_passes: int,
    timeout_seconds: float,
    retries: int,
) -> dict:
    query = urlencode(
        {
            "date": current_date.isoformat(),
            "max_passes": max_passes,
            "recompute": "true",
        }
    )
    url = f"{base_url.rstrip('/')}/api/rainfall?{query}"
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"request failed after {retries} attempts: {url}"
                ) from exc
            delay = min(5 * attempt, 30)
            print(
                f"date={current_date} attempt={attempt}/{retries} "
                f"error={exc!r} retry_in={delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def terminal_counts(payload: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for terminal in payload.get("terminals", []):
        terminal_id = str(terminal.get("terminal_id", "unknown"))
        summary = terminal.get("summary") or {}
        rows = terminal.get("passes") or []
        counts[terminal_id] = int(summary.get("pass_count", len(rows)))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8041")
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument(
        "--end-date",
        required=True,
        type=parse_date,
        help="Inclusive final date.",
    )
    parser.add_argument("--max-passes", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-path", required=True)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore successful entries already present in the progress file.",
    )
    args = parser.parse_args()

    if args.end_date < args.start_date:
        parser.error("--end-date must not precede --start-date")
    if not 1 <= args.max_passes <= 2000:
        parser.error("--max-passes must be between 1 and 2000")

    progress_path = Path(args.progress_path).expanduser().resolve()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.restart else completed_dates(progress_path)
    total_days = (args.end_date - args.start_date).days + 1
    successful = 0
    skipped = 0

    current = args.start_date
    while current <= args.end_date:
        date_text = current.isoformat()
        if date_text in completed:
            skipped += 1
            print(f"date={date_text} status=skipped_completed", flush=True)
            current += timedelta(days=1)
            continue

        started = time.monotonic()
        try:
            payload = request_day(
                args.base_url,
                current,
                args.max_passes,
                args.timeout_seconds,
                args.retries,
            )
            counts = terminal_counts(payload)
            record = {
                "date": date_text,
                "status": str(payload.get("status", "unknown")),
                "terminal_counts": counts,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            if record["status"] != "ok":
                raise RuntimeError(f"service returned non-ok payload: {payload}")
            successful += 1
            print(
                f"date={date_text} status=ok terminal_counts={counts} "
                f"elapsed={record['elapsed_seconds']}s",
                flush=True,
            )
        except Exception as exc:
            record = {
                "date": date_text,
                "status": "error",
                "error": repr(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            raise

        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        current += timedelta(days=1)

    print(
        f"backfill_complete total_days={total_days} successful={successful} "
        f"skipped={skipped} progress={progress_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
