from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _cn_number(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if "十" in value:
        left, right = value.split("十", 1)
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    digits = [_CN_DIGITS[c] for c in value if c in _CN_DIGITS]
    if not digits:
        raise ValueError(f"无法解析中文数字：{value}")
    return int("".join(str(x) for x in digits))


def _parse_date(text: str, now: datetime) -> tuple[datetime, bool]:
    explicit = re.search(r"(?:(\d{4})[年/-])?(\d{1,2})[月/-](\d{1,2})日?", text)
    if explicit:
        year = int(explicit.group(1) or now.year)
        return datetime(year, int(explicit.group(2)), int(explicit.group(3))), True
    if "前天" in text:
        return (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0), True
    if "昨天" in text or "昨日" in text:
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0), True
    if "今天" in text or "今日" in text:
        return now.replace(hour=0, minute=0, second=0, microsecond=0), True
    return now.replace(hour=0, minute=0, second=0, microsecond=0), False


def parse_history_time_query(
    text: str,
    *,
    now: datetime | None = None,
    nearest_hours: float = 3.0,
) -> dict[str, Any]:
    """Parse common Chinese relative dates and fuzzy time expressions.

    A precise clock expression becomes a target time with a symmetric search
    window. A date or day-period expression becomes a range query.
    """
    text = text.strip()
    if not text:
        raise ValueError("查询文本不能为空")
    now = (now or datetime.now()).replace(tzinfo=None)

    recent_days = re.search(
        r"(?:过去|最近|近)([零〇一二两三四五六七八九十\d]{1,3})天", text
    )
    if recent_days:
        days = max(_cn_number(recent_days.group(1)), 1)
        start = now - timedelta(days=days)
        return {
            "mode": "range",
            "original_text": text,
            "target_time": None,
            "start": start.isoformat(sep=" ", timespec="seconds"),
            "end": now.isoformat(sep=" ", timespec="seconds"),
            "period": f"最近{days}天",
        }
    if "上周" in text:
        this_monday = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start = this_monday - timedelta(days=7)
        return {
            "mode": "range",
            "original_text": text,
            "target_time": None,
            "start": start.isoformat(sep=" ", timespec="seconds"),
            "end": this_monday.isoformat(sep=" ", timespec="seconds"),
            "period": "上周",
        }
    if "历史" in text and not re.search(r"\d{1,2}[月/-]\d{1,2}", text):
        start = now - timedelta(days=7)
        return {
            "mode": "range",
            "original_text": text,
            "target_time": None,
            "start": start.isoformat(sep=" ", timespec="seconds"),
            "end": now.isoformat(sep=" ", timespec="seconds"),
            "period": "默认最近7天",
        }
    day, date_explicit = _parse_date(text, now)

    periods = {
        "凌晨": (0, 6),
        "早上": (6, 10),
        "上午": (6, 12),
        "中午": (11, 14),
        "下午": (12, 18),
        "傍晚": (17, 20),
        "晚上": (18, 24),
    }
    period = next((name for name in periods if name in text), None)
    clock = re.search(
        r"([零〇一二两三四五六七八九十\d]{1,3})[点时]"
        r"(?:(半)|([零〇一二两三四五六七八九十\d]{1,3})分?)?",
        text,
    )
    if clock:
        hour = _cn_number(clock.group(1))
        minute = 30 if clock.group(2) else (_cn_number(clock.group(3)) if clock.group(3) else 0)
        if period in {"下午", "傍晚", "晚上"} and hour < 12:
            hour += 12
        elif period == "中午" and hour < 11:
            hour += 12
        elif period == "凌晨" and hour == 12:
            hour = 0
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError(f"无效时间：{hour:02d}:{minute:02d}")
        target = day.replace(hour=hour, minute=minute)
        radius = timedelta(hours=max(float(nearest_hours), 0.0))
        return {
            "mode": "nearest",
            "original_text": text,
            "target_time": target.isoformat(sep=" ", timespec="seconds"),
            "start": (target - radius).isoformat(sep=" ", timespec="seconds"),
            "end": (target + radius).isoformat(sep=" ", timespec="seconds"),
            "nearest_hours": float(nearest_hours),
        }

    if period:
        start_hour, end_hour = periods[period]
        start = day.replace(hour=start_hour)
        end = day + timedelta(days=1) if end_hour == 24 else day.replace(hour=end_hour)
        return {
            "mode": "range",
            "original_text": text,
            "target_time": None,
            "start": start.isoformat(sep=" ", timespec="seconds"),
            "end": end.isoformat(sep=" ", timespec="seconds"),
            "period": period,
        }

    if not date_explicit:
        raise ValueError("未识别到日期或时间，请输入如“昨天”“前天下午三点”或“7月12日下午”")
    return {
        "mode": "range",
        "original_text": text,
        "target_time": None,
        "start": day.isoformat(sep=" ", timespec="seconds"),
        "end": (day + timedelta(days=1)).isoformat(sep=" ", timespec="seconds"),
        "period": "全天",
    }


class RainRetrievalHistory:
    """SQLite repository for terminal-aware rainfall retrieval results."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            table_exists = conn.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='rain_retrieval_passes'
                """
            ).fetchone()
            columns = (
                {
                    str(row["name"])
                    for row in conn.execute(
                        "PRAGMA table_info(rain_retrieval_passes)"
                    ).fetchall()
                }
                if table_exists
                else set()
            )
            obsolete = {"model_version", "status", "checkpoint_path"}
            if table_exists and (
                "terminal_id" not in columns or bool(columns & obsolete)
            ):
                self._migrate_to_simple_table(conn, columns)
            materialized_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(rain_history_materialized_days)"
                ).fetchall()
            }
            if materialized_columns and "model_version" not in materialized_columns:
                conn.execute("DROP TABLE rain_history_materialized_days")
            conn.executescript(
                self._create_table_sql()
                + """
                CREATE INDEX IF NOT EXISTS idx_rain_history_terminal_time
                    ON rain_retrieval_passes(
                        terminal_id, pass_start, pass_end
                    );
                CREATE INDEX IF NOT EXISTS idx_rain_history_time
                    ON rain_retrieval_passes(pass_start, pass_end);
                CREATE INDEX IF NOT EXISTS idx_rain_history_terminal_end
                    ON rain_retrieval_passes(terminal_id, pass_end);
                CREATE INDEX IF NOT EXISTS idx_rain_history_sat_time
                    ON rain_retrieval_passes(
                        satellite_id, pass_start, pass_end
                    );
                CREATE INDEX IF NOT EXISTS idx_rain_history_observed_rain_date
                    ON rain_retrieval_passes(substr(pass_start, 1, 10))
                    WHERE observed_available = 1
                      AND observed_rainfall_mm > 0;
                CREATE TABLE IF NOT EXISTS rain_history_materialized_days (
                    terminal_id TEXT NOT NULL,
                    query_date TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY(terminal_id, query_date, model_version)
                );
                """
            )
            self._normalize_timestamp_keys(conn)

    @staticmethod
    def _normalize_timestamp_keys(conn: sqlite3.Connection) -> None:
        """Canonicalize legacy timestamps and merge equivalent pass keys."""
        needs_normalization = conn.execute(
            """
            SELECT 1 FROM rain_retrieval_passes
            WHERE instr(pass_start, ' ') > 0 OR instr(pass_end, ' ') > 0
            LIMIT 1
            """
        ).fetchone()
        if needs_normalization is None:
            return
        conn.execute(
            """
            DELETE FROM rain_retrieval_passes
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY
                                   terminal_id,
                                   satellite_id,
                                   datetime(pass_start)
                               ORDER BY
                                   datetime(inferred_at) DESC,
                                   datetime(updated_at) DESC,
                                   id DESC
                           ) AS row_rank
                    FROM rain_retrieval_passes
                )
                WHERE row_rank > 1
            )
            """
        )
        conn.execute(
            """
            UPDATE rain_retrieval_passes
            SET pass_start = replace(pass_start, ' ', 'T'),
                pass_end = replace(pass_end, ' ', 'T')
            WHERE instr(pass_start, ' ') > 0 OR instr(pass_end, ' ') > 0
            """
        )

    @staticmethod
    def _create_table_sql(table_name: str = "rain_retrieval_passes") -> str:
        return f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    terminal_id TEXT NOT NULL
                        DEFAULT '01-31-0005-0001',
                    satellite_id INTEGER NOT NULL,
                    pass_start TEXT NOT NULL,
                    pass_end TEXT NOT NULL,
                    points INTEGER NOT NULL,
                    pred_rainfall_mm REAL NOT NULL,
                    reported_rainfall_mm REAL NOT NULL,
                    rainfall_level TEXT NOT NULL,
                    rain_probability REAL,
                    rain_rate_mean REAL,
                    rain_rate_max REAL,
                    rainy_ratio REAL,
                    prob_sunny REAL,
                    prob_cloudy REAL,
                    prob_rain REAL,
                    image_available INTEGER,
                    observed_rainfall_mm REAL,
                    observed_available INTEGER NOT NULL DEFAULT 0,
                    observed_reason TEXT,
                    absolute_error_mm REAL,
                    checkpoint_satellite_known INTEGER,
                    baseline_source TEXT,
                    position_source TEXT,
                    position_available_ratio REAL,
                    transfer_mode TEXT,
                    inferred_at TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(
                        terminal_id, satellite_id, pass_start
                    )
                );
                """

    def _migrate_to_simple_table(
        self, conn: sqlite3.Connection, columns: set[str]
    ) -> None:
        """Remove deployment metadata while preserving the newest pass result."""
        def source(name: str, fallback: str = "NULL") -> str:
            return name if name in columns else fallback

        terminal = source("terminal_id", "'01-31-0005-0001'")
        selected = [
            f"{terminal} AS terminal_id",
            "satellite_id", "pass_start", "pass_end", "points",
            "pred_rainfall_mm", "reported_rainfall_mm", "rainfall_level",
            "rain_probability", "rain_rate_mean", "rain_rate_max", "rainy_ratio",
            "prob_sunny", "prob_cloudy", "prob_rain", "image_available",
            source("observed_rainfall_mm") + " AS observed_rainfall_mm",
            source("observed_available", "0") + " AS observed_available",
            source("observed_reason") + " AS observed_reason",
            source("absolute_error_mm") + " AS absolute_error_mm",
            source("checkpoint_satellite_known")
            + " AS checkpoint_satellite_known",
            source("baseline_source") + " AS baseline_source",
            source("position_source") + " AS position_source",
            source("position_available_ratio") + " AS position_available_ratio",
            source("transfer_mode") + " AS transfer_mode",
            "inferred_at", "created_at", "updated_at",
        ]
        target_columns = [
            "terminal_id", "satellite_id", "pass_start", "pass_end", "points",
            "pred_rainfall_mm", "reported_rainfall_mm", "rainfall_level",
            "rain_probability", "rain_rate_mean", "rain_rate_max", "rainy_ratio",
            "prob_sunny", "prob_cloudy", "prob_rain", "image_available",
            "observed_rainfall_mm", "observed_available", "observed_reason",
            "absolute_error_mm", "checkpoint_satellite_known", "baseline_source",
            "position_source", "position_available_ratio", "transfer_mode",
            "inferred_at", "created_at", "updated_at",
        ]
        conn.executescript(
            "DROP TABLE IF EXISTS rain_retrieval_passes_simple;"
            + self._create_table_sql("rain_retrieval_passes_simple")
        )
        conn.execute(
            f"""
            INSERT INTO rain_retrieval_passes_simple (
                {",".join(target_columns)}
            )
            SELECT {",".join(selected)}
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY {terminal}, satellite_id, pass_start
                           ORDER BY updated_at DESC, id DESC
                       ) AS row_rank
                FROM rain_retrieval_passes
            )
            WHERE row_rank = 1
            """
        )
        conn.executescript(
            """
            DROP TABLE rain_retrieval_passes;
            ALTER TABLE rain_retrieval_passes_simple
                RENAME TO rain_retrieval_passes;
            DROP TABLE IF EXISTS rain_history_materialized_days;
            """
        )

    def upsert_many(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        columns = [
            "terminal_id", "satellite_id", "pass_start", "pass_end", "points",
            "pred_rainfall_mm", "reported_rainfall_mm", "rainfall_level", "rain_probability",
            "rain_rate_mean", "rain_rate_max", "rainy_ratio", "prob_sunny", "prob_cloudy",
            "prob_rain", "image_available",
            "observed_rainfall_mm", "observed_available", "observed_reason",
            "absolute_error_mm", "checkpoint_satellite_known", "baseline_source",
            "position_source", "position_available_ratio", "transfer_mode",
            "inferred_at",
        ]
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(
            f"{name}=excluded.{name}" for name in columns
            if name not in {
                "terminal_id", "satellite_id", "pass_start"
            }
        ) + ",updated_at=CURRENT_TIMESTAMP"
        sql = (
            f"INSERT INTO rain_retrieval_passes ({','.join(columns)}) VALUES ({placeholders}) "
            "ON CONFLICT(terminal_id, satellite_id, pass_start) "
            f"DO UPDATE SET {updates}"
        )
        values = []
        for row in rows:
            normalized = dict(row)
            normalized.setdefault("terminal_id", "01-31-0005-0001")
            for name in ("pass_start", "pass_end"):
                value = normalized.get(name)
                if value is not None:
                    normalized[name] = datetime.fromisoformat(
                        str(value)
                    ).isoformat()
            normalized.setdefault(
                "rainfall_level",
                "rain"
                if float(normalized.get("reported_rainfall_mm", 0.0)) > 0
                else "no_rain",
            )
            normalized.setdefault("observed_available", 0)
            normalized.setdefault("transfer_mode", "native_001")
            values.append([normalized.get(name) for name in columns])
        with self._connect() as conn:
            conn.executemany(sql, values)
        return len(values)

    @staticmethod
    def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        return [dict(row) for row in cursor.fetchall()]

    def query_range(
        self,
        start: str,
        end: str,
        *,
        terminal_id: str | None = None,
        satellite_id: int | None = None,
        model_version: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        where = [
            "pass_end >= ?",
            "pass_start <= ?",
        ]
        params: list[Any] = [start, end]
        if terminal_id:
            where.append("terminal_id = ?")
            params.append(terminal_id)
        if satellite_id is not None:
            where.append("satellite_id = ?")
            params.append(int(satellite_id))
        params.append(max(1, min(int(limit), 10000)))
        sql = (
            "SELECT * FROM rain_retrieval_passes WHERE " + " AND ".join(where)
            + " ORDER BY pass_start ASC LIMIT ?"
        )
        with self._connect() as conn:
            return self._rows(conn.execute(sql, params))

    def latest(
        self,
        *,
        terminal_id: str | None = None,
        model_version: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if terminal_id:
            where.append("terminal_id = ?")
            params.append(terminal_id)
        params.append(max(1, min(int(limit), 200)))
        where_sql = " WHERE " + " AND ".join(where) if where else ""
        with self._connect() as conn:
            return self._rows(conn.execute(
                "SELECT * FROM rain_retrieval_passes" + where_sql
                + " ORDER BY pass_end DESC LIMIT ?",
                params,
            ))

    def latest_rainy(
        self,
        *,
        terminal_id: str | None = None,
        model_version: str | None = None,
        satellite_id: int | None = None,
        min_rainfall_mm: float = 0.05,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        where = ["reported_rainfall_mm >= ?"]
        params: list[Any] = [float(min_rainfall_mm)]
        if terminal_id:
            where.append("terminal_id = ?")
            params.append(terminal_id)
        if satellite_id is not None:
            where.append("satellite_id = ?")
            params.append(int(satellite_id))
        params.append(max(1, min(int(limit), 200)))
        with self._connect() as conn:
            return self._rows(conn.execute(
                "SELECT * FROM rain_retrieval_passes WHERE " + " AND ".join(where)
                + " ORDER BY pass_end DESC LIMIT ?",
                params,
            ))

    def query_nearest(
        self,
        target_time: str,
        start: str,
        end: str,
        *,
        terminal_id: str | None = None,
        satellite_id: int | None = None,
        model_version: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self.query_range(
            start, end, terminal_id=terminal_id, satellite_id=satellite_id,
            model_version=model_version, limit=2000
        )
        target = datetime.fromisoformat(target_time)

        def distance(row: dict[str, Any]) -> float:
            start_dt = datetime.fromisoformat(row["pass_start"])
            end_dt = datetime.fromisoformat(row["pass_end"])
            if start_dt <= target <= end_dt:
                return 0.0
            return min(abs((target - start_dt).total_seconds()), abs((target - end_dt).total_seconds()))

        rows.sort(key=lambda row: (distance(row), row["pass_start"], row["satellite_id"]))
        rows = rows[: max(1, min(int(limit), 200))]
        for row in rows:
            row["distance_to_query_s"] = round(distance(row), 3)
            row["contains_query_time"] = row["distance_to_query_s"] == 0.0
        return rows

    def search_text(
        self,
        text: str,
        *,
        terminal_id: str | None = None,
        satellite_id: int | None = None,
        model_version: str | None = None,
        nearest_hours: float = 3.0,
        limit: int = 20,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        parsed = parse_history_time_query(text, now=now, nearest_hours=nearest_hours)
        if parsed["mode"] == "nearest":
            rows = self.query_nearest(
                parsed["target_time"], parsed["start"], parsed["end"],
                terminal_id=terminal_id, satellite_id=satellite_id,
                model_version=model_version, limit=limit,
            )
        else:
            rows = self.query_range(
                parsed["start"], parsed["end"], terminal_id=terminal_id,
                satellite_id=satellite_id, model_version=model_version, limit=limit,
            )
        return {"query": parsed, "count": len(rows), "results": rows}

    def query_day(
        self,
        query_date,
        *,
        terminal_id: str,
        model_version: str,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        start = datetime.combine(query_date, datetime.min.time())
        end = start + timedelta(days=1)
        return self.query_range(
            start.isoformat(),
            end.isoformat(),
            terminal_id=terminal_id,
            limit=limit,
        )

    def mark_day_materialized(
        self, query_date, terminal_id: str, model_version: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rain_history_materialized_days(
                    terminal_id, query_date, model_version, completed_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(terminal_id, query_date, model_version)
                DO UPDATE SET completed_at=excluded.completed_at
                """,
                [
                    terminal_id,
                    query_date.isoformat(),
                    model_version,
                    datetime.now().isoformat(timespec="seconds"),
                ],
            )

    def is_day_materialized(
        self, query_date, terminal_id: str, model_version: str
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM rain_history_materialized_days
                WHERE terminal_id = ?
                  AND query_date = ?
                  AND model_version = ?
                """,
                [terminal_id, query_date.isoformat(), model_version],
            ).fetchone()
        return row is not None

    def rainy_dates(self) -> list[dict[str, Any]]:
        """Return dates with gauge-observed rain in materialized history."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT substr(pass_start, 1, 10) AS query_date,
                       COUNT(*) AS observed_rows,
                       COUNT(DISTINCT terminal_id) AS terminals,
                       MAX(observed_rainfall_mm) AS max_minute_rainfall_mm
                FROM rain_retrieval_passes
                WHERE observed_available = 1
                  AND observed_rainfall_mm > 0
                GROUP BY substr(pass_start, 1, 10)
                ORDER BY query_date
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS records, MIN(pass_start) AS earliest_pass,
                          MAX(pass_end) AS latest_pass,
                          COUNT(DISTINCT terminal_id) AS terminals,
                          COUNT(DISTINCT satellite_id) AS satellites,
                          SUM(observed_available) AS observed_records
                   FROM rain_retrieval_passes"""
            ).fetchone()
            per_terminal = conn.execute(
                """
                SELECT terminal_id, COUNT(*) AS records,
                       MIN(pass_start) AS earliest_pass,
                       MAX(pass_end) AS latest_pass
                FROM rain_retrieval_passes
                GROUP BY terminal_id
                ORDER BY terminal_id
                """
            ).fetchall()
        return {
            "db_path": str(self.db_path),
            **dict(row),
            "per_terminal": [dict(item) for item in per_terminal],
        }

    def finalize_due(self, cutoff: datetime, *, model_version: str | None = None) -> int:
        # Retained as a no-op for compatibility with older workers.
        return 0
