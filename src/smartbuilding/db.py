"""PostgreSQL / TimescaleDB implementation of the scorer Sink."""

from __future__ import annotations

from datetime import datetime

import psycopg
from psycopg.types.json import Jsonb

SCORE_COLS = [
    "ts", "meter_id", "model_version", "actual_kw", "expected_kw", "lower_kw", "upper_kw", "z",
    "cusum_pos", "cusum_neg", "excess_kwh", "alert_type", "weather_stale", "latency_ms",
]  # fmt: skip
ALERT_COLS = [
    "meter_id", "alert_type", "started_at", "ended_at", "peak_z", "excess_kwh", "co2_kg",
    "cost_usd", "observed_kw", "expected_kw", "drivers", "model_version",
]  # fmt: skip


class PostgresSink:
    def __init__(self, url: str):
        self.conn = psycopg.connect(url, autocommit=True)

    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()):
        return self.conn.execute(sql, params).fetchone()

    # -- Sink --------------------------------------------------------------

    def insert_reading(self, meter_id: str, ts: datetime, kw: float) -> None:
        self.conn.execute(
            "INSERT INTO readings (ts, meter_id, kw) VALUES (%s, %s, %s) "
            "ON CONFLICT (ts, meter_id) DO UPDATE SET kw = EXCLUDED.kw",
            (ts, meter_id, kw),
        )

    def insert_weather(self, station_id: str, row: dict) -> None:
        cols = ["ts", *[c for c in row if c != "ts"]]
        self.conn.execute(
            f"INSERT INTO weather_obs (station_id, {', '.join(cols)}) "
            f"VALUES (%s, {', '.join(['%s'] * len(cols))}) ON CONFLICT DO NOTHING",
            (station_id, *[row[c] for c in cols]),
        )

    def insert_score(self, row: dict) -> None:
        self.conn.execute(
            f"INSERT INTO scores ({', '.join(SCORE_COLS)}) "
            f"VALUES ({', '.join(['%s'] * len(SCORE_COLS))}) ON CONFLICT DO NOTHING",
            tuple(row[c] for c in SCORE_COLS),
        )

    def open_alert(self, alert: dict) -> int:
        values = [Jsonb(alert[c]) if c == "drivers" else alert[c] for c in ALERT_COLS]
        row = self.conn.execute(
            f"INSERT INTO alerts ({', '.join(ALERT_COLS)}) "
            f"VALUES ({', '.join(['%s'] * len(ALERT_COLS))}) RETURNING id",
            values,
        ).fetchone()
        return int(row[0])

    def update_alert(self, alert_id: int, fields: dict) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = %s" for k in fields)
        vals = [Jsonb(v) if k == "drivers" else v for k, v in fields.items()]
        self.conn.execute(f"UPDATE alerts SET {sets} WHERE id = %s", (*vals, alert_id))

    # -- ops ---------------------------------------------------------------

    def record_model_version(self, version: str, run_id: str | None, metrics: dict, promoted: bool):
        self.conn.execute(
            "INSERT INTO model_versions (version, run_id, f1, fa_per_week, coverage, promoted) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (version) DO UPDATE SET "
            "promoted = EXCLUDED.promoted, f1 = EXCLUDED.f1, fa_per_week = EXCLUDED.fa_per_week, "
            "coverage = EXCLUDED.coverage",
            (
                version,
                run_id,
                metrics.get("f1"),
                metrics.get("fa_per_week"),
                metrics.get("coverage"),
                promoted,
            ),
        )
