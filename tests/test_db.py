"""Integration tests: need `docker compose up -d timescaledb` (DB_URL from .env or default)."""

from datetime import UTC, datetime

import pytest

from smartbuilding.db import PostgresSink

pytestmark = pytest.mark.integration


@pytest.fixture
def sink():
    import os

    url = os.environ.get("DB_URL", "postgresql://postgres:postgres@localhost:5433/smartbuilding")
    try:
        s = PostgresSink(url)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"database not reachable: {e}")
    s.execute("DELETE FROM alerts WHERE meter_id = 'test-meter'")
    s.execute("DELETE FROM scores WHERE meter_id = 'test-meter'")
    s.execute("DELETE FROM readings WHERE meter_id = 'test-meter'")
    yield s
    s.close()


def test_reading_score_and_alert_round_trip(sink):
    ts = datetime(2020, 1, 1, 5, 15, tzinfo=UTC)
    sink.insert_reading("test-meter", ts, 231.4)
    sink.insert_reading("test-meter", ts, 232.0)  # duplicate ts → upsert, no error
    sink.insert_score(
        {
            "ts": ts,
            "meter_id": "test-meter",
            "model_version": "t",
            "actual_kw": 232.0,
            "expected_kw": 220.0,
            "lower_kw": 210.0,
            "upper_kw": 230.0,
            "z": 2.1,
            "cusum_pos": 1.0,
            "cusum_neg": 0.0,
            "excess_kwh": 0.5,
            "alert_type": "",
            "weather_stale": False,
            "latency_ms": 3.2,
        }
    )
    alert_id = sink.open_alert(
        {
            "meter_id": "test-meter",
            "alert_type": "SUSTAINED_HIGH",
            "started_at": ts,
            "ended_at": None,
            "peak_z": 2.1,
            "excess_kwh": 0.0,
            "co2_kg": 0.0,
            "cost_usd": 0.0,
            "observed_kw": 232.0,
            "expected_kw": 220.0,
            "drivers": [{"feature": "sin_tod", "contribution_kw": 12.0}],
            "model_version": "t",
        }
    )
    sink.update_alert(
        alert_id, {"ended_at": ts, "excess_kwh": 1.5, "co2_kg": 0.375, "cost_usd": 0.21}
    )
    row = sink.fetchone("SELECT kw FROM readings WHERE meter_id='test-meter' AND ts=%s", (ts,))
    assert row[0] == 232.0
    row = sink.fetchone(
        "SELECT excess_kwh, drivers->0->>'feature' FROM alerts WHERE id=%s", (alert_id,)
    )
    assert row == (1.5, "sin_tod")


def test_update_alert_survives_a_dropped_connection(sink):
    ts = datetime(2020, 1, 2, 5, 15, tzinfo=UTC)
    alert_id = sink.open_alert(
        {
            "meter_id": "test-meter",
            "alert_type": "SUSTAINED_LOW",
            "started_at": ts,
            "ended_at": None,
            "peak_z": -2.5,
            "excess_kwh": 0.0,
            "co2_kg": 0.0,
            "cost_usd": 0.0,
            "observed_kw": 180.0,
            "expected_kw": 220.0,
            "drivers": {},
            "model_version": "t",
        }
    )
    sink.conn.close()  # simulate a database restart between two statements
    sink.update_alert(alert_id, {"peak_z": -3.0})
    row = sink.fetchone("SELECT peak_z FROM alerts WHERE id = %s", (alert_id,))
    assert row[0] == -3.0
