from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from smartbuilding.mqtt import DemandMsg, WeatherMsg
from smartbuilding.scorer import MemorySink, ScorerCore
from tests.conftest import synthetic


def _weather(ts, temp=20.0):
    return WeatherMsg(
        station_id="ithaca",
        ts=ts,
        temp_c=temp,
        dewpoint_c=10,
        rh=50,
        wind_kph=5,
        gust_kph=8,
        pressure_hpa=1000,
        precip_mm=0,
    )


@pytest.fixture
def core(detector):
    return ScorerCore(
        detector,
        sink=MemorySink(),
        station_id="ithaca",
        weather_stale_hours=3,
        emission_factor=0.25,
        tariff=0.14,
    )


def _feed(core, df, kw_scale=None):
    """Replay a synthetic frame through the core; returns the sink."""
    for i, row in df.iterrows():
        ts = row["ts"].to_pydatetime()
        if ts.minute == 0:
            core.handle_weather(_weather(ts, row["temp_c"]))
        kw = row["kw"] * (kw_scale(i) if kw_scale else 1.0)
        core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts, kw=kw))
    return core.sink


def test_each_reading_produces_a_reading_and_a_score_row(core):
    df = synthetic("2018-07-01", 1, seed=40)
    sink = _feed(core, df)
    assert len(sink.readings) == 96
    assert len(sink.scores) == 96
    assert sink.scores[0]["meter_id"] == "bldg-a"
    assert sink.scores[0]["weather_stale"] is False
    assert sink.scores[0]["latency_ms"] >= 0
    assert len(sink.weather) == 24


def test_stale_or_missing_weather_is_flagged_not_fatal(core):
    ts = datetime(2018, 7, 1, tzinfo=UTC)
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts, kw=200.0))
    assert core.sink.scores[0]["weather_stale"] is True
    core.handle_weather(_weather(ts))
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts + timedelta(hours=2), kw=200.0))
    assert core.sink.scores[1]["weather_stale"] is False
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts + timedelta(hours=4), kw=200.0))
    assert core.sink.scores[2]["weather_stale"] is True  # 4 h old > 3 h


def test_sustained_offset_opens_one_alert_then_closes_it(core):
    df = synthetic("2018-07-01", 3, seed=41)
    sink = _feed(core, df, kw_scale=lambda i: 1.3 if 100 <= i <= 130 else 1.0)
    opened = [a for a in sink.alerts if a["alert_type"] == "SUSTAINED_HIGH"]
    assert len(opened) == 1
    a = opened[0]
    assert a["ended_at"] is not None and a["ended_at"] > a["started_at"]
    assert a["peak_z"] > 0
    assert a["excess_kwh"] > 0
    assert a["co2_kg"] == pytest.approx(a["excess_kwh"] * 0.25)
    assert a["cost_usd"] == pytest.approx(a["excess_kwh"] * 0.14)
    assert len(a["drivers"]) == 3 and "feature" in a["drivers"][0]
    assert a["model_version"] == "test"


def test_out_of_order_and_duplicate_readings_are_ignored(core):
    ts = datetime(2018, 7, 1, tzinfo=UTC)
    core.handle_weather(_weather(ts))
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts, kw=200.0))
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts, kw=201.0))  # duplicate
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts - timedelta(minutes=15), kw=1.0))
    assert len(core.sink.scores) == 1


def test_health_reports_model_version_and_last_score(core):
    ts = datetime(2018, 7, 1, tzinfo=UTC)
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts, kw=200.0))
    h = core.health()
    assert h["model_version"] == "test"
    assert h["meters"]["bldg-a"]["last_ts"] == ts.isoformat()
    assert h["meters"]["bldg-a"]["open_alerts"] == []
    # the running configuration is visible, not only the version string
    assert h["rules"]["spike_z"] == core.detector.rules.spike_z
    assert h["calibration_factor"] == pytest.approx(core.detector.calibrator.factor)


def test_swap_detector_keeps_state_but_changes_version(core, detector):
    ts = datetime(2018, 7, 1, tzinfo=UTC)
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts, kw=200.0))
    import copy

    new = copy.copy(detector)
    new.model_version = "v2"
    core.swap_detector(new)
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts + timedelta(minutes=15), kw=200.0))
    assert core.sink.scores[-1]["model_version"] == "v2"
    assert core.health()["meters"]["bldg-a"]["n_scored"] == 2


def test_jump_far_into_the_past_resets_meter_state_instead_of_ignoring(core):
    ts = datetime(2020, 6, 1, tzinfo=UTC)
    core.handle_weather(_weather(ts))
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts, kw=200.0))
    # a replay restarted from an earlier date: > 1 day back → treat as a new session
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=ts - timedelta(days=30), kw=200.0))
    assert len(core.sink.scores) == 2
    assert core.health()["meters"]["bldg-a"]["n_scored"] == 1  # state was reset


def test_swapping_models_does_not_carry_the_old_level_bias_into_alerts(core, fitted):
    """Reviewer finding: after a promotion the old q50-relative bias made quiet days alert."""

    from smartbuilding.features import SeasonalMeanImputer, build_features
    from smartbuilding.model import Calibrator, Detector, QuantileLGBM
    from tests.conftest import PARAMS, RULES, TZ

    # old model adapts to a building that dropped 20 kW
    low = synthetic("2018-07-01", 20, seed=70)
    low["kw"] -= 20.0
    _feed(core, low)
    # a new model trained on the shifted building takes over
    train = synthetic("2017-01-01", 365, seed=71)
    train["kw"] -= 20.0
    imp = SeasonalMeanImputer(TZ).fit(train)
    model = QuantileLGBM(PARAMS).fit(build_features(train, imp, TZ), train["kw"])
    cal = synthetic("2018-05-01", 30, seed=72)
    cal["kw"] -= 20.0
    calib = Calibrator(0.9).fit(cal["kw"], model.predict(build_features(cal, imp, TZ)))
    core.swap_detector(Detector(model, imp, calib, RULES, TZ, model_version="new"))
    before = len(core.sink.alerts)
    quiet = synthetic("2018-07-21", 5, seed=73)
    quiet["kw"] -= 20.0
    _feed(core, quiet)
    new_alerts = [a for a in core.sink.alerts[before:] if "SUSTAINED" in a["alert_type"]]
    assert new_alerts == []
    assert abs(pd.Series([s["z"] for s in core.sink.scores[-96:]]).mean()) < 1.0
