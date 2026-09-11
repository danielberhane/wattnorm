import json

import numpy as np

from smartbuilding.simulator import replay_plan, run
from tests.conftest import synthetic


def test_replay_plan_emits_weather_hourly_and_demand_per_slot_in_order():
    df = synthetic("2020-01-01", 1, seed=50)
    df.loc[10, "kw"] = np.nan  # a meter gap
    plan = replay_plan(df, "bldg-a", "ithaca", inject_kind=None)
    topics = [t for t, _ in plan]
    assert topics.count("weather/ithaca/obs") == 24
    assert topics.count("building/bldg-a/demand") == 95
    ts = [json.loads(p)["ts"] for t, p in plan if t.endswith("/demand")]
    assert ts == sorted(ts)
    assert topics[0] == "weather/ithaca/obs"  # weather for the hour precedes its demand slots


def test_replay_plan_injects_anomaly_early():
    df = synthetic("2020-01-01", 5, seed=51)
    plan = replay_plan(df, "bldg-a", "ithaca", inject_kind="sustained_offset", seed=1)
    kw = [json.loads(p)["kw"] for t, p in plan if t.endswith("/demand")]
    assert max(kw[: 3 * 96]) > df["kw"].iloc[: 3 * 96].max() * 1.1


def test_run_publishes_everything_without_waiting_at_high_speed():
    df = synthetic("2020-01-01", 1, seed=52)
    plan = replay_plan(df, "bldg-a", "ithaca", None)
    sent = []
    run(plan, lambda t, p: sent.append(t), speed=1e9)
    assert len(sent) == len(plan)
