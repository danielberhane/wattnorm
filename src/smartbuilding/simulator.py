"""Replay historical demand + weather over MQTT as if a meter were publishing live.

uv run python -m smartbuilding.simulator --start 2020-01-01 --speed 60 --inject sustained_offset
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from smartbuilding.config import Env, load_config
from smartbuilding.data.weather import WEATHER_COLS
from smartbuilding.eval import inject
from smartbuilding.mqtt import DemandMsg, WeatherMsg, demand_topic, make_client, weather_topic
from smartbuilding.pipeline import load_dataset

log = logging.getLogger("simulator")

DEMO_INJECTABLE = ["spike", "sustained_offset", "dropout", "schedule_shift", "stuck_meter"]


def replay_plan(
    data: pd.DataFrame, meter_id: str, station_id: str, inject_kind: str | None, seed: int = 0
) -> list[tuple[str, str]]:
    """Ordered (topic, payload) pairs: a weather message at each hour, a demand message per slot.

    Slots with a meter gap are skipped — that is what a real meter would do too.
    """
    df = data.reset_index(drop=True)
    if inject_kind:
        if inject_kind not in DEMO_INJECTABLE:
            raise ValueError(f"--inject must be one of {DEMO_INJECTABLE} (fits a 3-day head)")
        # place the anomaly a day into the replay so a demo shows it quickly
        rng = np.random.default_rng(seed)
        head = df.iloc[: 3 * 96].copy()
        head, label = inject(head, inject_kind, rng)
        df.loc[: len(head) - 1, "kw"] = head["kw"].to_numpy()
        log.info("injected %s at slots %s–%s", inject_kind, label["start"], label["end"])
    out: list[tuple[str, str]] = []
    for row in df.itertuples(index=False):
        ts = row.ts.to_pydatetime()
        if ts.minute == 0 and all(np.isfinite(getattr(row, c)) for c in WEATHER_COLS):
            wx = WeatherMsg(
                station_id=station_id, ts=ts, **{c: getattr(row, c) for c in WEATHER_COLS}
            )
            out.append((weather_topic(station_id), wx.model_dump_json()))
        if np.isfinite(row.kw):
            out.append(
                (
                    demand_topic(meter_id),
                    DemandMsg(meter_id=meter_id, ts=ts, kw=row.kw).model_dump_json(),
                )
            )
    return out


def run(
    plan: list[tuple[str, str]],
    publish: Callable[[str, str], None],
    speed: float,
    slot_s: float = 900,
) -> None:
    """Publish the plan; each 15-min slot takes slot_s/speed wall seconds."""
    delay = slot_s / speed
    for topic, payload in plan:
        publish(topic, payload)
        if topic.endswith("/demand"):
            time.sleep(delay)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--start", default=None, help="UTC date; default = configured replay start")
    ap.add_argument("--end", default=None)
    ap.add_argument("--speed", type=float, default=60.0, help="60 → one 15-min slot per 15 s")
    ap.add_argument("--inject", default=None, help="anomaly type to inject a day into the replay")
    args = ap.parse_args()

    cfg, env = load_config(args.config), Env()
    start, end = args.start or cfg.splits.replay[0], args.end or cfg.splits.replay[1]
    data = load_dataset(cfg, start, end)
    plan = replay_plan(data, cfg.site.meter_id, cfg.site.station_id, args.inject)
    log.info("replaying %s → %s: %d messages at %sx", start, end, len(plan), args.speed)

    client = make_client(env, client_id=f"simulator-{cfg.site.meter_id}")
    client.connect(env.mqtt_host, env.mqtt_port)
    client.loop_start()
    try:
        run(plan, lambda t, p: client.publish(t, p, qos=1), args.speed)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
