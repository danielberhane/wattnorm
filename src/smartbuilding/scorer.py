"""Live scoring: one reading in → one score row out, alerts opened/closed on transitions.

ScorerCore is transport- and storage-agnostic (tests drive it with MemorySink); the MQTT
consumer and FastAPI app at the bottom are thin adapters.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from smartbuilding.data.weather import WEATHER_COLS
from smartbuilding.features import build_features
from smartbuilding.model import Detector, DetectorState, top_drivers
from smartbuilding.mqtt import DemandMsg, WeatherMsg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- storage interface


class Sink(Protocol):
    def insert_reading(self, meter_id: str, ts: datetime, kw: float) -> None: ...
    def insert_weather(self, station_id: str, row: dict) -> None: ...
    def insert_score(self, row: dict) -> None: ...
    def open_alert(self, alert: dict) -> int: ...
    def update_alert(self, alert_id: int, fields: dict) -> None: ...


class MemorySink:
    """In-memory Sink for tests and dry runs."""

    def __init__(self) -> None:
        self.readings: list[dict] = []
        self.weather: list[dict] = []
        self.scores: list[dict] = []
        self.alerts: list[dict] = []

    def insert_reading(self, meter_id, ts, kw):
        self.readings.append({"meter_id": meter_id, "ts": ts, "kw": kw})

    def insert_weather(self, station_id, row):
        self.weather.append({"station_id": station_id, **row})

    def insert_score(self, row):
        self.scores.append(row)

    def open_alert(self, alert):
        alert = {**alert, "id": len(self.alerts) + 1}
        self.alerts.append(alert)
        return alert["id"]

    def update_alert(self, alert_id, fields):
        self.alerts[alert_id - 1].update(fields)


# ---------------------------------------------------------------- per-meter state


@dataclass
class OpenAlert:
    id: int
    started_at: datetime
    peak_z: float = 0.0
    excess_kwh: float = 0.0


@dataclass
class MeterState:
    detector_state: DetectorState
    last_ts: datetime | None = None
    n_scored: int = 0
    open: dict[str, OpenAlert] = field(default_factory=dict)


# ---------------------------------------------------------------- core


class ScorerCore:
    def __init__(
        self,
        detector: Detector,
        sink: Sink,
        station_id: str,
        weather_stale_hours: float,
        emission_factor: float,
        tariff: float,
    ):
        self.detector = detector
        self.sink = sink
        self.station_id = station_id
        self.stale_after = timedelta(hours=weather_stale_hours)
        self.emission_factor = emission_factor
        self.tariff = tariff
        self.meters: dict[str, MeterState] = {}
        self.latest_weather: dict[str, WeatherMsg] = {}
        self._lock = threading.Lock()

    # -- inbound -----------------------------------------------------------

    def handle_weather(self, msg: WeatherMsg) -> None:
        with self._lock:
            self.latest_weather[msg.station_id] = msg
            self.sink.insert_weather(msg.station_id, msg.as_row())

    def handle_demand(self, msg: DemandMsg) -> None:
        t0 = time.perf_counter()
        with self._lock:
            state = self.meters.setdefault(
                msg.meter_id, MeterState(detector_state=self.detector.new_state())
            )
            if state.last_ts is not None and msg.ts <= state.last_ts:
                log.warning("ignoring out-of-order reading %s @ %s", msg.meter_id, msg.ts)
                return
            wx, stale = self._weather_for(msg.ts)
            row = pd.DataFrame([{"ts": msg.ts, "kw": msg.kw, **wx}])
            row["ts"] = pd.to_datetime(row["ts"], utc=True)
            scores, state.detector_state = self.detector.score(row, state.detector_state)
            s = scores.iloc[0]

            self.sink.insert_reading(msg.meter_id, msg.ts, msg.kw)
            self.sink.insert_score(
                {
                    "ts": msg.ts,
                    "meter_id": msg.meter_id,
                    "model_version": self.detector.model_version,
                    "actual_kw": float(s["actual_kw"]),
                    "expected_kw": float(s["expected_kw"]),
                    "lower_kw": float(s["lower_kw"]),
                    "upper_kw": float(s["upper_kw"]),
                    "z": float(s["z"]),
                    "cusum_pos": float(s["cusum_pos"]),
                    "cusum_neg": float(s["cusum_neg"]),
                    "excess_kwh": float(s["excess_kwh"]),
                    "alert_type": s["alert_type"],
                    "weather_stale": stale,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                }
            )
            self._transition_alerts(msg, state, s, row)
            state.last_ts = msg.ts
            state.n_scored += 1

    # -- helpers -----------------------------------------------------------

    def _weather_for(self, ts: datetime) -> tuple[dict, bool]:
        wx = self.latest_weather.get(self.station_id)
        if wx is None or abs(ts - wx.ts) > self.stale_after:
            return dict.fromkeys(WEATHER_COLS, np.nan), True  # imputer fills from seasonal means
        return {c: getattr(wx, c) for c in WEATHER_COLS}, False

    def _transition_alerts(self, msg: DemandMsg, state: MeterState, s, row: pd.DataFrame) -> None:
        active = set(s["alert_type"].split("|")) - {""}
        for kind in active - set(state.open):
            drivers = (
                top_drivers(
                    self.detector.model,
                    build_features(row, self.detector.imputer, self.detector.tz),
                )
                if hasattr(self.detector.model, "median")
                else []
            )
            alert_id = self.sink.open_alert(
                {
                    "meter_id": msg.meter_id,
                    "alert_type": kind,
                    "started_at": msg.ts,
                    "ended_at": None,
                    "peak_z": float(s["z"]),
                    "excess_kwh": 0.0,
                    "co2_kg": 0.0,
                    "cost_usd": 0.0,
                    "observed_kw": float(s["actual_kw"]),
                    "expected_kw": float(s["expected_kw"]),
                    "drivers": drivers,
                    "model_version": self.detector.model_version,
                }
            )
            state.open[kind] = OpenAlert(id=alert_id, started_at=msg.ts)
        for kind in list(state.open):
            oa = state.open[kind]
            if kind in active:
                oa.peak_z = max(oa.peak_z, abs(float(s["z"])), key=abs)
                oa.excess_kwh += float(s["excess_kwh"])
                self.sink.update_alert(oa.id, self._sustainability(oa))
            else:
                self.sink.update_alert(oa.id, {"ended_at": msg.ts, **self._sustainability(oa)})
                del state.open[kind]

    def _sustainability(self, oa: OpenAlert) -> dict:
        return {
            "peak_z": oa.peak_z,
            "excess_kwh": oa.excess_kwh,
            "co2_kg": oa.excess_kwh * self.emission_factor,
            "cost_usd": oa.excess_kwh * self.tariff,
        }

    # -- operations --------------------------------------------------------

    def swap_detector(self, detector: Detector) -> None:
        """Hot-swap the model; per-meter CUSUM/level state is kept."""
        with self._lock:
            self.detector = detector

    def health(self) -> dict:
        with self._lock:
            return {
                "model_version": self.detector.model_version,
                "stations": {k: v.ts.isoformat() for k, v in self.latest_weather.items()},
                "meters": {
                    m: {
                        "last_ts": st.last_ts.isoformat() if st.last_ts else None,
                        "n_scored": st.n_scored,
                        "open_alerts": sorted(st.open),
                    }
                    for m, st in self.meters.items()
                },
            }
