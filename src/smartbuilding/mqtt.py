"""The MQTT contract. A simulator today and a real meter gateway tomorrow publish exactly this.

Topics:  building/{meter_id}/demand      payload DemandMsg
         weather/{station_id}/obs        payload WeatherMsg
Timestamps are ISO-8601 *with* offset; naive times are rejected so nobody guesses a time zone.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from smartbuilding.config import Env
from smartbuilding.data.weather import WEATHER_COLS


def _require_aware_utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timestamp must carry a UTC offset, e.g. 2020-01-01T00:15:00-05:00")
    return v.astimezone(UTC)


class DemandMsg(BaseModel):
    meter_id: str = Field(min_length=1)
    ts: datetime
    kw: float = Field(gt=0)

    _utc = field_validator("ts")(_require_aware_utc)


class WeatherMsg(BaseModel):
    station_id: str = Field(min_length=1)
    ts: datetime
    temp_c: float
    dewpoint_c: float
    rh: float
    wind_kph: float
    gust_kph: float
    pressure_hpa: float
    precip_mm: float

    _utc = field_validator("ts")(_require_aware_utc)

    def as_row(self) -> dict:
        return {"ts": self.ts, **{c: getattr(self, c) for c in WEATHER_COLS}}


def demand_topic(meter_id: str) -> str:
    return f"building/{meter_id}/demand"


def weather_topic(station_id: str) -> str:
    return f"weather/{station_id}/obs"


DEMAND_WILDCARD = "building/+/demand"
WEATHER_WILDCARD = "weather/+/obs"


def parse_message(topic: str, payload: bytes) -> DemandMsg | WeatherMsg | None:
    parts = topic.split("/")
    if len(parts) == 3 and parts[0] == "building" and parts[2] == "demand":
        return DemandMsg.model_validate_json(payload)
    if len(parts) == 3 and parts[0] == "weather" and parts[2] == "obs":
        return WeatherMsg.model_validate_json(payload)
    return None


def make_client(env: Env, client_id: str):
    """A connected-on-demand paho client; TLS is a config switch so IoT Core is a .env change."""
    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    if env.mqtt_tls:
        client.tls_set()
    return client
