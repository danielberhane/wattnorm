import pytest
from pydantic import ValidationError

from smartbuilding.mqtt import DemandMsg, WeatherMsg, demand_topic, parse_message, weather_topic


def test_demand_message_parses_iso_timestamp_with_offset_to_utc():
    msg = DemandMsg.model_validate_json(
        '{"meter_id": "bldg-a", "ts": "2020-01-01T00:15:00-05:00", "kw": 231.4}'
    )
    assert msg.meter_id == "bldg-a"
    assert msg.kw == 231.4
    assert msg.ts.isoformat() == "2020-01-01T05:15:00+00:00"


def test_demand_message_rejects_non_positive_kw_and_naive_timestamp():
    with pytest.raises(ValidationError):
        DemandMsg(meter_id="m", ts="2020-01-01T00:15:00-05:00", kw=0)
    with pytest.raises(ValidationError):
        DemandMsg(meter_id="m", ts="2020-01-01T00:15:00", kw=5)


def test_weather_message_carries_metric_columns():
    msg = WeatherMsg(
        station_id="ithaca",
        ts="2020-01-01T05:00:00+00:00",
        temp_c=-1.0,
        dewpoint_c=-3.0,
        rh=86,
        wind_kph=9,
        gust_kph=15,
        pressure_hpa=976.2,
        precip_mm=0.0,
    )
    assert msg.as_row()["temp_c"] == -1.0
    assert set(msg.as_row()) >= {"ts", "temp_c", "pressure_hpa", "precip_mm"}


def test_topics_and_dispatch():
    assert demand_topic("bldg-a") == "building/bldg-a/demand"
    assert weather_topic("ithaca") == "weather/ithaca/obs"
    d = parse_message(
        "building/bldg-a/demand", b'{"meter_id":"bldg-a","ts":"2020-01-01T05:15:00+00:00","kw":1}'
    )
    assert isinstance(d, DemandMsg)
    w = parse_message(
        "weather/ithaca/obs",
        b'{"station_id":"ithaca","ts":"2020-01-01T05:00:00+00:00","temp_c":1,"dewpoint_c":0,'
        b'"rh":50,"wind_kph":1,"gust_kph":1,"pressure_hpa":1000,"precip_mm":0}',
    )
    assert isinstance(w, WeatherMsg)
    assert parse_message("something/else", b"{}") is None
