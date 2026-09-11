from datetime import date

import httpx
import pandas as pd
import pytest
import respx

from smartbuilding.data.openmeteo import OpenMeteoClient
from smartbuilding.data.weather import WEATHER_COLS

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

PAYLOAD = {
    "hourly": {
        "time": ["2019-01-01T00:00", "2019-01-01T01:00"],
        "temperature_2m": [-1.0, -1.5],
        "dew_point_2m": [-3.0, -3.5],
        "relative_humidity_2m": [86, 87],
        "wind_speed_10m": [9.0, 10.0],
        "wind_gusts_10m": [15.0, 16.0],
        "surface_pressure": [976.2, 976.0],
        "precipitation": [0.0, 0.1],
    }
}


@respx.mock
def test_fetch_archive_returns_utc_ts_and_weather_cols():
    route = respx.get(ARCHIVE).mock(return_value=httpx.Response(200, json=PAYLOAD))
    client = OpenMeteoClient()
    df = client.fetch_archive(42.44, -76.5, date(2019, 1, 1), date(2019, 1, 1))
    assert route.called
    params = route.calls.last.request.url.params
    assert params["timezone"] == "UTC"
    assert params["start_date"] == "2019-01-01"
    assert list(df.columns) == ["ts", *WEATHER_COLS]
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["temp_c"].tolist() == [-1.0, -1.5]
    assert df["rh"].tolist() == [86.0, 87.0]


@respx.mock
def test_fetch_archive_uses_parquet_cache_on_second_call(tmp_path):
    route = respx.get(ARCHIVE).mock(return_value=httpx.Response(200, json=PAYLOAD))
    cache = tmp_path / "w.parquet"
    client = OpenMeteoClient(cache_path=cache)
    first = client.fetch_archive(42.44, -76.5, date(2019, 1, 1), date(2019, 1, 1))
    second = client.fetch_archive(42.44, -76.5, date(2019, 1, 1), date(2019, 1, 1))
    assert route.call_count == 1
    assert cache.exists()
    pd.testing.assert_frame_equal(first, second)


@respx.mock
def test_fetch_archive_raises_on_http_error():
    respx.get(ARCHIVE).mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        OpenMeteoClient().fetch_archive(42.44, -76.5, date(2019, 1, 1), date(2019, 1, 1))
