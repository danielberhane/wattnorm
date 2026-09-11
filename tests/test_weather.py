import pandas as pd
import pytest

from smartbuilding.data.weather import (
    WEATHER_COLS,
    align_to_grid,
    load_weather_imperial,
    load_weather_metric,
    to_metric,
)

TZ = "America/New_York"

METRIC_HEADER = (
    "Time,Temperature,Dew Point,Humidity,Wind Speed,Wind Gust,"
    "Pressure,Precipitation,Wind,Condition,Date\n"
)
IMPERIAL_HEADER = "," + METRIC_HEADER


def test_to_metric_converts_imperial_units():
    df = pd.DataFrame(
        {
            "temp_c": [32.0],
            "dewpoint_c": [50.0],
            "rh": [80.0],
            "wind_kph": [10.0],
            "gust_kph": [0.0],
            "pressure_hpa": [29.92],
            "precip_mm": [1.0],
        }
    )
    out = to_metric(df)
    assert out["temp_c"].iloc[0] == pytest.approx(0.0)
    assert out["dewpoint_c"].iloc[0] == pytest.approx(10.0)
    assert out["rh"].iloc[0] == 80.0
    assert out["wind_kph"].iloc[0] == pytest.approx(16.0934, rel=1e-4)
    assert out["pressure_hpa"].iloc[0] == pytest.approx(1013.25, rel=1e-4)
    assert out["precip_mm"].iloc[0] == pytest.approx(25.4)


def test_load_weather_metric_parses_and_nulls_bad_values(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text(
        METRIC_HEADER
        + "7:56 AM,-1,-3,86,9,0,976.19,0,WSW,Cloudy,2016-01-01\n"
        + "8:56 AM,-1,-3,86,11,294,0,0,WSW,Cloudy,2016-01-01\n"
    )
    df = load_weather_metric(p)
    assert list(df.columns) == ["ts_local", *WEATHER_COLS]
    assert df["ts_local"].iloc[0] == pd.Timestamp("2016-01-01 07:56")
    assert df["temp_c"].iloc[0] == -1.0
    assert pd.isna(df["pressure_hpa"].iloc[1])  # 0 hPa is a sensor null
    assert pd.isna(df["gust_kph"].iloc[1])  # 294 kph gust is an outlier


def test_load_weather_metric_handles_thousands_separator(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text(METRIC_HEADER + '7:56 AM,-1,-3,86,9,0,"1,001.9",0,WSW,Cloudy,2016-01-01\n')
    df = load_weather_metric(p)
    assert df["pressure_hpa"].iloc[0] == pytest.approx(1001.9)


def test_load_weather_imperial_concatenates_and_converts(tmp_path):
    (tmp_path / "Year 2015.csv").write_text(
        IMPERIAL_HEADER + "0,1:56 AM,32,5,54,12,0,29.92,0.0,SW,Cloudy,2015-01-01\n"
    )
    (tmp_path / "Year 2016.csv").write_text(
        IMPERIAL_HEADER + "0,1:56 AM,50,5,54,0,0,29.92,0.0,SW,Cloudy,2016-01-01\n"
    )
    df = load_weather_imperial(tmp_path)
    assert len(df) == 2
    assert df["ts_local"].is_monotonic_increasing
    assert df["temp_c"].tolist() == pytest.approx([0.0, 10.0])
    assert df["pressure_hpa"].iloc[0] == pytest.approx(1013.25, rel=1e-4)


def _obs(times: list[str], temps: list[float]) -> pd.DataFrame:
    df = pd.DataFrame({"ts": pd.to_datetime(times, utc=True), "temp_c": temps})
    for c in WEATHER_COLS:
        if c not in df:
            df[c] = 1.0
    return df


def test_align_to_grid_rounds_to_hour_and_forward_fills_15min_slots():
    obs = _obs(["2019-01-01 07:56", "2019-01-01 08:56"], [5.0, 6.0])
    grid = pd.date_range("2019-01-01 08:00", "2019-01-01 09:45", freq="15min", tz="UTC")
    out = align_to_grid(obs, grid)
    assert len(out) == 8
    assert out["temp_c"].tolist() == [5.0] * 4 + [6.0] * 4


def test_align_to_grid_keeps_last_when_two_obs_round_to_same_hour():
    obs = _obs(["2019-01-01 07:30", "2019-01-01 07:56"], [1.0, 2.0])
    grid = pd.date_range("2019-01-01 08:00", "2019-01-01 08:00", freq="15min", tz="UTC")
    out = align_to_grid(obs, grid)
    assert out["temp_c"].tolist() == [2.0]


def test_align_to_grid_stops_forward_fill_after_one_hour():
    obs = _obs(["2019-01-01 08:00"], [5.0])
    grid = pd.date_range("2019-01-01 08:00", "2019-01-01 10:00", freq="15min", tz="UTC")
    out = align_to_grid(obs, grid)
    # 08:00, 08:15, 08:30, 08:45 filled; 09:00 onwards NaN (no observation for that hour)
    assert out["temp_c"].notna().tolist() == [True] * 4 + [False] * 5


def test_load_weather_prefers_parquet_then_metric_then_imperial(tmp_path):
    from smartbuilding.data.weather import load_weather

    metric = tmp_path / "metric.csv"
    metric.write_text(METRIC_HEADER + "7:56 AM,-1,-3,86,9,0,976.19,0,WSW,Cloudy,2016-01-01\n")
    imperial_dir = tmp_path / "imperial"
    imperial_dir.mkdir()
    (imperial_dir / "Year 2015.csv").write_text(
        IMPERIAL_HEADER + "0,1:56 AM,32,5,54,12,0,29.92,0.0,SW,Cloudy,2015-01-01\n"
    )
    parquet = tmp_path / "w.parquet"

    # no parquet → metric csv wins
    df = load_weather(parquet, metric, imperial_dir, tz=TZ)
    assert df["temp_c"].tolist() == [-1.0]
    assert str(df["ts"].dt.tz) == "UTC"

    # no parquet, no metric → imperial (converted)
    df = load_weather(parquet, tmp_path / "missing.csv", imperial_dir, tz=TZ)
    assert df["temp_c"].tolist() == pytest.approx([0.0])

    # parquet present → used as-is
    pd.DataFrame(
        {"ts": pd.to_datetime(["2019-01-01"], utc=True), **{c: [9.0] for c in WEATHER_COLS}}
    ).to_parquet(parquet, index=False)
    df = load_weather(parquet, metric, imperial_dir, tz=TZ)
    assert df["temp_c"].tolist() == [9.0]
