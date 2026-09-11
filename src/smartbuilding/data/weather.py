"""Weather loading, unit normalisation and alignment onto the demand grid.

Everything is metric internally: °C, %, km/h, hPa, mm.
"""

from pathlib import Path

import pandas as pd

WEATHER_COLS = ["temp_c", "dewpoint_c", "rh", "wind_kph", "gust_kph", "pressure_hpa", "precip_mm"]

# Weather Underground export column names → our names
_RAW_TO_COL = {
    "Temperature": "temp_c",
    "Dew Point": "dewpoint_c",
    "Humidity": "rh",
    "Wind Speed": "wind_kph",
    "Wind Gust": "gust_kph",
    "Pressure": "pressure_hpa",
    "Precipitation": "precip_mm",
}

_GUST_OUTLIER_KPH = 200.0  # anything above this is a sensor glitch, not weather


def to_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a frame holding imperial values in WEATHER_COLS to metric."""
    out = df.copy()
    out["temp_c"] = (out["temp_c"] - 32.0) * 5.0 / 9.0
    out["dewpoint_c"] = (out["dewpoint_c"] - 32.0) * 5.0 / 9.0
    out["wind_kph"] = out["wind_kph"] * 1.609344
    out["gust_kph"] = out["gust_kph"] * 1.609344
    out["pressure_hpa"] = out["pressure_hpa"] * 33.8639
    out["precip_mm"] = out["precip_mm"] * 25.4
    return out


def _read_wunderground(path: Path | str) -> pd.DataFrame:
    """Parse one Weather Underground CSV to ts_local + WEATHER_COLS (units as in the file)."""
    raw = pd.read_csv(path, thousands=",")
    df = raw.rename(columns=_RAW_TO_COL)[list(_RAW_TO_COL.values())].astype(float)
    df.insert(0, "ts_local", pd.to_datetime(raw["Date"] + " " + raw["Time"]))
    return df.sort_values("ts_local").reset_index(drop=True)


def _null_bad_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[out["pressure_hpa"] <= 0, "pressure_hpa"] = float("nan")
    out.loc[out["gust_kph"] > _GUST_OUTLIER_KPH, "gust_kph"] = float("nan")
    return out


def load_weather_metric(path: Path | str) -> pd.DataFrame:
    """The metric export (course_project/data/weather_data.csv): 0 hPa and absurd gusts → NaN."""
    return _null_bad_values(_read_wunderground(path))


def load_weather_imperial(directory: Path | str) -> pd.DataFrame:
    """The `Year 20xx.csv` scrapes (°F, mph, inHg, in) concatenated and converted to metric."""
    files = sorted(Path(directory).glob("*.csv"))
    df = pd.concat([_read_wunderground(f) for f in files], ignore_index=True)
    df = to_metric(df).sort_values("ts_local").reset_index(drop=True)
    return _null_bad_values(df)


def align_to_grid(obs: pd.DataFrame, grid: pd.DatetimeIndex, max_ffill: int = 4) -> pd.DataFrame:
    """Round hourly-ish observations to the hour and forward-fill onto a 15-min grid.

    Observations that round to the same hour keep the last one. Forward fill is capped at
    max_ffill slots (one hour), so an hour with no observation stays NaN for the imputer.
    """
    hourly = obs.assign(ts=obs["ts"].dt.round("h")).drop_duplicates("ts", keep="last")
    hourly = hourly.set_index("ts")[WEATHER_COLS].sort_index()
    on_grid = hourly.reindex(grid).ffill(limit=max_ffill - 1)
    return on_grid.rename_axis("ts").reset_index()


def load_weather(
    parquet: Path | str, metric_csv: Path | str, imperial_dir: Path | str, tz: str
) -> pd.DataFrame:
    """Hourly observations as UTC ts + WEATHER_COLS from the best available source.

    Precedence: Open-Meteo parquet (already UTC) → metric CSV → imperial CSVs. The CSV sources
    carry naive local times, so they are localised here (DST edge rows dropped).
    """
    if Path(parquet).exists():
        return pd.read_parquet(parquet)
    if Path(metric_csv).exists():
        df = load_weather_metric(metric_csv)
    else:
        df = load_weather_imperial(imperial_dir)
    ts = df["ts_local"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    out = df.drop(columns="ts_local").assign(ts=ts.dt.tz_convert("UTC")).dropna(subset=["ts"])
    return out[["ts", *WEATHER_COLS]].reset_index(drop=True)
