"""Open-Meteo historical weather client (free, no key) — the canonical weather source."""

import argparse
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from smartbuilding.data.weather import WEATHER_COLS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo hourly variable → our column. Units requested are already metric.
_VARS = {
    "temperature_2m": "temp_c",
    "dew_point_2m": "dewpoint_c",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind_kph",
    "wind_gusts_10m": "gust_kph",
    "surface_pressure": "pressure_hpa",
    "precipitation": "precip_mm",
}


class OpenMeteoClient:
    def __init__(self, cache_path: Path | str | None = None, timeout: float = 60.0):
        self.cache_path = Path(cache_path) if cache_path else None
        self.timeout = timeout

    def fetch_archive(self, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
        """Hourly weather for [start, end] as UTC ts + WEATHER_COLS; cached to parquet if set."""
        if self.cache_path and self.cache_path.exists():
            return pd.read_parquet(self.cache_path)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(_VARS),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        resp = httpx.get(ARCHIVE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        df = pd.DataFrame({"ts": pd.to_datetime(hourly["time"], utc=True)})
        for var, col in _VARS.items():
            df[col] = pd.Series(hourly[var], dtype="float64")
        df = df[["ts", *WEATHER_COLS]]
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.cache_path, index=False)
        return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Open-Meteo archive to parquet")
    ap.add_argument("--lat", type=float, default=42.44)
    ap.add_argument("--lon", type=float, default=-76.50)
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--out", default="data/processed/weather_openmeteo.parquet")
    args = ap.parse_args()
    client = OpenMeteoClient(cache_path=args.out)
    df = client.fetch_archive(args.lat, args.lon, args.start, args.end)
    print(f"{len(df)} hourly rows {df['ts'].min()} → {df['ts'].max()} written to {args.out}")
    print(df[WEATHER_COLS].describe().loc[["min", "mean", "max"]].round(1))


if __name__ == "__main__":
    main()
