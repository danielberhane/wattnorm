"""Assemble the modelling frame: complete 15-min UTC grid of demand joined with weather."""

import pandas as pd

from smartbuilding.data.demand import complete_grid
from smartbuilding.data.weather import align_to_grid


def build_dataset(
    demand: pd.DataFrame, weather: pd.DataFrame, start, end, freq: str = "15min"
) -> pd.DataFrame:
    """ts (UTC), kw (NaN where the meter has a gap), WEATHER_COLS (NaN where no obs that hour)."""
    grid = complete_grid(demand[["ts", "kw"]], start, end, freq)
    wx = align_to_grid(weather, pd.DatetimeIndex(grid["ts"]))
    return grid.merge(wx, on="ts", how="left")
