import pandas as pd

from smartbuilding.data.dataset import build_dataset
from smartbuilding.data.weather import WEATHER_COLS


def test_build_dataset_joins_demand_grid_with_aligned_weather():
    demand = pd.DataFrame(
        {"ts": pd.to_datetime(["2019-01-01 08:00", "2019-01-01 08:30"], utc=True), "kw": [1.0, 3.0]}
    )
    weather = pd.DataFrame({"ts": pd.to_datetime(["2019-01-01 08:00"], utc=True)})
    for c in WEATHER_COLS:
        weather[c] = 5.0
    out = build_dataset(demand, weather, "2019-01-01 08:00", "2019-01-01 08:45")
    assert list(out.columns) == ["ts", "kw", *WEATHER_COLS]
    assert len(out) == 4
    assert out["kw"].isna().tolist() == [False, True, False, True]
    assert out["temp_c"].tolist() == [5.0] * 4
