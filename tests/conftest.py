import numpy as np
import pandas as pd
import pytest

from smartbuilding.config import ModelParams, Rules
from smartbuilding.data.weather import WEATHER_COLS
from smartbuilding.features import SeasonalMeanImputer, build_features
from smartbuilding.model import Calibrator, Detector, QuantileLGBM

TZ = "America/New_York"
PARAMS = ModelParams(
    quantiles=[0.05, 0.5, 0.95],
    n_estimators=60,
    learning_rate=0.1,
    num_leaves=15,
    min_child_samples=20,
    target_coverage=0.9,
)
RULES = Rules(
    spike_z=4.0,
    cusum_k=0.5,
    cusum_h=8.0,
    stuck_slots=8,
    close_after_in_band_slots=8,
    weather_stale_hours=3,
)


def synthetic(start: str, days: int, seed: int = 0) -> pd.DataFrame:
    """15-min frame: demand follows time-of-day + temperature with small noise."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    local = ts.tz_convert(TZ)
    tod = (local.hour * 60 + local.minute) / 1440
    temp = (
        10 + 12 * np.sin(2 * np.pi * (local.dayofyear / 365.25 - 0.3)) + rng.normal(0, 2, len(ts))
    )
    kw = (
        200
        + 40 * np.sin(2 * np.pi * (tod - 0.25))
        + 1.5 * np.abs(temp - 18)
        + rng.normal(0, 3, len(ts))
    )
    df = pd.DataFrame({"ts": ts, "kw": kw, "temp_c": temp})
    for c in WEATHER_COLS:
        if c not in df:
            df[c] = 1.0
    return df


@pytest.fixture(scope="module")
def fitted():
    train = synthetic("2017-01-01", 365, seed=1)
    cal = synthetic("2018-05-01", 30, seed=2)
    imputer = SeasonalMeanImputer(TZ).fit(train)
    model = QuantileLGBM(PARAMS).fit(build_features(train, imputer, TZ), train["kw"])
    calibrator = Calibrator(PARAMS.target_coverage).fit(
        cal["kw"], model.predict(build_features(cal, imputer, TZ))
    )
    return imputer, model, calibrator


@pytest.fixture(scope="module")
def detector(fitted):
    imputer, model, calibrator = fitted
    return Detector(model, imputer, calibrator, RULES, TZ, model_version="test")
