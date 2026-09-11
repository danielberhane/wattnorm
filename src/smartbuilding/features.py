"""Context features: calendar (local time, cyclic) + weather. No demand lags by design —
the model must answer "what should this building draw now", not "what did it draw a minute ago".
"""

import holidays
import numpy as np
import pandas as pd

from smartbuilding.data.weather import WEATHER_COLS

CALENDAR_FEATURES = [
    "sin_tod",
    "cos_tod",
    "sin_dow",
    "cos_dow",
    "sin_doy",
    "cos_doy",
    "is_holiday",
    "is_weekend",
]
WEATHER_FEATURES = [*WEATHER_COLS, "hdd", "cdd"]
FEATURES = [*CALENDAR_FEATURES, *WEATHER_FEATURES]

HEATING_BASE_C = 18.0
COOLING_BASE_C = 22.0


def _cyclic(values: pd.Series, period: float, name: str) -> pd.DataFrame:
    angle = 2 * np.pi * values / period
    return pd.DataFrame({f"sin_{name}": np.sin(angle), f"cos_{name}": np.cos(angle)})


def calendar_features(ts: pd.Series, tz: str) -> pd.DataFrame:
    """Cyclic time-of-day / day-of-week / day-of-year + holiday/weekend flags, in local time."""
    local = ts.dt.tz_convert(tz)
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    years = range(local.dt.year.min(), local.dt.year.max() + 1)
    us_holidays = holidays.US(years=years)
    out = pd.concat(
        [
            _cyclic(minute_of_day, 24 * 60, "tod"),
            _cyclic(local.dt.dayofweek, 7, "dow"),
            _cyclic(local.dt.dayofyear, 365.25, "doy"),
        ],
        axis=1,
    )
    out["is_holiday"] = local.dt.date.map(lambda d: int(d in us_holidays)).astype(int)
    out["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    out.index = ts.index
    return out


def weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Heating / cooling degree terms — the parts of temperature a building responds to."""
    return pd.DataFrame(
        {
            "hdd": (HEATING_BASE_C - df["temp_c"]).clip(lower=0.0),
            "cdd": (df["temp_c"] - COOLING_BASE_C).clip(lower=0.0),
        },
        index=df.index,
    )


class SeasonalMeanImputer:
    """Fill missing weather with the training mean for that (month, hour) in local time.

    Weather is strongly seasonal and diurnal, so this beats a global mean, and fitting on the
    training split only keeps the calibration/test years untouched.
    """

    def __init__(self, tz: str):
        self.tz = tz
        self.means_: pd.DataFrame | None = None
        self.global_: pd.Series | None = None

    def _keys(self, ts: pd.Series) -> pd.DataFrame:
        local = ts.dt.tz_convert(self.tz)
        return pd.DataFrame({"month": local.dt.month, "hour": local.dt.hour}, index=ts.index)

    def fit(self, df: pd.DataFrame) -> "SeasonalMeanImputer":
        keys = self._keys(df["ts"])
        self.means_ = df[WEATHER_COLS].groupby([keys["month"], keys["hour"]]).mean()
        self.global_ = df[WEATHER_COLS].mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.global_ is None:
            raise RuntimeError("SeasonalMeanImputer must be fit before transform")
        keys = self._keys(df["ts"])
        idx = pd.MultiIndex.from_arrays([keys["month"], keys["hour"]])
        fill = self.means_.reindex(idx).set_axis(df.index).fillna(self.global_)
        out = df.copy()
        out[WEATHER_COLS] = out[WEATHER_COLS].fillna(fill)
        return out


def build_features(df: pd.DataFrame, imputer: SeasonalMeanImputer, tz: str) -> pd.DataFrame:
    """ts + WEATHER_COLS frame → model matrix with exactly FEATURES columns, no NaN."""
    filled = imputer.transform(df)
    X = pd.concat(
        [calendar_features(filled["ts"], tz), filled[WEATHER_COLS], weather_features(filled)],
        axis=1,
    )
    return X[FEATURES]
