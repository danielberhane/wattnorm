import numpy as np
import pandas as pd
import pytest

from smartbuilding.data.weather import WEATHER_COLS
from smartbuilding.features import (
    FEATURES,
    SeasonalMeanImputer,
    build_features,
    calendar_features,
    weather_features,
)

TZ = "America/New_York"


def _ts(*local_times):
    return pd.to_datetime(list(local_times)).tz_localize(TZ).tz_convert("UTC")


def test_calendar_features_are_unit_circle_encodings():
    ts = pd.Series(_ts("2019-07-04 12:00", "2019-12-31 23:45"))
    f = calendar_features(ts, TZ)
    for name in ["tod", "dow", "doy"]:
        assert np.allclose(f[f"sin_{name}"] ** 2 + f[f"cos_{name}"] ** 2, 1.0)


def test_calendar_features_wrap_midnight():
    ts = pd.Series(_ts("2019-01-01 23:45", "2019-01-02 00:00"))
    f = calendar_features(ts, TZ)
    d = np.hypot(f["sin_tod"].diff().iloc[1], f["cos_tod"].diff().iloc[1])
    assert d < 0.1  # 23:45 and 00:00 are neighbours, not opposite ends


def test_calendar_features_flag_holiday_and_weekend_in_local_time():
    ts = pd.Series(_ts("2019-07-04 10:00", "2019-07-06 10:00", "2019-07-08 10:00"))
    f = calendar_features(ts, TZ)
    assert f["is_holiday"].tolist() == [1, 0, 0]
    assert f["is_weekend"].tolist() == [0, 1, 0]


def test_calendar_features_use_local_not_utc_day():
    # 2019-07-04 23:00 local is 2019-07-05 03:00 UTC — must still count as the holiday.
    f = calendar_features(pd.Series(_ts("2019-07-04 23:00")), TZ)
    assert f["is_holiday"].iloc[0] == 1


def test_weather_features_add_heating_and_cooling_degrees():
    df = pd.DataFrame({"temp_c": [10.0, 20.0, 30.0]})
    f = weather_features(df)
    assert f["hdd"].tolist() == [8.0, 0.0, 0.0]
    assert f["cdd"].tolist() == [0.0, 0.0, 8.0]


def _weather_frame(ts, temp):
    df = pd.DataFrame({"ts": ts, "temp_c": temp})
    for c in WEATHER_COLS:
        if c not in df:
            df[c] = 1.0
    return df


def test_seasonal_mean_imputer_fills_from_training_month_hour_means():
    train = _weather_frame(
        _ts("2018-01-05 08:00", "2018-01-12 08:00", "2018-01-05 20:00"), [-2.0, -4.0, 5.0]
    )
    imp = SeasonalMeanImputer(TZ).fit(train)
    cal = _weather_frame(_ts("2019-01-20 08:00", "2019-01-20 20:00"), [np.nan, np.nan])
    out = imp.transform(cal)
    assert out["temp_c"].tolist() == pytest.approx([-3.0, 5.0])


def test_seasonal_mean_imputer_falls_back_to_global_mean_for_unseen_cell():
    train = _weather_frame(_ts("2018-01-05 08:00", "2018-01-05 20:00"), [-2.0, 6.0])
    imp = SeasonalMeanImputer(TZ).fit(train)
    out = imp.transform(_weather_frame(_ts("2019-06-01 12:00"), [np.nan]))
    assert out["temp_c"].iloc[0] == pytest.approx(2.0)


def test_seasonal_mean_imputer_leaves_observed_values_untouched():
    train = _weather_frame(_ts("2018-01-05 08:00"), [-2.0])
    imp = SeasonalMeanImputer(TZ).fit(train)
    out = imp.transform(_weather_frame(_ts("2019-01-05 08:00"), [7.0]))
    assert out["temp_c"].iloc[0] == 7.0


def test_build_features_returns_exactly_the_feature_columns():
    ts = _ts("2019-01-01 08:00", "2019-01-01 08:15")
    df = _weather_frame(ts, [1.0, np.nan]).assign(kw=[100.0, 101.0])
    imp = SeasonalMeanImputer(TZ).fit(_weather_frame(_ts("2018-01-01 08:00"), [3.0]))
    X = build_features(df, imp, TZ)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()
    assert len(X) == 2
