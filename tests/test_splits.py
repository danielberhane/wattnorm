import pandas as pd
import pytest

from smartbuilding.data.splits import temporal_split

BOUNDS = {
    "train": ("2016-01-01", "2017-12-31"),
    "calibrate": ("2018-01-01", "2018-12-31"),
    "test": ("2019-01-01", "2019-12-31"),
}


def _df(start, end):
    ts = pd.date_range(start, end, freq="1D", tz="UTC")
    return pd.DataFrame({"ts": ts, "kw": 1.0})


def test_temporal_split_covers_each_range_inclusively():
    parts = temporal_split(_df("2015-06-01", "2020-06-01"), BOUNDS)
    assert set(parts) == {"train", "calibrate", "test"}
    assert parts["train"]["ts"].min() == pd.Timestamp("2016-01-01", tz="UTC")
    assert parts["train"]["ts"].max() == pd.Timestamp("2017-12-31", tz="UTC")
    assert parts["test"]["ts"].min() == pd.Timestamp("2019-01-01", tz="UTC")
    assert parts["test"]["ts"].max() == pd.Timestamp("2019-12-31", tz="UTC")


def test_temporal_split_end_bound_includes_whole_last_day():
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2017-12-31 23:45", "2018-01-01 00:00"], utc=True), "kw": 1.0}
    )
    parts = temporal_split(df, BOUNDS)
    assert len(parts["train"]) == 1
    assert len(parts["calibrate"]) == 1


def test_temporal_split_rejects_overlapping_bounds():
    bad = {"train": ("2016-01-01", "2018-06-30"), "calibrate": ("2018-01-01", "2018-12-31")}
    with pytest.raises(ValueError, match="overlap"):
        temporal_split(_df("2016-01-01", "2019-01-01"), bad)


def test_temporal_split_rejects_out_of_order_bounds():
    bad = {"test": ("2019-01-01", "2019-12-31"), "train": ("2016-01-01", "2017-12-31")}
    with pytest.raises(ValueError, match="order"):
        temporal_split(_df("2016-01-01", "2020-01-01"), bad)
