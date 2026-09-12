from pathlib import Path

import pandas as pd
import pytest

from smartbuilding.data.demand import complete_grid, gap_report, load_demand

TZ = "America/New_York"


def write_csv(tmp_path: Path, rows: list[tuple[str, float]]) -> Path:
    p = tmp_path / "demand.csv"
    pd.DataFrame(rows, columns=["Timestamp", "Demand"]).to_csv(p, index=False)
    return p


def test_load_demand_returns_utc_ts_and_kw(tmp_path):
    p = write_csv(tmp_path, [("2019-01-01 05:00:00", 210.9), ("2019-01-01 05:15:00", 228.2)])
    df = load_demand(p)
    assert list(df.columns) == ["ts", "kw"]
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["ts"].iloc[0] == pd.Timestamp("2019-01-01 05:00:00", tz="UTC")
    assert df["kw"].tolist() == [210.9, 228.2]


def test_load_demand_keeps_dst_hours_when_export_is_utc(tmp_path):
    # The EMCS export clock is UTC: 02:xx on the US spring-forward day and a single 01:xx on the
    # fall-back day are ordinary readings and must survive.
    p = write_csv(
        tmp_path,
        [("2019-03-10 02:15:00", 1.0), ("2019-11-03 01:15:00", 2.0), ("2019-07-01 12:00:00", 3.0)],
    )
    df = load_demand(p)  # default tz="UTC"
    assert df["kw"].tolist() == [1.0, 3.0, 2.0]
    assert df["ts"].iloc[0] == pd.Timestamp("2019-03-10 02:15:00", tz="UTC")


def test_load_demand_local_export_drops_ambiguous_and_nonexistent(tmp_path):
    # A genuinely local export cannot say which fall-back hour a reading belongs to, and has no
    # spring-forward 02:xx hour at all: both are dropped rather than guessed.
    p = write_csv(
        tmp_path,
        [("2019-03-10 02:15:00", 1.0), ("2019-11-03 01:15:00", 2.0), ("2019-07-01 12:00:00", 3.0)],
    )
    df = load_demand(p, tz=TZ)
    assert df["kw"].tolist() == [3.0]
    assert df["ts"].iloc[0] == pd.Timestamp("2019-07-01 16:00:00", tz="UTC")  # EDT = UTC-4


def test_load_demand_drops_duplicate_timestamps_keeping_last(tmp_path):
    p = write_csv(tmp_path, [("2019-01-01 05:00:00", 1.0), ("2019-01-01 05:00:00", 2.0)])
    df = load_demand(p)
    assert len(df) == 1
    assert df["kw"].iloc[0] == 2.0


def test_load_demand_rejects_non_positive_demand(tmp_path):
    p = write_csv(tmp_path, [("2019-01-01 05:00:00", 0.0)])
    with pytest.raises(ValueError, match="kw"):
        load_demand(p)


def test_complete_grid_fills_missing_slots_with_nan():
    ts = pd.to_datetime(["2019-01-01 00:00", "2019-01-01 00:30"], utc=True)
    df = pd.DataFrame({"ts": ts, "kw": [1.0, 3.0]})
    out = complete_grid(df, "2019-01-01 00:00", "2019-01-01 00:45")
    assert len(out) == 4
    assert out["ts"].is_monotonic_increasing
    assert out["kw"].isna().tolist() == [False, True, False, True]


def test_complete_grid_is_continuous_across_dst_in_utc():
    df = pd.DataFrame({"ts": pd.to_datetime(["2019-03-10 00:00"], utc=True), "kw": [1.0]})
    out = complete_grid(df, "2019-03-10 00:00", "2019-03-11 00:00")
    assert len(out) == 24 * 4 + 1
    assert (out["ts"].diff().dropna() == pd.Timedelta("15min")).all()


def test_gap_report_lists_runs_of_missing_slots():
    ts = pd.date_range("2019-01-01", periods=8, freq="15min", tz="UTC")
    kw = [1.0, None, None, 1.0, 1.0, None, 1.0, 1.0]
    rep = gap_report(pd.DataFrame({"ts": ts, "kw": kw}))
    assert list(rep.columns) == ["gap_start", "gap_end", "n_slots"]
    assert rep["n_slots"].tolist() == [2, 1]
    assert rep["gap_start"].iloc[0] == ts[1]
    assert rep["gap_end"].iloc[0] == ts[2]
