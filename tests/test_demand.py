from pathlib import Path

import pandas as pd
import pytest

from smartbuilding.data.demand import complete_grid, gap_report, load_demand, localize

TZ = "America/New_York"


def write_csv(tmp_path: Path, rows: list[tuple[str, float]]) -> Path:
    p = tmp_path / "demand.csv"
    pd.DataFrame(rows, columns=["Timestamp", "Demand"]).to_csv(p, index=False)
    return p


def test_load_demand_returns_ts_local_and_kw(tmp_path):
    p = write_csv(tmp_path, [("2019-01-01 05:00:00", 210.9), ("2019-01-01 05:15:00", 228.2)])
    df = load_demand(p)
    assert list(df.columns) == ["ts_local", "kw"]
    assert pd.api.types.is_datetime64_any_dtype(df["ts_local"])
    assert df["kw"].tolist() == [210.9, 228.2]


def test_load_demand_drops_duplicate_timestamps_keeping_last(tmp_path):
    p = write_csv(tmp_path, [("2019-01-01 05:00:00", 1.0), ("2019-01-01 05:00:00", 2.0)])
    df = load_demand(p)
    assert len(df) == 1
    assert df["kw"].iloc[0] == 2.0


def test_load_demand_rejects_non_positive_demand(tmp_path):
    p = write_csv(tmp_path, [("2019-01-01 05:00:00", 0.0)])
    with pytest.raises(ValueError, match="kw"):
        load_demand(p)


def test_localize_converts_to_utc():
    df = pd.DataFrame({"ts_local": pd.to_datetime(["2019-07-01 12:00:00"]), "kw": [1.0]})
    out = localize(df, TZ)
    assert str(out["ts"].dt.tz) == "UTC"
    assert out["ts"].iloc[0] == pd.Timestamp("2019-07-01 16:00:00", tz="UTC")  # EDT = UTC-4
    assert "ts_local" not in out.columns


def test_localize_drops_ambiguous_fall_back_hour():
    # 2019-11-03 01:30 local occurs twice (EDT and EST); we cannot tell which → drop it.
    df = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(["2019-11-03 00:45:00", "2019-11-03 01:30:00"]),
            "kw": [1.0, 2.0],
        }
    )
    out = localize(df, TZ)
    assert out["kw"].tolist() == [1.0]


def test_localize_drops_nonexistent_spring_forward_hour():
    # 2019-03-10 02:30 local does not exist (clocks jump 02:00 -> 03:00); drop it.
    df = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(["2019-03-10 01:45:00", "2019-03-10 02:30:00"]),
            "kw": [1.0, 2.0],
        }
    )
    out = localize(df, TZ)
    assert out["kw"].tolist() == [1.0]


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
