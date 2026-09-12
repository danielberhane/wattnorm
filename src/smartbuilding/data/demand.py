"""Demand (electric load) loading, time-zone handling and grid completion."""

from pathlib import Path

import pandas as pd


def load_demand(path: Path | str, tz: str = "UTC") -> pd.DataFrame:
    """Read the raw demand CSV (columns Timestamp, Demand) into ts (UTC) / kw.

    `tz` is the clock the export was written in. The EMCS portal exports in UTC, so the default
    keeps every reading; a genuinely local export (`America/New_York`) drops the fall-back hour
    (the file cannot say which occurrence a reading is) and the non-existent spring-forward hour
    rather than guess.

    Duplicate timestamps keep the last reading. Non-positive demand is rejected:
    a meter never reports zero or negative load, so that is a data error, not an anomaly.
    """
    raw = pd.read_csv(path)
    ts = pd.to_datetime(raw["Timestamp"]).dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    df = pd.DataFrame({"ts": ts.dt.tz_convert("UTC"), "kw": raw["Demand"].astype(float)})
    df = df.dropna(subset=["ts"]).drop_duplicates("ts", keep="last").sort_values("ts")
    if (df["kw"] <= 0).any():
        bad = int((df["kw"] <= 0).sum())
        raise ValueError(f"{bad} rows have non-positive kw; demand must be > 0")
    return df.reset_index(drop=True)


def complete_grid(df: pd.DataFrame, start: str, end: str, freq: str = "15min") -> pd.DataFrame:
    """Left-join df onto a complete UTC grid from start to end; missing slots become NaN."""
    grid = pd.DataFrame({"ts": pd.date_range(start, end, freq=freq, tz="UTC")})
    return grid.merge(df, on="ts", how="left")


def gap_report(df: pd.DataFrame) -> pd.DataFrame:
    """Describe runs of NaN kw as (gap_start, gap_end, n_slots)."""
    missing = df["kw"].isna()
    run_id = (missing != missing.shift()).cumsum()
    gaps = (
        df[missing]
        .groupby(run_id[missing])
        .agg(gap_start=("ts", "first"), gap_end=("ts", "last"), n_slots=("ts", "size"))
        .reset_index(drop=True)
    )
    return gaps[["gap_start", "gap_end", "n_slots"]]
