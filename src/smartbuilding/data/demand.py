"""Demand (electric load) loading, time-zone handling and grid completion."""

from pathlib import Path

import pandas as pd


def load_demand(path: Path | str) -> pd.DataFrame:
    """Read the raw demand CSV (columns Timestamp, Demand) into ts_local / kw.

    Duplicate timestamps keep the last reading. Non-positive demand is rejected:
    a meter never reports zero or negative load, so that is a data error, not an anomaly.
    """
    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {"ts_local": pd.to_datetime(raw["Timestamp"]), "kw": raw["Demand"].astype(float)}
    )
    df = df.drop_duplicates("ts_local", keep="last").sort_values("ts_local")
    if (df["kw"] <= 0).any():
        bad = int((df["kw"] <= 0).sum())
        raise ValueError(f"{bad} rows have non-positive kw; demand must be > 0")
    return df.reset_index(drop=True)


def localize(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Turn naive local ts_local into tz-aware UTC ts.

    Fall-back hour (ambiguous) readings are dropped because the source does not say which
    occurrence they are; spring-forward (non-existent) readings are dropped for the same reason.
    """
    ts = df["ts_local"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    out = df.drop(columns="ts_local").assign(ts=ts.dt.tz_convert("UTC"))
    out = out.dropna(subset=["ts"])
    return out[["ts", *[c for c in out.columns if c != "ts"]]].reset_index(drop=True)


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
