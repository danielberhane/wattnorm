"""Temporal train / calibrate / test splitting. Never shuffle a time series."""

import pandas as pd


def temporal_split(
    df: pd.DataFrame, bounds: dict[str, tuple[str, str]], ts_col: str = "ts"
) -> dict[str, pd.DataFrame]:
    """Slice df into named, non-overlapping, chronologically ordered date ranges.

    Bounds are inclusive calendar dates: ("2016-01-01", "2017-12-31") keeps everything up to
    2017-12-31 23:59:59. Ranges must be given in chronological order and must not overlap.
    """
    tz = df[ts_col].dt.tz
    ranges = {
        name: (pd.Timestamp(s, tz=tz), pd.Timestamp(e, tz=tz) + pd.Timedelta(days=1))
        for name, (s, e) in bounds.items()
    }
    prev_name, prev_start, prev_end = None, None, None
    for name, (start, end) in ranges.items():
        if prev_start is not None:
            if start < prev_start:
                raise ValueError(f"split bounds out of order: {prev_name} must precede {name}")
            if start < prev_end:
                raise ValueError(f"split bounds overlap: {prev_name} and {name}")
        prev_name, prev_start, prev_end = name, start, end
    return {
        name: df[(df[ts_col] >= s) & (df[ts_col] < e)].reset_index(drop=True)
        for name, (s, e) in ranges.items()
    }
