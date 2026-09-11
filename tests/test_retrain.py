from datetime import date

import pandas as pd

from smartbuilding.retrain import decide, retrain_windows


def test_retrain_windows_are_contiguous_and_end_at_as_of():
    w = retrain_windows(date(2020, 9, 1), window_months=24)
    assert w["train"] == ("2018-09-01", "2020-02-29")
    assert w["calibrate"] == ("2020-03-01", "2020-05-31")
    assert w["evaluate"] == ("2020-06-01", "2020-08-31")


def _table(f1, fa, cov):
    idx = ["spike", "sustained_offset", "clean"]
    return pd.DataFrame(
        {"recall": [f1, f1, float("nan")], "fa_per_week": [0, 0, fa], "coverage": [0, 0, cov]},
        index=idx,
    )


def test_decide_promotes_when_better_than_baseline_and_not_worse_than_production():
    ok, reasons = decide(
        candidate=_table(0.8, 2.0, 0.9),
        baseline=_table(0.5, 3, 0.85),
        production=_table(0.85, 2.5, 0.9),
    )
    assert ok, reasons


def test_decide_rejects_regression_against_production():
    ok, reasons = decide(
        candidate=_table(0.5, 2.0, 0.9),
        baseline=_table(0.4, 3, 0.85),
        production=_table(0.9, 1.0, 0.9),
    )
    assert not ok
    assert any("production" in r for r in reasons)


def test_decide_without_production_uses_baseline_gate_only():
    ok, _ = decide(candidate=_table(0.8, 2.0, 0.9), baseline=_table(0.5, 3, 0.85), production=None)
    assert ok
