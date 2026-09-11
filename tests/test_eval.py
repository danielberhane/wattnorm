import numpy as np
import pandas as pd
import pytest

from smartbuilding.eval import (
    ANOMALY_TYPES,
    BaselineDetector,
    event_metrics,
    inject,
    inject_many,
    run_eval,
)
from tests.conftest import RULES, TZ, synthetic

# ---------- injection ----------


@pytest.mark.parametrize("kind", ANOMALY_TYPES)
def test_inject_changes_only_the_labelled_window(kind):
    df = synthetic("2019-03-01", 30, seed=1)
    out, label = inject(df, kind, np.random.default_rng(0))
    assert label["type"] == kind
    s, e = label["start"], label["end"]
    assert 0 <= s <= e < len(df)
    changed = out["kw"].to_numpy() != df["kw"].to_numpy()
    assert changed[s : e + 1].any()
    assert not changed[:s].any() and not changed[e + 1 :].any()
    assert out.drop(columns="kw").equals(df.drop(columns="kw"))


def test_inject_spike_is_a_single_slot():
    _, label = inject(synthetic("2019-03-01", 10), "spike", np.random.default_rng(1))
    assert label["start"] == label["end"]


def test_inject_stuck_meter_holds_a_constant_value():
    out, label = inject(synthetic("2019-03-01", 10), "stuck_meter", np.random.default_rng(2))
    window = out["kw"].iloc[label["start"] : label["end"] + 1]
    assert window.nunique() == 1


def test_inject_slow_drift_ramps_monotonically_upward():
    df = synthetic("2019-03-01", 30, seed=3)
    out, label = inject(df, "slow_drift", np.random.default_rng(3))
    ratio = (out["kw"] / df["kw"]).iloc[label["start"] : label["end"] + 1]
    assert ratio.iloc[0] < ratio.iloc[-1]
    assert (ratio.diff().dropna() >= -1e-9).all()


def test_inject_many_produces_non_overlapping_labels_with_gaps():
    df = synthetic("2019-01-01", 120, seed=4)
    out, labels = inject_many(df, n_per_type=2, rng=np.random.default_rng(4))
    assert len(labels) == 2 * len(ANOMALY_TYPES)
    labels = labels.sort_values("start")
    gaps = labels["start"].to_numpy()[1:] - labels["end"].to_numpy()[:-1]
    assert (gaps > 96).all()  # at least a day between events so alerts can close
    assert len(out) == len(df)


# ---------- metrics ----------


def _scores_with_alerts(n, windows):
    alert = np.array([""] * n, dtype=object)
    for s, e, t in windows:
        alert[s : e + 1] = t
    return pd.DataFrame(
        {
            "ts": pd.date_range("2019-01-01", periods=n, freq="15min", tz="UTC"),
            "actual_kw": 1.0,
            "lower_kw": 0.0,
            "upper_kw": 2.0,
            "alert_type": alert,
        }
    )


def test_event_metrics_counts_hits_misses_and_false_alarms():
    labels = pd.DataFrame(
        {"start": [10, 200], "end": [20, 210], "type": ["sustained_offset", "dropout"]}
    )
    scores = _scores_with_alerts(1000, [(12, 25, "SUSTAINED_HIGH"), (500, 503, "SPIKE_HIGH")])
    m = event_metrics(labels, scores, tolerance_slots=4)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["ttd_median_min"] == pytest.approx(30.0)  # detected 2 slots after onset


def test_event_metrics_counts_early_detection_within_tolerance_as_hit():
    labels = pd.DataFrame({"start": [100], "end": [110], "type": ["spike"]})
    scores = _scores_with_alerts(300, [(98, 99, "SPIKE_HIGH")])
    m = event_metrics(labels, scores, tolerance_slots=4)
    assert m["tp"] == 1 and m["fp"] == 0


def test_event_metrics_reports_false_alarms_per_week_and_coverage():
    scores = _scores_with_alerts(96 * 14, [(5, 6, "SPIKE_HIGH"), (900, 905, "SPIKE_LOW")])
    m = event_metrics(pd.DataFrame(columns=["start", "end", "type"]), scores, 4)
    assert m["fa_per_week"] == pytest.approx(1.0)
    assert m["coverage"] == pytest.approx(1.0)


# ---------- baseline ----------


def test_baseline_detector_scores_and_catches_large_offset(fitted):
    imputer, _, _ = fitted
    fit = synthetic("2018-01-01", 60, seed=5)
    base = BaselineDetector.fit(fit, RULES, TZ)
    df = synthetic("2018-07-01", 14, seed=6)
    df.loc[96 * 8 + 40 : 96 * 8 + 80, "kw"] *= 1.4
    scores, _ = base.score(df)
    assert "expected_kw" in scores and "alert_type" in scores
    assert scores.loc[96 * 8 + 40 : 96 * 8 + 80, "alert_type"].str.contains("HIGH").any()


# ---------- run_eval ----------


def test_run_eval_returns_one_row_per_type_plus_clean(detector):
    clean = synthetic("2019-01-01", 90, seed=7)
    table = run_eval(detector, clean, n_per_type=1, seeds=[0, 1])
    assert set(table.index) == {*ANOMALY_TYPES, "clean"}
    for col in ["precision", "recall", "f1", "ttd_median_min", "fa_per_week", "coverage"]:
        assert col in table.columns
    assert table.loc["sustained_offset", "recall"] > 0.5
    assert table.loc["clean", "coverage"] > 0.8
