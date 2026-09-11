from smartbuilding.model import Detector
from smartbuilding.pipeline import gate, pinball_loss, train_detector
from tests.conftest import PARAMS, RULES, TZ, synthetic


def test_train_detector_returns_detector_and_calibration_metrics():
    train = synthetic("2017-01-01", 365, seed=1)
    cal = synthetic("2018-01-01", 60, seed=2)
    det, metrics = train_detector(train, cal, PARAMS, RULES, TZ, model_version="t")
    assert isinstance(det, Detector)
    assert 0.85 < metrics["cal_coverage"] < 0.95
    assert metrics["cal_pinball_q50"] > 0
    assert metrics["cal_mae"] > 0
    assert metrics["n_train"] == len(train)


def test_pinball_loss_is_asymmetric():
    import numpy as np

    y = np.array([10.0, 10.0])
    assert pinball_loss(y, np.array([8.0, 8.0]), 0.9) > pinball_loss(y, np.array([12.0, 12.0]), 0.9)


def test_gate_passes_when_better_than_baseline_and_within_limits():
    import pandas as pd

    idx = ["spike", "sustained_offset", "clean"]
    ours = pd.DataFrame(
        {"f1": [0.9, 0.9, float("nan")], "fa_per_week": [0, 0, 2.0], "coverage": [0, 0, 0.9]},
        index=idx,
    )
    base = pd.DataFrame({"f1": [0.5, 0.5, float("nan")]}, index=idx)
    ok, reasons = gate(ours, base, max_fa_per_week=3, coverage=(0.85, 0.95))
    assert ok and reasons == []


def test_gate_fails_and_explains():
    import pandas as pd

    idx = ["spike", "clean"]
    ours = pd.DataFrame(
        {"f1": [0.4, float("nan")], "fa_per_week": [0, 5.0], "coverage": [0, 0.80]}, index=idx
    )
    base = pd.DataFrame({"f1": [0.5, float("nan")]}, index=idx)
    ok, reasons = gate(ours, base, max_fa_per_week=3, coverage=(0.85, 0.95))
    assert not ok
    assert any("spike" in r for r in reasons)
    assert any("false alarms" in r for r in reasons)
    assert any("coverage" in r for r in reasons)
