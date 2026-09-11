import numpy as np
import pandas as pd
import pytest

from smartbuilding.features import FEATURES, build_features
from smartbuilding.model import CUSUM, Calibrator, Detector, top_drivers
from tests.conftest import TZ, synthetic

# ---------- QuantileLGBM ----------


def test_quantiles_are_ordered_and_cover_about_90_percent(fitted):
    imputer, model, _ = fitted
    hold = synthetic("2018-06-01", 30, seed=3)
    pred = model.predict(build_features(hold, imputer, TZ))
    assert list(pred.columns) == ["q05", "q50", "q95"]
    assert (pred["q05"] <= pred["q50"]).all() and (pred["q50"] <= pred["q95"]).all()
    cov = ((hold["kw"] >= pred["q05"]) & (hold["kw"] <= pred["q95"])).mean()
    assert 0.8 < cov < 0.98


# ---------- Calibrator ----------


def test_calibrator_hits_target_coverage_on_its_fit_set(fitted):
    imputer, model, calibrator = fitted
    cal = synthetic("2018-05-01", 30, seed=2)
    pred = model.predict(build_features(cal, imputer, TZ))
    lower, upper = calibrator.bands(pred)
    cov = ((cal["kw"] >= lower) & (cal["kw"] <= upper)).mean()
    assert cov == pytest.approx(0.9, abs=0.02)


def test_calibrator_z_is_roughly_standard_normal_on_fit_set(fitted):
    imputer, model, calibrator = fitted
    cal = synthetic("2018-05-01", 30, seed=2)
    z = calibrator.z(cal["kw"], model.predict(build_features(cal, imputer, TZ)))
    assert abs(z.mean()) < 0.3
    assert 0.7 < z.std() < 1.3


def test_calibrator_round_trips_through_dict(fitted):
    _, _, calibrator = fitted
    again = Calibrator.from_dict(calibrator.to_dict())
    assert again.factor == calibrator.factor


# ---------- CUSUM ----------


def test_cusum_does_not_trip_on_standard_noise():
    rng = np.random.default_rng(0)
    c = CUSUM(k=0.5, h=8.0)
    assert all(c.update(z) == 0 for z in rng.normal(0, 1, 1000))


def test_cusum_trips_quickly_and_in_the_right_direction_after_shift():
    c = CUSUM(k=0.5, h=8.0)
    steps = [c.update(z) for z in [2.0] * 10]
    assert 1 in steps and steps.index(1) <= 7
    c = CUSUM(k=0.5, h=8.0)
    steps = [c.update(z) for z in [-2.0] * 10]
    assert -1 in steps


def test_cusum_resets_after_trip():
    c = CUSUM(k=0.5, h=8.0)
    [c.update(2.0) for _ in range(10)]
    assert c.s_pos < 8.0


# ---------- Detector ----------


def test_detector_scores_have_expected_columns_and_no_alerts_on_clean_data(detector):
    clean = synthetic("2018-07-01", 10, seed=4)
    scores, _ = detector.score(clean)
    for col in [
        "ts",
        "actual_kw",
        "expected_kw",
        "lower_kw",
        "upper_kw",
        "z",
        "cusum_pos",
        "cusum_neg",
        "excess_kwh",
        "alert_type",
        "model_version",
    ]:
        assert col in scores.columns
    assert not scores["alert_type"].str.contains("SUSTAINED|STUCK").any()
    assert (scores["alert_type"] == "").mean() > 0.98  # rare tail spikes are tolerated here
    assert (scores["model_version"] == "test").all()


def test_detector_flags_sustained_high_consumption(detector):
    df = synthetic("2018-07-01", 3, seed=5)
    i0, i1 = 96 + 8, 96 + 32  # day 2, 02:00–08:00 UTC, +30 %
    df.loc[i0:i1, "kw"] *= 1.3
    scores, _ = detector.score(df)
    flagged = scores.loc[i0:i1, "alert_type"].str.contains("SUSTAINED_HIGH")
    assert flagged.any()
    assert scores.loc[i0:i1, "excess_kwh"].sum() > 0


def test_detector_flags_sustained_low_consumption(detector):
    df = synthetic("2018-07-01", 3, seed=6)
    df.loc[100:130, "kw"] *= 0.6
    scores, _ = detector.score(df)
    assert scores.loc[100:130, "alert_type"].str.contains("SUSTAINED_LOW").any()


def test_detector_flags_single_slot_spike(detector):
    df = synthetic("2018-07-01", 2, seed=7)
    df.loc[120, "kw"] *= 1.6
    scores, _ = detector.score(df)
    assert "SPIKE_HIGH" in scores.loc[120, "alert_type"]


def test_detector_flags_stuck_meter(detector):
    df = synthetic("2018-07-01", 2, seed=8)
    df.loc[50:70, "kw"] = 210.0
    scores, _ = detector.score(df)
    assert scores.loc[50:70, "alert_type"].str.contains("STUCK").any()
    assert "STUCK" not in scores.loc[50, "alert_type"]  # needs stuck_slots identical readings


def test_detector_batch_and_stepwise_scoring_agree(detector):
    df = synthetic("2018-07-01", 2, seed=9)
    df.loc[100:120, "kw"] *= 1.3
    batch, _ = detector.score(df)
    rows, state = [], None
    for i in range(len(df)):
        s, state = detector.score(df.iloc[[i]], state)
        rows.append(s)
    step = pd.concat(rows, ignore_index=True)
    pd.testing.assert_frame_equal(batch, step)


def test_detector_alert_closes_after_in_band_run(detector):
    df = synthetic("2018-07-01", 3, seed=10)
    df.loc[100:130, "kw"] *= 1.3
    scores, _ = detector.score(df)
    assert scores.loc[100:130, "alert_type"].str.contains("SUSTAINED_HIGH").any()
    assert not scores.loc[200:, "alert_type"].str.contains("SUSTAINED").any()


def test_detector_round_trips_through_save_and_load(detector, tmp_path):
    df = synthetic("2018-07-01", 2, seed=11)
    detector.save(tmp_path / "artifact")
    again = Detector.load(tmp_path / "artifact")
    a, _ = detector.score(df)
    b, _ = again.score(df)
    pd.testing.assert_frame_equal(a, b)


# ---------- explanation ----------


def test_top_drivers_returns_named_feature_contributions(fitted):
    imputer, model, _ = fitted
    X = build_features(synthetic("2018-07-01", 1, seed=12), imputer, TZ)
    drivers = top_drivers(model, X.iloc[[40]], k=3)
    assert len(drivers) == 3
    assert all(d["feature"] in FEATURES for d in drivers)
    assert all(isinstance(d["contribution_kw"], float) for d in drivers)


# ---------- level adaptation ----------


def test_calibrator_fit_ignores_nan_targets(fitted):
    imputer, model, _ = fitted
    cal = synthetic("2018-05-01", 10, seed=30)
    pred = model.predict(build_features(cal, imputer, TZ))
    y = cal["kw"].copy()
    y.iloc[::7] = np.nan
    c = Calibrator(0.9).fit(y, pred)
    assert np.isfinite(c.factor) and c.factor > 0


def test_detector_adapts_to_a_permanent_level_shift(detector):
    df = synthetic("2018-07-01", 40, seed=31)
    df["kw"] -= 20.0  # a permanent level drop (e.g. a load was removed)
    scores, _ = detector.score(df)
    first_week = scores.iloc[: 96 * 7]
    last_10_days = scores.iloc[-96 * 10 :]
    assert first_week["alert_type"].str.contains("SUSTAINED_LOW").any()
    assert not last_10_days["alert_type"].str.contains("SUSTAINED").any()
    # the displayed expectation follows the building's new level
    ratio = (last_10_days["expected_kw"] / last_10_days["actual_kw"]).mean()
    assert ratio == pytest.approx(1.0, abs=0.03)


def test_detector_still_catches_offset_after_adapting(detector):
    df = synthetic("2018-07-01", 40, seed=32)
    df["kw"] -= 20.0
    i0, i1 = 96 * 35, 96 * 35 + 24
    df.loc[i0:i1, "kw"] *= 1.3
    scores, _ = detector.score(df)
    assert scores.loc[i0:i1, "alert_type"].str.contains("SUSTAINED_HIGH").any()
