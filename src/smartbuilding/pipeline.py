"""Train / calibrate / gate — shared by scripts/train.py and the nightly retrain job."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbuilding.config import Config, ModelParams, Rules
from smartbuilding.data.dataset import build_dataset
from smartbuilding.data.demand import load_demand, localize
from smartbuilding.data.weather import load_weather
from smartbuilding.features import SeasonalMeanImputer, build_features
from smartbuilding.model import SLOTS_PER_DAY, Calibrator, Detector, QuantileLGBM


def load_dataset(cfg: Config, start: str, end: str) -> pd.DataFrame:
    """ts, kw, weather on the complete 15-min grid for [start, end] (UTC dates)."""
    demand = localize(load_demand(cfg.paths.demand_csv), cfg.site.tz)
    weather = load_weather(
        cfg.paths.weather_parquet,
        cfg.paths.weather_metric_csv,
        cfg.paths.weather_imperial_dir,
        cfg.site.tz,
    )
    return build_dataset(demand, weather, start, f"{end} 23:45", cfg.site.freq)


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def train_detector(
    train: pd.DataFrame,
    calibrate: pd.DataFrame,
    params: ModelParams,
    rules: Rules,
    tz: str,
    model_version: str,
) -> tuple[Detector, dict]:
    """Fit imputer + quantile model on `train`, calibrate the band on `calibrate`."""
    train = train.dropna(subset=["kw"])
    calibrate = calibrate.dropna(subset=["kw"])

    imputer = SeasonalMeanImputer(tz).fit(train)
    model = QuantileLGBM(params).fit(build_features(train, imputer, tz), train["kw"])
    pred_cal = model.predict(build_features(calibrate, imputer, tz))
    # Calibrate on the same level-adjusted residual the scorer uses: the slow EWMA of the kW
    # residual (the building's recent level) is removed before measuring the band width, so a
    # regime change inside the calibration window does not blow the band open.
    resid = calibrate["kw"].to_numpy() - pred_cal["q50"].to_numpy()
    level = (
        pd.Series(resid)
        .ewm(halflife=rules.level_halflife_days * SLOTS_PER_DAY, adjust=False)
        .mean()
        .shift(1)
        .fillna(0.0)
        .to_numpy()
    )
    y_adj = pd.Series(calibrate["kw"].to_numpy() - level, index=pred_cal.index)
    calibrator = Calibrator(params.target_coverage).fit(y_adj, pred_cal)
    lower, upper = calibrator.bands(pred_cal)
    lower, upper = lower + level, upper + level

    y = calibrate["kw"].to_numpy()
    metrics = {
        "n_train": int(len(train)),
        "n_calibrate": int(len(calibrate)),
        "cal_pinball_q05": pinball_loss(y, pred_cal["q05"].to_numpy(), 0.05),
        "cal_pinball_q50": pinball_loss(y, pred_cal["q50"].to_numpy(), 0.50),
        "cal_pinball_q95": pinball_loss(y, pred_cal["q95"].to_numpy(), 0.95),
        "cal_mae": float(np.mean(np.abs(y - pred_cal["q50"].to_numpy()))),
        "cal_raw_coverage": float(((y >= pred_cal["q05"]) & (y <= pred_cal["q95"])).mean()),
        "cal_coverage": float(((y >= lower) & (y <= upper)).mean()),
        "calibration_factor": calibrator.factor,
    }
    return Detector(model, imputer, calibrator, rules, tz, model_version=model_version), metrics


def gate(
    ours: pd.DataFrame,
    baseline: pd.DataFrame,
    max_fa_per_week: float = 3.0,
    coverage: tuple[float, float] = (0.80, 0.97),
    recall_ratio: float = 1.0,
) -> tuple[bool, list[str]]:
    """Promotion gate on the operational metrics.

    Recall per anomaly type must be at least `recall_ratio` × the baseline's, false alarms on
    clean data must stay under budget, and the band must be roughly calibrated. F1/precision are
    deliberately not used: with a handful of injected events per year they only proxy FA count.
    """
    reasons: list[str] = []
    for kind in ours.index.drop("clean", errors="ignore"):
        if ours.loc[kind, "recall"] < recall_ratio * baseline.loc[kind, "recall"]:
            reasons.append(
                f"{kind}: recall {ours.loc[kind, 'recall']:.2f} < "
                f"{recall_ratio:.2f} × baseline {baseline.loc[kind, 'recall']:.2f}"
            )
    fa = ours.loc["clean", "fa_per_week"]
    if fa > max_fa_per_week:
        reasons.append(f"false alarms {fa:.2f}/week > {max_fa_per_week}")
    cov = ours.loc["clean", "coverage"]
    if not coverage[0] <= cov <= coverage[1]:
        reasons.append(f"coverage {cov:.3f} outside {coverage}")
    return not reasons, reasons
