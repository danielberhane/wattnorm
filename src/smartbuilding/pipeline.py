"""Train / calibrate / gate — shared by scripts/train.py and the nightly retrain job."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbuilding.config import Config, ModelParams, Rules
from smartbuilding.data.dataset import build_dataset
from smartbuilding.data.demand import load_demand, localize
from smartbuilding.data.weather import load_weather
from smartbuilding.features import SeasonalMeanImputer, build_features
from smartbuilding.model import Calibrator, Detector, QuantileLGBM


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
    calibrator = Calibrator(params.target_coverage).fit(calibrate["kw"], pred_cal)
    lower, upper = calibrator.bands(pred_cal)

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
    coverage: tuple[float, float] = (0.85, 0.95),
    f1_ratio: float = 1.0,
) -> tuple[bool, list[str]]:
    """Promotion gate: beat the baseline F1 per type, few false alarms, band calibrated."""
    reasons: list[str] = []
    for kind in ours.index.drop("clean", errors="ignore"):
        if ours.loc[kind, "f1"] < f1_ratio * baseline.loc[kind, "f1"]:
            reasons.append(
                f"{kind}: F1 {ours.loc[kind, 'f1']:.2f} < "
                f"{f1_ratio:.2f} × baseline {baseline.loc[kind, 'f1']:.2f}"
            )
    fa = ours.loc["clean", "fa_per_week"]
    if fa > max_fa_per_week:
        reasons.append(f"false alarms {fa:.2f}/week > {max_fa_per_week}")
    cov = ours.loc["clean", "coverage"]
    if not coverage[0] <= cov <= coverage[1]:
        reasons.append(f"coverage {cov:.3f} outside {coverage}")
    return not reasons, reasons
