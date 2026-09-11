"""The detector: quantile model → calibrated band → signed residual → alert rules.

Detector.score() is the single interface both the offline evaluation and the live scorer use,
so what we measure in eval is exactly what runs in production.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from smartbuilding.config import ModelParams, Rules
from smartbuilding.features import FEATURES, SeasonalMeanImputer, build_features

SLOT_HOURS = 0.25
SLOTS_PER_DAY = 96
Z_PER_HALF_WIDTH = 1.645  # a 90 % band spans ±1.645σ under normality
LEVEL_CLIP_SIGMA = 2.0  # robust level tracking: clip residuals fed to the EWMA


# ---------------------------------------------------------------- quantile model


class QuantileLGBM:
    """One LightGBM booster per quantile; predictions are re-sorted so q05 ≤ q50 ≤ q95."""

    def __init__(self, params: ModelParams):
        self.params = params
        self.boosters: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileLGBM:
        for q in self.params.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.params.n_estimators,
                learning_rate=self.params.learning_rate,
                num_leaves=self.params.num_leaves,
                min_child_samples=self.params.min_child_samples,
                verbose=-1,
            )
            self.boosters[q] = m.fit(X[FEATURES], y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = {self._name(q): b.predict(X[FEATURES]) for q, b in self.boosters.items()}
        pred = pd.DataFrame(cols, index=X.index)
        pred[:] = np.sort(pred.to_numpy(), axis=1)  # quantile crossing fix
        return pred

    @property
    def median(self) -> lgb.LGBMRegressor:
        return self.boosters[0.5]

    @staticmethod
    def _name(q: float) -> str:
        return f"q{int(round(q * 100)):02d}"


# ---------------------------------------------------------------- calibration


class Calibrator:
    """One scalar that widens/narrows the raw quantile band until it hits target coverage.

    z is the residual expressed in band-half-widths, scaled so a 90 % band ≈ ±1.645σ.
    """

    def __init__(self, target_coverage: float, factor: float = 1.0, min_half_width: float = 1e-3):
        self.target_coverage = target_coverage
        self.factor = factor
        self.min_half_width = min_half_width

    def _half_width(self, pred: pd.DataFrame) -> pd.Series:
        return ((pred["q95"] - pred["q05"]) / 2).clip(lower=self.min_half_width)

    def fit(self, y: pd.Series, pred: pd.DataFrame) -> Calibrator:
        ratio = (y.to_numpy() - pred["q50"].to_numpy()) / self._half_width(pred).to_numpy()
        self.factor = float(np.nanquantile(np.abs(ratio), self.target_coverage))
        return self

    def bands(self, pred: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        hw = self._half_width(pred) * self.factor
        return pred["q50"] - hw, pred["q50"] + hw

    def sigma(self, pred: pd.DataFrame) -> pd.Series:
        """Residual scale implied by the calibrated band (kW per unit z)."""
        return self._half_width(pred) * self.factor / Z_PER_HALF_WIDTH

    def z(self, y: pd.Series, pred: pd.DataFrame) -> pd.Series:
        return (y.to_numpy() - pred["q50"]) / self.sigma(pred)

    def to_dict(self) -> dict:
        return {
            "target_coverage": self.target_coverage,
            "factor": self.factor,
            "min_half_width": self.min_half_width,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Calibrator:
        return cls(**d)


# ---------------------------------------------------------------- CUSUM


@dataclass
class CUSUM:
    """Two-sided CUSUM on z. Returns +1 / -1 when the positive / negative sum trips, else 0."""

    k: float
    h: float
    s_pos: float = 0.0
    s_neg: float = 0.0

    def update(self, z: float) -> int:
        self.s_pos = max(0.0, self.s_pos + z - self.k)
        self.s_neg = max(0.0, self.s_neg - z - self.k)
        if self.s_pos > self.h:
            self.s_pos = 0.0
            return 1
        if self.s_neg > self.h:
            self.s_neg = 0.0
            return -1
        return 0


# ---------------------------------------------------------------- detector state


@dataclass
class DetectorState:
    cusum: CUSUM
    last_kw: float | None = None
    stuck_run: int = 0
    # slow EWMA of the raw kW residual: the building's current level relative to the model
    level_bias_kw: float = 0.0
    # alert type → consecutive in-band slots seen since it last fired
    open_alerts: dict[str, int] = field(default_factory=dict)


SCORE_COLUMNS = [
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
]


class Detector:
    def __init__(
        self,
        model: QuantileLGBM,
        imputer: SeasonalMeanImputer,
        calibrator: Calibrator,
        rules: Rules,
        tz: str,
        model_version: str = "dev",
    ):
        self.model = model
        self.imputer = imputer
        self.calibrator = calibrator
        self.rules = rules
        self.tz = tz
        self.model_version = model_version

    def new_state(self) -> DetectorState:
        return DetectorState(cusum=CUSUM(k=self.rules.cusum_k, h=self.rules.cusum_h))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(build_features(df, self.imputer, self.tz))

    def score(
        self, df: pd.DataFrame, state: DetectorState | None = None
    ) -> tuple[pd.DataFrame, DetectorState]:
        """Score rows of (ts, kw, weather) in order; state carries CUSUM/alerts across calls."""
        state = state or self.new_state()
        pred = self.predict(df)
        sigma = self.calibrator.sigma(pred).to_numpy()
        half = self.calibrator._half_width(pred).to_numpy() * self.calibrator.factor
        q50 = pred["q50"].to_numpy()
        kw = df["kw"].to_numpy(dtype=float)
        alpha = 1 - 0.5 ** (1 / (self.rules.level_halflife_days * SLOTS_PER_DAY))

        alerts, cpos, cneg = [], [], []
        expected = np.empty(len(df))
        z_adj = np.empty(len(df))
        for i, kw_i in enumerate(kw):
            expected[i] = q50[i] + state.level_bias_kw
            z_adj[i] = (kw_i - expected[i]) / sigma[i]
            fired = self._apply_rules(state, kw_i, z_adj[i])
            in_band = bool(np.isfinite(kw_i) and abs(kw_i - expected[i]) <= half[i])
            alerts.append(self._update_open_alerts(state, fired, in_band))
            cpos.append(state.cusum.s_pos)
            cneg.append(state.cusum.s_neg)
            if np.isfinite(kw_i):
                # robust update: an anomaly may nudge the level by at most 2σ per slot,
                # so it does not become "normal" and cause a rebound alert when it ends
                resid = np.clip(
                    kw_i - expected[i], -LEVEL_CLIP_SIGMA * sigma[i], LEVEL_CLIP_SIGMA * sigma[i]
                )
                state.level_bias_kw += alpha * resid

        lower, upper = expected - half, expected + half
        excess = np.nan_to_num(np.clip(df["kw"].to_numpy() - upper, 0.0, None) * SLOT_HOURS)

        out = pd.DataFrame(
            {
                "ts": df["ts"].to_numpy(),
                "actual_kw": df["kw"].to_numpy(),
                "expected_kw": expected,
                "lower_kw": lower,
                "upper_kw": upper,
                "z": z_adj,
                "cusum_pos": cpos,
                "cusum_neg": cneg,
                "excess_kwh": excess,
                "alert_type": alerts,
                "model_version": self.model_version,
            }
        )
        return out[SCORE_COLUMNS], state

    # -- rules -------------------------------------------------------------

    def _apply_rules(self, state: DetectorState, kw: float, z: float) -> set[str]:
        fired: set[str] = set()
        if not np.isfinite(kw):  # meter gap: nothing to judge, keep state as is
            return fired
        if abs(z) > self.rules.spike_z:
            fired.add("SPIKE_HIGH" if z > 0 else "SPIKE_LOW")
        direction = state.cusum.update(float(z))
        if direction > 0:
            fired.add("SUSTAINED_HIGH")
        elif direction < 0:
            fired.add("SUSTAINED_LOW")
        state.stuck_run = state.stuck_run + 1 if kw == state.last_kw else 1
        state.last_kw = kw
        if state.stuck_run >= self.rules.stuck_slots:
            fired.add("STUCK")
        return fired

    def _update_open_alerts(self, state: DetectorState, fired: set[str], in_band: bool) -> str:
        for t in fired:
            state.open_alerts[t] = 0
        for t in list(state.open_alerts):
            if t in fired:
                continue
            if t == "STUCK" or in_band:
                state.open_alerts[t] += 1
            if state.open_alerts[t] >= self.rules.close_after_in_band_slots:
                del state.open_alerts[t]
        return "|".join(sorted(state.open_alerts))

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for q, booster in self.model.boosters.items():
            booster.booster_.save_model(d / f"model_{QuantileLGBM._name(q)}.txt")
        with open(d / "imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)
        meta = {
            "model_version": self.model_version,
            "tz": self.tz,
            "features": FEATURES,
            "params": self.model.params.model_dump(),
            "rules": self.rules.model_dump(),
            "calibrator": self.calibrator.to_dict(),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path | str) -> Detector:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        params = ModelParams.model_validate(meta["params"])
        model = QuantileLGBM(params)
        for q in params.quantiles:
            booster = lgb.Booster(model_file=str(d / f"model_{QuantileLGBM._name(q)}.txt"))
            model.boosters[q] = _BoosterAdapter(booster)
        with open(d / "imputer.pkl", "rb") as f:
            imputer = pickle.load(f)
        return cls(
            model,
            imputer,
            Calibrator.from_dict(meta["calibrator"]),
            Rules.model_validate(meta["rules"]),
            meta["tz"],
            model_version=meta["model_version"],
        )


class _BoosterAdapter:
    """Gives a raw lgb.Booster the tiny bit of the sklearn interface Detector uses."""

    def __init__(self, booster: lgb.Booster):
        self.booster_ = booster

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster_.predict(X)


# ---------------------------------------------------------------- explanation


def top_drivers(model: QuantileLGBM, X_row: pd.DataFrame, k: int = 3) -> list[dict]:
    """Largest |SHAP| contributions (kW) to the median prediction for one feature row."""
    import shap  # heavy import, only needed when explaining

    explainer = shap.TreeExplainer(model.median.booster_)
    values = np.asarray(explainer.shap_values(X_row[FEATURES]))[0]
    order = np.argsort(-np.abs(values))[:k]
    return [{"feature": FEATURES[i], "contribution_kw": float(values[i])} for i in order]
