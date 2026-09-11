"""Evaluation without labels: inject known anomalies into a clean period, score, measure.

Injection happens on the raw kw series *before* scoring, so the detector sees exactly what it
would see in production.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from smartbuilding.config import Rules
from smartbuilding.model import Calibrator, Detector, DetectorState

SLOTS_PER_HOUR = 4
SLOTS_PER_DAY = 96
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY

ANOMALY_TYPES = [
    "spike",
    "sustained_offset",
    "dropout",
    "slow_drift",
    "schedule_shift",
    "stuck_meter",
]


_PLACEMENT_ORDER = [
    "slow_drift", "schedule_shift", "sustained_offset", "stuck_meter", "dropout", "spike",
]  # fmt: skip


# ---------------------------------------------------------------- injection


def _window(rng: np.random.Generator, n: int, length: int, margin: int) -> tuple[int, int]:
    start = int(rng.integers(margin, n - length - margin))
    return start, start + length - 1


def inject(df: pd.DataFrame, kind: str, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Return a copy of df with one anomaly of `kind` and its label {start, end, type}."""
    out = df.copy()
    kw = out["kw"].to_numpy(copy=True)
    n = len(kw)
    margin = SLOTS_PER_DAY

    if kind == "spike":
        s = int(rng.integers(margin, n - margin))
        e = s
        kw[s] *= rng.uniform(1.3, 2.0)
    elif kind == "sustained_offset":
        s, e = _window(rng, n, int(rng.integers(2, 13)) * SLOTS_PER_HOUR, margin)
        kw[s : e + 1] *= 1 + rng.uniform(0.15, 0.40)
    elif kind == "dropout":
        s, e = _window(rng, n, int(rng.integers(1, 7)) * SLOTS_PER_HOUR, margin)
        kw[s : e + 1] *= 1 - rng.uniform(0.30, 0.60)
    elif kind == "slow_drift":
        length = 14 * SLOTS_PER_DAY
        s, e = _window(rng, n, length, margin)
        ramp = np.linspace(0.0, 0.14, length)  # +1 %/day for two weeks
        kw[s : e + 1] *= 1 + ramp
    elif kind == "schedule_shift":
        # the building runs its daytime profile at night: roll one day's readings by 6 h
        day = int(rng.integers(margin // SLOTS_PER_DAY, n // SLOTS_PER_DAY - 1))
        s, e = day * SLOTS_PER_DAY, (day + 1) * SLOTS_PER_DAY - 1
        kw[s : e + 1] = np.roll(kw[s : e + 1], 6 * SLOTS_PER_HOUR)
    elif kind == "stuck_meter":
        s, e = _window(rng, n, int(rng.integers(2, 7)) * SLOTS_PER_HOUR, margin)
        kw[s : e + 1] = kw[s]
    else:
        raise ValueError(f"unknown anomaly type {kind!r}; choose from {ANOMALY_TYPES}")

    out["kw"] = kw
    return out, {"start": s, "end": e, "type": kind}


def inject_many(
    df: pd.DataFrame,
    n_per_type: int,
    rng: np.random.Generator,
    types: Iterable[str] = ANOMALY_TYPES,
    min_gap: int = SLOTS_PER_DAY + 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject n anomalies of each type at non-overlapping positions with ≥ min_gap slots between."""
    out = df
    labels: list[dict] = []
    # place the longest anomalies first so short ones do not fragment the free space
    for kind in sorted(types, key=lambda k: _PLACEMENT_ORDER.index(k)):
        for _ in range(n_per_type):
            for _attempt in range(500):
                candidate, label = inject(out, kind, rng)
                if all(
                    label["start"] > lb["end"] + min_gap or label["end"] < lb["start"] - min_gap
                    for lb in labels
                ):
                    out, labels = candidate, [*labels, label]
                    break
            else:
                raise RuntimeError(f"could not place {kind} without overlap; use a longer period")
    return out, pd.DataFrame(labels, columns=["start", "end", "type"])


# ---------------------------------------------------------------- metrics


def alert_events(alert_type: pd.Series) -> list[tuple[int, int]]:
    """Runs of consecutive non-empty alert_type → [(start_idx, end_idx)] by position."""
    active = alert_type.fillna("").ne("").to_numpy()
    events, start = [], None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(active) - 1))
    return events


def event_metrics(labels: pd.DataFrame, scores: pd.DataFrame, tolerance_slots: int) -> dict:
    """Event-level precision/recall/F1, median time-to-detect, false alarms/week, band coverage."""
    events = alert_events(scores["alert_type"])
    matched_events: set[int] = set()
    ttd: list[float] = []
    tp = 0
    for lab in labels.itertuples(index=False):
        lo, hi = lab.start - tolerance_slots, lab.end + tolerance_slots
        hits = [i for i, (s, e) in enumerate(events) if s <= hi and e >= lo]
        if hits:
            tp += 1
            matched_events.update(hits)
            first = min(events[i][0] for i in hits)
            ttd.append(max(0, first - lab.start) * 60 / SLOTS_PER_HOUR)
    fn = len(labels) - tp
    fp = len(events) - len(matched_events)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weeks = len(scores) / SLOTS_PER_WEEK
    in_band = (scores["actual_kw"] >= scores["lower_kw"]) & (
        scores["actual_kw"] <= scores["upper_kw"]
    )
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ttd_median_min": float(np.median(ttd)) if ttd else float("nan"),
        "fa_per_week": fp / weeks if weeks else float("nan"),
        "coverage": float(in_band[scores["actual_kw"].notna()].mean()),
    }


# ---------------------------------------------------------------- baseline


class BaselineDetector(Detector):
    """Seasonal-naive: expected = same slot last week; band from (weekday, slot) residual spread.

    Uses the same Calibrator and alert rules as the real detector, so the comparison is fair:
    the only difference is where the expectation comes from.
    """

    def __init__(self, profile: pd.Series, spread: pd.Series, calibrator, rules, tz):
        self.profile = profile  # mean kw by (dow, slot) — fallback when no last-week value
        self.spread = spread  # std of kw by (dow, slot)
        self.calibrator = calibrator
        self.rules = rules
        self.tz = tz
        self.model_version = "seasonal-naive"

    @staticmethod
    def _keys(ts: pd.Series, tz: str) -> pd.MultiIndex:
        local = ts.dt.tz_convert(tz)
        slot = local.dt.hour * SLOTS_PER_HOUR + local.dt.minute // 15
        return pd.MultiIndex.from_arrays([local.dt.dayofweek, slot], names=["dow", "slot"])

    @classmethod
    def fit(cls, df: pd.DataFrame, rules: Rules, tz: str, target_coverage: float = 0.9):
        keys = cls._keys(df["ts"], tz)
        grouped = df["kw"].groupby([keys.get_level_values(0), keys.get_level_values(1)])
        profile, spread = grouped.mean(), grouped.std().fillna(df["kw"].std())
        base = cls(profile, spread, Calibrator(target_coverage), rules, tz)
        base.calibrator.fit(df["kw"], base.predict(df))
        return base

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = self._keys(df["ts"], self.tz)
        last_week = df["kw"].shift(SLOTS_PER_WEEK)
        q50 = last_week.fillna(pd.Series(self.profile.reindex(keys).to_numpy(), index=df.index))
        sigma = pd.Series(self.spread.reindex(keys).to_numpy(), index=df.index).fillna(
            float(self.spread.mean())
        )
        return pd.DataFrame({"q05": q50 - 1.645 * sigma, "q50": q50, "q95": q50 + 1.645 * sigma})

    def new_state(self) -> DetectorState:
        return super().new_state()


# ---------------------------------------------------------------- harness


def run_eval(
    detector: Detector,
    clean: pd.DataFrame,
    n_per_type: int = 3,
    seeds: Iterable[int] = range(5),
    tolerance_slots: int = 4,
) -> pd.DataFrame:
    """Per-anomaly-type detection metrics (averaged over seeds) plus a 'clean' row."""
    rows: dict[str, list[dict]] = {t: [] for t in ANOMALY_TYPES}
    for seed in seeds:
        injected, labels = inject_many(clean, n_per_type, np.random.default_rng(seed))
        scores, _ = detector.score(injected)
        for kind in ANOMALY_TYPES:
            rows[kind].append(
                event_metrics(labels[labels["type"] == kind], scores, tolerance_slots)
            )
    clean_scores, _ = detector.score(clean)
    clean_m = event_metrics(pd.DataFrame(columns=["start", "end", "type"]), clean_scores, 0)

    table = pd.DataFrame(
        {kind: pd.DataFrame(rs).mean(numeric_only=True) for kind, rs in rows.items()}
    ).T
    table.loc["clean"] = pd.Series(clean_m)
    cols = [
        "precision",
        "recall",
        "f1",
        "ttd_median_min",
        "fa_per_week",
        "coverage",
        "tp",
        "fn",
        "fp",
    ]
    return table[cols]
