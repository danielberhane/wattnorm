"""Nightly retrain on a trailing window with a promotion gate.

    make retrain AS_OF=2020-09-01      (host cron: 0 2 * * * cd <repo> && make retrain)

Train on the oldest part of the window, calibrate on the next three months, evaluate (injected
anomalies + baseline + the current production model) on the most recent three months. Promote
only if the candidate beats the baseline and is not worse than what is already in production.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, date, datetime

import httpx
import mlflow
import pandas as pd

from smartbuilding.config import Env, load_config
from smartbuilding.data.splits import temporal_split
from smartbuilding.eval import BaselineDetector, run_eval
from smartbuilding.pipeline import gate, load_dataset, train_detector
from smartbuilding.registry import MODEL_NAME, load_detector, log_and_register

log = logging.getLogger("retrain")


def _month_start(d: date, months_back: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 - months_back, 12)
    return date(y, m + 1, 1)


def retrain_windows(as_of: date, window_months: int = 24) -> dict[str, tuple[str, str]]:
    """train = [as_of−window, as_of−6mo) · calibrate = 3 months · evaluate = last 3 months."""
    as_of = date(as_of.year, as_of.month, 1)
    eval_start = _month_start(as_of, 3)
    cal_start = _month_start(as_of, 6)
    train_start = _month_start(as_of, window_months)
    day_before = lambda d: (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # noqa: E731
    return {
        "train": (train_start.isoformat(), day_before(cal_start)),
        "calibrate": (cal_start.isoformat(), day_before(eval_start)),
        "evaluate": (eval_start.isoformat(), day_before(as_of)),
    }


def decide(
    candidate: pd.DataFrame, baseline: pd.DataFrame, production: pd.DataFrame | None
) -> tuple[bool, list[str]]:
    ok, reasons = gate(candidate, baseline)
    if production is not None:
        _, reasons2 = gate(candidate, production, recall_ratio=0.9)
        reasons += [f"vs production: {r}" for r in reasons2 if "recall" in r]
        if candidate.loc["clean", "fa_per_week"] > production.loc["clean", "fa_per_week"] + 1.0:
            reasons.append(
                f"vs production: false alarms {candidate.loc['clean', 'fa_per_week']:.2f}/week "
                f"> {production.loc['clean', 'fa_per_week']:.2f} + 1"
            )
        ok = ok and not reasons
    return ok, reasons


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--as-of", type=date.fromisoformat, default=datetime.now(UTC).date())
    ap.add_argument("--window-months", type=int, default=24)
    ap.add_argument("--scorer-url", default=os.environ.get("SCORER_URL", "http://scorer:8000"))
    args = ap.parse_args()

    cfg, env = load_config(args.config), Env()
    mlflow.set_tracking_uri(env.mlflow_tracking_uri)
    mlflow.set_experiment("smartbuilding")
    windows = retrain_windows(args.as_of, args.window_months)
    log.info("windows: %s", windows)

    data = load_dataset(cfg, windows["train"][0], windows["evaluate"][1])
    parts = temporal_split(data, windows)
    version = f"retrain-{args.as_of.isoformat()}"
    candidate, metrics = train_detector(
        parts["train"], parts["calibrate"], cfg.model, cfg.rules, cfg.site.tz, version
    )
    # a 3-month window fits 2 of each anomaly type (slow_drift alone is 14 days)
    cand_table = run_eval(candidate, parts["evaluate"], n_per_type=2)
    base = BaselineDetector.fit(parts["calibrate"].dropna(subset=["kw"]), cfg.rules, cfg.site.tz)
    base_table = run_eval(base, parts["evaluate"], n_per_type=2)
    prod_table = None
    try:
        prod_table = run_eval(load_detector(env.model_uri), parts["evaluate"], n_per_type=2)
    except Exception as e:  # noqa: BLE001 — first run has no production model yet
        log.warning("no production model to compare against (%s)", e)

    ok, reasons = decide(cand_table, base_table, prod_table)
    summary = {
        **metrics,
        "eval_fa_per_week": cand_table.loc["clean", "fa_per_week"],
        "eval_coverage": cand_table.loc["clean", "coverage"],
        "eval_mean_f1": cand_table.drop("clean")["f1"].mean(),
        "gate_pass": float(ok),
    }
    mv = log_and_register(
        candidate,
        params={
            "as_of": args.as_of.isoformat(),
            "window_months": args.window_months,
            **{k: "→".join(v) for k, v in windows.items()},
        },
        metrics=summary,
        alias="production" if ok else "staging",
        run_name=version,
    )
    print(cand_table.round(3).to_string())
    print("\nGATE:", "PASS → promoted to production" if ok else "FAIL → left at staging")
    for r in reasons:
        print("  -", r)
    if ok:
        try:
            r = httpx.post(f"{args.scorer_url}/reload-model", timeout=60)
            log.info("scorer reload: %s %s", r.status_code, r.text)
        except Exception as e:  # noqa: BLE001 — the scorer also polls the registry every 10 min
            log.warning(
                "could not reach scorer for reload (%s); it will pick the model up itself", e
            )
    print(f"registered {MODEL_NAME} v{mv.version}")


if __name__ == "__main__":
    main()
