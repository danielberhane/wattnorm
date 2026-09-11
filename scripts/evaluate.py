"""Evaluate a detector against injected anomalies on the test year and the seasonal-naive baseline.

uv run python scripts/evaluate.py --model models:/smartbuilding-detector@staging --baseline
uv run python scripts/evaluate.py --model models:/smartbuilding-detector@staging --full-series
"""

import argparse
import os
import sys

import mlflow
import pandas as pd

from smartbuilding.config import load_config
from smartbuilding.data.splits import temporal_split
from smartbuilding.eval import BaselineDetector, run_eval
from smartbuilding.pipeline import gate, load_dataset
from smartbuilding.registry import MODEL_NAME, load_detector

pd.set_option("display.width", 140)


def full_series(detector, cfg) -> None:
    data = load_dataset(cfg, cfg.splits.train[0], cfg.splits.replay[1])
    scores, _ = detector.score(data)
    scores["month"] = scores["ts"].dt.tz_convert(cfg.site.tz).dt.strftime("%Y-%m")
    flagged = scores[scores["alert_type"] != ""]
    by_month = (
        flagged.assign(kind=flagged["alert_type"].str.split("|"))
        .explode("kind")
        .groupby(["month", "kind"])
        .size()
        .unstack(fill_value=0)
    )
    print("alert slots per month (expect SUSTAINED_LOW to dominate from 2020-04):")
    print(by_month.to_string())
    print(f"\nmean z by year:\n{scores.groupby(scores['ts'].dt.year)['z'].mean().round(2)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=f"models:/{MODEL_NAME}@staging")
    ap.add_argument("--baseline", action="store_true", help="also evaluate seasonal-naive")
    ap.add_argument("--full-series", action="store_true", help="score 2016→2021 and summarise")
    ap.add_argument("--promote", action="store_true", help="on gate PASS, alias → production")
    ap.add_argument("--n-per-type", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "calibrate"],
        help="evaluate on the test year (final report) or the calibration year (tuning)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
    detector = load_detector(args.model)
    print(f"model: {detector.model_version}")

    if args.full_series:
        full_series(detector, cfg)
        return

    data = load_dataset(cfg, cfg.splits.train[0], cfg.splits.test[1])
    parts = temporal_split(data, cfg.splits.as_bounds())
    test = parts[args.split]
    print(f"evaluating on split '{args.split}' ({len(test)} slots)")
    ours = run_eval(detector, test, n_per_type=args.n_per_type, seeds=range(args.seeds))
    print("\n=== detector ===")
    print(ours.round(3).to_string())

    if not args.baseline:
        return
    base_fit = parts["train"] if args.split == "calibrate" else parts["calibrate"]
    base = BaselineDetector.fit(base_fit.dropna(subset=["kw"]), cfg.rules, cfg.site.tz)
    base_table = run_eval(base, test, n_per_type=args.n_per_type, seeds=range(args.seeds))
    print("\n=== seasonal-naive baseline ===")
    print(base_table.round(3).to_string())

    ok, reasons = gate(ours, base_table)
    print("\nGATE:", "PASS" if ok else "FAIL")
    for r in reasons:
        print("  -", r)
    mlflow.set_experiment("smartbuilding")
    with mlflow.start_run(run_name=f"eval-{detector.model_version}"):
        mlflow.log_params({"model": args.model, "n_per_type": args.n_per_type, "seeds": args.seeds})
        for kind, row in ours.iterrows():
            for col in ["precision", "recall", "f1", "ttd_median_min", "fa_per_week", "coverage"]:
                if row[col] == row[col]:
                    mlflow.log_metric(f"{kind}_{col}", float(row[col]))
        mlflow.log_metric("gate_pass", float(ok))
    if ok and args.promote and detector.model_version.startswith(MODEL_NAME):
        version = detector.model_version.rsplit("/", 1)[-1]
        mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version)
        print(f"promoted {MODEL_NAME} v{version} → alias 'production'")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
