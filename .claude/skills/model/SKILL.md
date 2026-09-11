---
name: model
description: Train, evaluate, inspect, or retrain the anomaly detector. Use when changing features, thresholds, or model params, or when asked why an alert fired.
---

# Model workflow

- Train: `make train` → trains context-only LightGBM quantile model on 2016–2017, calibrates on 2018, logs to MLflow, registers `smartbuilding-detector` with alias `@staging`.
- Evaluate: `make eval` → injected anomalies over 2019 + seasonal-naive baseline. Gate: F1 ≥ baseline on every type, clean false alarms ≤ 3/week, coverage 85–95 %. Add `--promote` to alias a passing model `@production`.
- Tune rules ONLY with `--split calibrate` (2018). Never tune on 2019. Headline metrics: fa_per_week, recall, ttd_median_min, coverage (precision/F1 are dominated by injection density).
- Full-series sanity: `uv run python scripts/evaluate.py --full-series` — expect SUSTAINED_LOW around 2016-01 and from 2020-04 (COVID).
- Retrain: `make retrain AS_OF=YYYY-MM-DD` — trailing 24-month window, gate, promote.

Rules: never fit anything (imputer, scaler, calibration factor, thresholds) on the test year. Change one thing per experiment and log it to MLflow with a descriptive run name. If eval regresses, compare runs in MLflow before changing thresholds. Prefer fixing features over widening bands.
