# Contextual anomaly detection for building energy: from a leaky notebook to a gated, real-time system

*A technical write-up of the Smartbuildings project — data, features, model selection, evaluation without labels, deployment and monitoring. Every number below comes from the repository at the commit this document ships with and can be reproduced with the commands in the appendix.*

---

## 1. The problem

A building's electricity demand is a function of *context*: time of day, day of week, season, holidays and weather. An anomaly is not "a high reading"; it is a reading that is high **for its context** — 280 kW at 2 a.m. on a Sunday in April is a problem, 280 kW at 2 p.m. on a July Tuesday is not. For a sustainability programme the two directions matter differently:

- **Too high for context** → energy waste (HVAC or lighting left running, scheduling errors). Quantifiable in kWh, CO₂e and cost.
- **Too low for context** → equipment failure, metering fault, or an unplanned change worth knowing about.

The operational constraint that shaped everything: a building manager will act on **a few alerts a week**. Above roughly three false alarms per week, a dashboard is ignored. The whole system is designed around that budget.

There are **no labelled anomalies**. Nobody recorded when the HVAC was left on. So the evaluation problem — how do you know it works? — is as central as the modelling problem.

### 1.1 Where the project started

The prior state was a Jupyter notebook (2022–2025): an H2O deep autoencoder on hourly sliding windows of demand plus weather, scored by reconstruction MSE and evaluated against synthetic anomalies. Three defects made its results unreliable, and each one taught something about the redesign:

| Defect in the notebook | Consequence | Design response |
|---|---|---|
| `StratifiedShuffleSplit` on 5-reading overlapping windows | ~80 % of every test window's values were also in training; ROC looked excellent and meant nothing | Strictly temporal splits; never shuffle a time series |
| Weather imputation fit on the whole frame | Test-period information in the training features | Imputer fit on the training years only |
| Reconstruction MSE as the score | Direction-blind: +80 kW and −80 kW give the same score; waste and failure are indistinguishable | Predict *expected* demand, score the **signed** residual |

## 2. Data

### 2.1 Demand

One meter, 15-minute interval, `2015-01-01 05:00` → `2021-05-31 03:45`, **221,387 readings**, mean 235 (units assumed kW), range 118–391.

Profiling before modelling (Phase 1, `gap_report`):

- **1.53 % of slots missing** across **200 gaps**; six gaps longer than a day; the longest **307 hours** (29 March → 11 April 2020). Gaps are kept as `NaN` on a complete 15-minute grid, never dropped, so the grid is honest about what the meter did and did not report.
- **Regimes.** 2015 ran ~9 % higher than 2016–2019 (equipment or metering changed); April 2020 onward runs ~10 % lower (lockdown). These are not noise — they are the reason the system needs a level tracker and a retraining path (§5.3, §8).
- **Daylight saving.** Timestamps are naive local time. A 15-minute grid built in local time has a duplicated hour every November and a missing one every March. The fix is to localise to `America/New_York` with `ambiguous="NaT", nonexistent="NaT"` and drop those slots (52 readings in 6.5 years), store everything in UTC, and compute calendar features back in local time.

### 2.2 Weather

Two scraped Weather Underground exports were in the repository: one metric (2016-01 → 2020-02, with `Pressure == 0` sensor nulls and a 294 km/h "gust"), one imperial (2015–2019). Different units, overlapping years, ending fourteen months before the demand data does. Neither is a defensible canonical source.

The system uses the **Open-Meteo historical archive** (ERA5-based reanalysis, free, no key) for Ithaca (42.44 N, −76.50 W), 2015-01 → 2021-05, 56,232 hourly rows, seven variables: temperature, dew point, relative humidity, wind speed, gusts, surface pressure, precipitation. Validated against the station export on the 35,808 overlapping hours:

| Variable | Correlation | Note |
|---|---|---|
| Temperature | 0.984 | MAE 1.95 °C |
| Dew point | 0.983 | |
| Pressure | 0.978 | constant ~25 hPa offset (station altitude vs surface pressure) — irrelevant to a tree model |
| Humidity | 0.795 | |
| Wind speed | 0.767 | |
| Gusts | 0.621 | the station column is mostly zeros; the export is the unreliable one |
| Precipitation | 0.310 | reanalysis precipitation is coarse; treated as a weak feature |

The CSVs remain as an offline fallback behind a unit-normalising loader (°F→°C, inHg→hPa, mph→km/h, in→mm). Hourly observations are aligned to the 15-minute grid by rounding to the hour and forward-filling at most four slots; an hour with no observation stays `NaN` for the imputer.

### 2.3 Splits

| Purpose | Period | Used for |
|---|---|---|
| Train | 2016-01 → 2017-12 | boosters, weather imputer |
| Calibrate | 2018 | band width, **all** rule thresholds |
| Test | 2019 (full year) | one evaluation with injected anomalies |
| Replay | 2020-01 → 2021-05 | streamed through the live system as if it were a meter |

2015 is excluded (different regime). Testing on a full year matters: an earlier draft tested on 2020-Q1 only, which would have evaluated every anomaly type in winter.

## 3. Features

The model is **context-only** by design: it answers *"what should this building draw right now?"*, not *"what will the next reading be?"*. Demand lags were deliberately excluded. A model with lags is a better forecaster, but a sustained anomaly becomes its "normal" within a few readings — precisely the anomaly a sustainability programme cares about most.

Fourteen features:

- **Calendar (local time)** — sin/cos of minute-of-day, day-of-week and day-of-year (cyclic encoding: 23:45 and 00:00 are neighbours, December is next to January); `is_holiday` (US federal); `is_weekend`.
- **Weather** — the seven Open-Meteo variables plus **heating degrees** `max(18 − T, 0)` and **cooling degrees** `max(T − 22, 0)`: the parts of temperature a building actually responds to.

Missing weather is filled by a `SeasonalMeanImputer`: the training-period mean for that (month, hour) in local time, with a global fallback. It is fit on the training years only and serialised with the model.

## 4. Model selection

### 4.1 Candidates considered

- **A. Residual-based (chosen).** Predict expected demand for the context; the anomaly is the signed, standardised residual. Gradient-boosted quantile regression.
- **B. Autoencoder (the incumbent).** Reconstruction error on hourly windows plus context. Direction-blind; the threshold is opaque; the notebook's own correlation table showed the windowed demand values dominate the reconstruction, i.e. it mostly learned *demand ≈ its neighbours*.
- **C. Ensemble of A and B.** Marginal recall gain on a single meter for double the operational surface.

Reasons A wins for this data: ~70k training rows of tabular features is the regime where gradient boosting is consistently at or above neural models; quantile loss gives calibrated bands natively; SHAP on trees is exact and cheap, which is what makes an alert explainable ("expected 231 kW because: time of day +38, weekday +12, temperature −6"); training takes ~10 s on CPU, so nightly retraining is trivial.

### 4.2 The model

Three LightGBM boosters with `objective="quantile"` at α = 0.05, 0.50, 0.95 (800 trees, learning rate 0.03, 31 leaves, `min_child_samples` 50). Predicted quantiles are re-sorted row-wise so `q05 ≤ q50 ≤ q95` (quantile crossing fix).

Calibration-year (2018) fit: MAE 9.9 kW on a 235 kW mean (4.2 %). Raw 5–95 % band coverage on 2018: **0.60** — the model is overconfident out of year.

### 4.3 Calibration

One scalar `c` widens the raw band until it covers the target 90 % on the calibration year:

```
half = c · (q95 − q05) / 2        band = q50 ± half
σ    = half / 1.645               z    = (y − q50) / σ
```

Coverage-based calibration of a raw quantile band is standard (split-conformal in spirit); what turned out to matter is **which residual it is fit on** — see §5.3.

## 5. From residual to alert

### 5.1 First attempt — and why it failed

The obvious detector: `SPIKE` if |z| > 4 on one slot; two-sided **CUSUM** on z (k = 0.5, h = 5) for sustained deviation; `STUCK` for eight identical consecutive readings.

On the held-out year it produced **12 false alarms per week**. Diagnosis on 2019 scores:

- lag-1 autocorrelation of z = **0.80**; lag-96 (one day) = 0.45;
- monthly mean z ranged from **−1.48 (January) to +0.57 (November)**;
- daily-mean z had σ = 0.88; 96 of 366 days had |mean z| > 1.

The building's level, relative to a model trained on 2016–17, wanders by more than a σ from month to month. CUSUM with k = 0.5 correctly accumulates any persistent 0.5σ bias — and flagged entire months. A manager cannot act on "January was low".

Two textbook parameters were also wrong for this cadence: CUSUM with k = 0.5, h = 5 has an average run length of ~465 steps two-sided on *white noise* — at 96 readings a day, that is a false sustained alert every five days per direction before any real structure is considered.

### 5.2 Level tracking

The fix that kept the design simple: a slow, robust **EWMA of the kW residual** (half-life 3 days) tracks the building's current level relative to the model. Expected demand, the band and z are all computed relative to it:

```
expected = q50 + level_kw          z = (y − expected) / σ
level_kw += α · clip(y − expected, ±2σ)
```

Design notes that came out of testing rather than theory:

- **Track in kW, not in z.** A first version tracked the bias in z-units; because the band is heteroscedastic (σ varies with context), `bias·σᵢ` gave a different kW correction at every slot and left a ±2σ daily oscillation. A level shift is a kW quantity.
- **Clip the update at ±2σ.** Without clipping, a +30 % offset lasting six hours pulled the level up enough that the return to normal produced a *rebound* `SUSTAINED_LOW` alert. With clipping, an anomaly can move the level by at most 2σ·α per slot (~0.14σ over a six-hour event), while a permanent shift is still absorbed within a week or two.
- **Winsorise the CUSUM input at the spike threshold.** Otherwise a single 30σ slot trips CUSUM by itself and gets labelled "sustained". Clipped at 5, one slot contributes at most 3.5, so `SUSTAINED` genuinely means ≥ ~1 hour.
- **`SUSTAINED` supersedes `SPIKE` in the same direction.** A spike that keeps going *is* the sustained event; the manager should see one alert, not two.

### 5.3 Calibrating on the same residual the scorer uses

The first retrain run (as-of September 2020) calibrated its band on March–May 2020 — the lockdown — and produced a band covering **100 %** of readings that detected nothing. The promotion gate refused it, which is what the gate exists for, but the cause was a design inconsistency: the scorer judged *level-adjusted* residuals while the calibrator was fit on *raw* ones, so a regime break inside the calibration window inflated the band.

Both now use one `track_level()` routine; the calibration factor is fit on `y − level` (with a fixed-point iteration because the clipping depends on σ, which depends on the factor). A unit test asserts that reported calibration coverage equals what `Detector.score` realises on the same frame across a −30 kW mid-window break (they now agree to within 0.015; before the fix they differed by 0.04). Side effect: the 2018 factor fell from **2.12 to 1.59** — the raw residual had been inflated by exactly the level drift the tracker removes.

### 5.4 Threshold selection — on 2018 only

A small grid over k ∈ {1.0, 1.5, 2.0}, h ∈ {12, 16}, spike threshold ∈ {4, 5}, evaluated with injected anomalies on the **calibration** year, never on 2019:

| k | h | spike | FA / week | recall: offset · dropout · spike · drift · schedule |
|---|---|---|---|---|
| 1.0 | 12 | 4 | 2.57 | 1.0 · 1.0 · 1.0 · 1.0 · 1.0 |
| 1.0 | 16 | 5 | 1.98 | 1.0 · 1.0 · 1.0 · 1.0 · 1.0 |
| **1.5** | **12** | **5** | **1.11** | **1.0 · 1.0 · 1.0 · 1.0 · 0.83** |
| 2.0 | 12 | 5 | 0.61 | 1.0 · 1.0 · 1.0 · 1.0 · 0.83 |

`k` is the dominant lever. k = 1.5 was chosen over 2.0 to keep sensitivity to offsets around +10 % — the overnight-waste case — at the cost of ~0.5 more false alarms a week.

## 6. Evaluation without labels

### 6.1 Injection

Six anomaly types are injected into the *raw* series before scoring (so lags, level tracking and imputation all see exactly what production would):

| Type | What it simulates | Injected as |
|---|---|---|
| spike | transient load / meter glitch | one slot × 1.3–2.0 |
| sustained_offset | HVAC or lighting left on | +15–40 % for 2–12 h |
| dropout | equipment failure | −30–60 % for 1–6 h |
| slow_drift | degradation | +1 %/day ramp over 14 days |
| schedule_shift | building running its day profile at night | one day's readings rolled by 6 h |
| stuck_meter | comms/metering fault | constant value for 2–6 h |

Three of each per seed, five seeds, placed at least a day apart (longest first, so 14-day drifts do not get squeezed out) — 90 labelled events per evaluation on top of the year's real behaviour.

### 6.2 Metrics

Event-level, not slot-level: a labelled event is *detected* if any alert overlaps it (±1 h tolerance). Reported per type: recall, median time-to-detect, and — on the clean, un-injected year — **false alarms per week** and **band coverage**. Precision and F1 are computed but not used for decisions: with three injected events per type against a year of real alerts they only proxy the false-alarm count.

### 6.3 Baseline

A **seasonal-naive** detector — expected = same slot last week, σ per (weekday, slot) from the fit period — run through the *same* calibrator and the *same* alert rules. Only the source of the expectation differs, so the comparison isolates what the model adds.

### 6.4 Result on the held-out year (2019)

| | Detector | Seasonal-naive |
|---|---|---|
| False alarms / week (clean) | **1.73** | 0.75 |
| Band coverage (clean) | 0.81 | 0.88 |
| Recall: spike | 1.00 | 1.00 |
| Recall: sustained offset | 1.00 (detected in 12 min) | 1.00 (54 min) |
| Recall: dropout | 1.00 | 1.00 |
| Recall: slow drift | 0.93 (~2.5 days) | 0.80 (~6 days) |
| Recall: schedule shift | 1.00 | 0.80 |
| Recall: stuck meter | 1.00 | 1.00 |

Reading it honestly:

- The detector catches everything the baseline catches, catches drift and schedule shifts the baseline misses, and reacts 4–5× faster to offsets — within the three-per-week budget.
- The baseline is *quieter*. Its expectation is last week's reading, which already contains the building's current level; it pays for that with blindness to anything that also happened last week and slower reaction.
- **Coverage 0.81** for a band calibrated to 0.90 on 2018 is a real finding, not a rounding issue: a model frozen at end-2018 fits 2019 worse. It is the quantitative argument for §8.

A sanity check with the only "labels" nature provided: scoring 2016–2021 with the 2018-calibrated model shows almost nothing in the training years, sparse alerts in 2018–19, and `SUSTAINED_LOW` dominating **March–April 2020** — the lockdown — followed by a wave of `SUSTAINED_HIGH` in May as partial reopening reads as high against the newly learned low level. Nobody told the model about COVID.

## 7. Deployment

### 7.1 Shape

Five always-on containers, two on demand, one image, one `docker compose up`:

```
Meter / Open-Meteo ──MQTT──▶ mosquitto ──▶ Scorer ──▶ TimescaleDB ──▶ Grafana ──▶ building manager
                                             ▲                │
                                     MLflow registry ◀── Trainer ◀─── history
```

- **MQTT is the only contract.** `building/<meter_id>/demand` with `{"meter_id", "ts" (ISO-8601 with offset), "kw"}`, QoS 1. A real meter gateway and the replay simulator publish the same shape; naive timestamps are rejected so nobody guesses a time zone.
- **`Detector.score()` is the single scoring path** for offline evaluation, the live scorer and the MLflow pyfunc wrapper; a test pins that batch scoring and one-reading-at-a-time scoring produce identical output. What was measured in §6 is what runs.
- **Scorer** is a transport- and storage-agnostic core behind a `Sink` protocol (unit-tested with an in-memory sink), wrapped by a thin MQTT consumer and a FastAPI `/health` + `/reload-model`. Per-meter state: level, CUSUM sums, open alerts. Stale weather (> 3 h) degrades to seasonal means and is flagged, never fatal. A reading more than a day older than the last is treated as a replay restart and resets that meter's state.
- **Alerts** are opened and closed on transitions and carry observed vs expected kW, peak z, accumulated excess kWh → CO₂e and cost, and the top-3 SHAP drivers of the expectation.
- **Replay** streams the untouched 2020-01 → 2021-05 readings at a chosen speed. Full backfill of 48,172 readings: every one scored (48,173 rows; the +1 is a boundary slot), ~15 ms per reading, no message loss once the broker queue was made unbounded.

### 7.2 Registry and promotion

Models are logged to MLflow as a pyfunc with the boosters, imputer, calibration factor and rules bundled; aliases `@staging` and `@production` replace deprecated stages. The scorer loads `@production` at start, polls the alias every ten minutes and hot-swaps. On swap the level bias — which is relative to the *old* model's q50 — is reset and CUSUM is rebuilt with the new rules; a test drives an adapted old model, swaps in a retrained one and asserts that quiet days produce no alerts (they did before this fix, reproduced by the code review).

### 7.3 Lessons from running it

Things that only showed up when data actually flowed: macOS AirPlay squats on port 5000 (MLflow moved to 5001); another Postgres owned 5432; Grafana's table panel silently fails to render without its full default option block; a 10-second dashboard refresh cancels TimescaleDB queries before first paint; mosquitto's default 1,000-message queue drops QoS 1 messages when a 30,000× replay outruns a ~25 reading/s scorer.

## 8. Monitoring and retraining

**Model health is SQL on the `scores` table**, on the same dashboard the manager uses: seconds since last reading, scoring latency p95, 24-hour band coverage, 24-hour MAE, stale-weather share, model version. Three provisioned alert rules: an anomaly is open; no score for 30 minutes; coverage under 80 % for a day. No separate metrics stack — the design goals were simplicity and observability by construction, and every score row already carries `model_version`.

**Retraining** runs on a trailing 24-month window: train on the oldest 18 months, calibrate on the next three, evaluate on the most recent three with injected anomalies — against the seasonal-naive baseline *and* the current production model. The **gate** promotes only if recall per type is ≥ 0.9× the baseline's (or ≥ 0.8 absolute — with ten injected events per type recall is quantised to 0.1 and one miss must not block a promotion), clean false alarms ≤ 3/week, coverage within 0.80–0.97, and no recall regression against production. In this project the gate has refused every candidate so far, each time for a defensible reason (a band that covered everything; a lockdown-calibrated model less sensitive to offsets than the 2018 one). A gate that never says no is not a gate.

**Drift** is watched through coverage and MAE rather than a distribution-drift library: for a single meter, the band's realised coverage *is* the drift statistic that matters.

## 9. Limits and open questions

- **Units and factors.** `Demand` is assumed to be kW; the CO₂e factor (0.25 kg/kWh) and tariff ($0.14/kWh) are placeholders. For upstate New York on NYISO the factor is likely 0.1–0.2 and varies by hour; an hourly marginal factor would change the sustainability numbers materially.
- **A frozen model degrades within a year** (coverage 0.90 → 0.81). Retraining is part of the system, not an option — and the gate must be tuned so it can actually pass.
- **Retraining is file-fed** in replay mode; a real deployment needs the trainer to read `readings`/`weather_obs` from the database (same frame shape; a small reader).
- **Small offsets** below ~8 % are below the context model's sensitivity by design (k = 1.5). A second, lag-based model behind the same `Detector` interface would catch them at the cost of the "new normal" problem; it was left as an extension until evaluation shows the need.
- **Synthetic evaluation** measures detectability of *assumed* anomaly shapes. The real building's anomalies may not look like any of the six; the 2020 replay is the only real-world check, and it is unlabelled.
- **One meter.** Per-(hour, weekday) residual scales, holiday calendars (academic vs federal) and cold-start behaviour (4–6 weeks of readings across a temperature range before the band is trustworthy) would all need revisiting for a portfolio of buildings.

## 10. What I would tell someone starting the same project

1. **Profile the gaps, the units, the time zone and the regimes before modelling.** Half the defects in the original notebook were data handling, not ML.
2. **Split by time and evaluate on a whole year.** Anything else flatters you.
3. **Build the evaluation harness before tuning anything.** Every design change above — level tracking, clipping, winsorising, calibration on the adjusted residual — was found by a number the harness produced, not by inspection.
4. **Tune on the calibration year, look at the test year once.** Write down when you looked twice.
5. **Score in production with the exact function you evaluated.** One path, one test that batch equals streaming.
6. **Design the gate to say no**, then make sure it can also say yes.

---

## Appendix — reproducing the numbers

```
make setup && make weather          # environment; Open-Meteo archive → data/processed/
MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run python scripts/train.py
uv run python scripts/evaluate.py --baseline --split calibrate      # §5.4 tuning data (2018)
uv run python scripts/evaluate.py --baseline                        # §6.4 held-out result (2019)
uv run python scripts/evaluate.py --full-series                     # §6.4 sanity check (2016–2021)
make up && make train && make eval && make backfill                 # §7 live stack + full replay
make retrain AS_OF=2020-09-01                                       # §8 gate behaviour
uv run pytest -q                                                    # 102 unit tests + 1 integration
```

Design record and decision log: `docs/PLAN.md`. Operations: `docs/runbook.md`.
