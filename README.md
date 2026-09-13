# Smartbuildings — real-time contextual energy anomaly detection

Buildings account for roughly a third of global energy use, and a large share of that is wasted
quietly: an air-handling unit left running over a weekend, a lighting schedule that never switched
back after a holiday, a chiller cycling against a failed sensor. A meter records all of it, but a
raw reading cannot say whether 280 kW is a problem — at 2 p.m. on a hot July weekday it is normal,
at 2 a.m. on a Sunday in April it is not. An anomaly is a reading that is unusual *for its context*:
the time, the day, the season, the weather.

This repository is a complete, deployable system built on that idea, developed on six years of
15-minute electricity data from a university research building. It learns what the building
*should* draw for the current time and weather, scores every meter reading against that
expectation as it arrives, and raises direction-aware alerts on a Grafana dashboard — **too high =
waste, too low = failure** — with the avoidable energy translated to CO₂e and cost so that a
building manager, not a data scientist, can act on it. Historical data is replayed through the
live pipeline today; a real meter publishes the same message format tomorrow.

It is an evolution of work I began for my master's at Western University, published as
[*An ensemble learning framework for anomaly detection in building energy consumption*](https://www.sciencedirect.com/science/article/pii/S0378778817306904)
(Araya, Grolinger, ElYamany, Capretz and Bitsuamlak, *Energy and Buildings* 144, 2017). That paper
introduced CCAD-SW, a pattern-based classifier that scores overlapping sliding windows of
consumption with an autoencoder, and combined it with two prediction-based classifiers (support
vector regression and random forest) in a majority-vote ensemble that, on real consumption data from
a building in Brampton, Ontario, raised the sensitivity of CCAD-SW by 3.6 % and cut its false-alarm
rate by 2.7 %. This system takes the prediction-based branch of that framework as its core
and adds what a paper does not need but a deployment does: calibrated bands, adaptation to a
building that changes, a promotion gate, and a dashboard.

The methodology, data work, evaluation and the defects found along the way are written up in full
in [`docs/writeup.md`](docs/writeup.md).

<img src="docs/img/dashboard.png" alt="Building Energy Anomalies dashboard — two weeks of real meter data, January 2020" width="100%">

*Real 15-min meter data replayed through the live pipeline (no synthetic values). Blue = metered demand;
orange dashed = expected demand for that time and weather, adjusted for the building's recent level;
shaded band = the 90 % normal range; red = alerts. Below: the residual in σ and the CUSUM that
catches sustained deviations.*

<img src="docs/img/tiles.png" alt="Sustainability tiles" width="100%">

*Excess energy recorded on HIGH alerts that started in the selected range, times the grid emission
factor and tariff (both editable on the dashboard — the defaults are placeholders). Readings outside an
alert are not counted, so a normal building reads zero.*

## How it works

<img src="docs/img/architecture.svg" alt="Architecture — meter and weather over MQTT into the scorer, TimescaleDB and Grafana; trainer and MLflow registry feed the model" width="100%">

- **Model** — LightGBM quantile regression (q05/q50/q95) on calendar (cyclic time-of-day, weekday,
  day-of-year, holidays) and weather (temperature, humidity, wind, pressure, heating/cooling degrees).
  No demand lags: the model answers *what should this building draw now*, so a sustained anomaly
  never becomes "normal" within hours.
- **Band** — one scalar calibration so the band covers 90 % of ordinary readings; a slow level
  tracker (3-day half-life, robust to anomalies) follows genuine regime changes.
- **Rules** — `SPIKE` (one reading > 5σ), `SUSTAINED_HIGH/LOW` (CUSUM), `STUCK` (2 h identical).
  Each alert carries observed vs expected kW, excess kWh → CO₂e/$, and the top-3 SHAP drivers of the
  expectation.
- **Evaluation without labels** — six anomaly types injected into a held-out year, event-level
  recall / false alarms per week / time-to-detect, against a seasonal-naive baseline sharing the
  same rules. Splits are strictly temporal: train 2016–17 · calibrate 2018 · test 2019 · replay
  2020-01 → 2021-05.
- **Operations** — MLflow registry with a promotion gate, scorer hot-swap, trailing-window retrain
  (`make retrain`, cron-able), one Grafana dashboard whose health row is plain SQL on the `scores`
  table. `/health` reports the model version and the rules it is running.

## Terms used above

- **Quantile regression** — a model that predicts a chosen percentile of demand (here the 5th, 50th and 95th) instead of only the average, so it gives a range as well as a central estimate.
- **Band / coverage** — the range between the 5th and 95th percentile predictions, scaled so that 90 % of ordinary readings fall inside it; coverage is the share that actually does.
- **Residual and σ** — the gap between the metered and expected demand, expressed in units of the band's half-width so that "3σ" means the same thing in summer and winter.
- **Level tracker** — a slow-moving estimate of how far the building's baseline has shifted from the model's expectation, so a permanent change in occupancy does not look like a permanent anomaly.
- **CUSUM** — a running sum of the residual that grows while demand stays on one side of expectation and resets otherwise; it catches small but persistent deviations that no single reading would.
- **SHAP drivers** — the features (hour, temperature, weekday …) that contributed most to the expected value for that reading, used to explain each alert.
- **Seasonal-naive baseline** — the simplest competitor: expected demand equals the reading one week earlier at the same time.
- **Recall / false alarms per week** — the share of injected anomalies the detector catches, and how many alerts it raises on data with no anomaly injected.
- **MQTT** — a lightweight publish/subscribe messaging protocol widely used by meters and IoT gateways.
- **TimescaleDB** — PostgreSQL with time-series extensions, so Grafana queries it with plain SQL.
- **MLflow registry / hot-swap** — MLflow stores each trained model with its metrics under a version and an alias (`@staging`, `@production`); the scorer can load a newly promoted version without restarting.

## Run it

```
make setup      # uv sync + pre-commit
make test       # unit tests
make up         # mosquitto + timescaledb + mlflow + grafana + scorer   (Docker Compose)
make train      # train + calibrate in the stack → MLflow @staging
make eval       # evaluate vs baseline on the held-out year; promotes @production on PASS
make backfill   # load the whole 2020-01 → 2021-05 history fast (background)
make live       # human-paced demo: one building day per ~15 min   (START=… INJECT=sustained_offset)
make retrain    # trailing-window retrain with promotion gate  (AS_OF=YYYY-MM-DD)
make figures    # regenerate docs/img/fig*.png and numbers.json from the stack's model and database
```

Grafana http://localhost:3000 (view without login; admin/admin to edit) · MLflow http://localhost:5001 ·
scorer http://localhost:8001/health

## Data

`data/raw/demand.csv` (gitignored) is the 15-minute electric demand of Clark Hall, Cornell University,
in kW (the portal reports building electricity as power; the pipeline converts each slot to kWh as
`kW × 0.25 h`). The export's clock is UTC (`site.demand_tz` in `configs/default.yaml`); calendar
features are computed in `America/New_York`. The data are not redistributed with this repository.

> Cornell University, Facilities and Campus Services — Energy and Sustainability. *Energy Management and Control System (EMCS) Portal*, Clark Hall electric demand, 15-minute interval, January 2015 – May 2021. https://portal.emcs.cornell.edu (accessed 2022 for the original study; re-used here). Data are the property of Cornell University and are used for research and educational purposes.

Weather is the Open-Meteo ERA5 archive for the building's location (`make weather`). Design, phases and
decisions: `docs/PLAN.md`. Operating guide for building managers: `docs/runbook.md`.

## Connecting a real meter

Publish `{"meter_id": "...", "ts": "<ISO-8601 with offset>", "kw": <float>}` to
`building/<meter_id>/demand` (QoS 1) and weather to `weather/<station_id>/obs`. Nothing downstream
changes; the broker and database are `.env` settings.

## License

MIT — see `LICENSE`. The Clark Hall data are Cornell University's and are not covered by it.
