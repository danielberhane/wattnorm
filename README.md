# Wattnorm — real-time contextual energy anomaly detection for buildings

Buildings account for roughly a third of global energy use, and much of it is wasted quietly: an
air-handling unit left running over a weekend, a lighting schedule that never switched back after a
holiday, a chiller cycling against a failed sensor. A raw meter reading cannot say whether 280 kW is
a problem — at 2 p.m. on a hot July weekday it is normal, at 2 a.m. on a Sunday in April it is not.

Wattnorm learns what a building *should* draw for the current time and weather, scores every
15-minute meter reading against that expectation as it arrives, and raises direction-aware alerts on
a Grafana dashboard — **too high = waste, too low = failure** — with the avoidable energy expressed
in kWh, CO₂e and cost so that a building manager, not a data scientist, can act on it. It was
developed on six years of data from a university research building. Historical data is replayed
through the live pipeline today; a real meter publishes the same message format tomorrow.

<img src="docs/img/dashboard.png" alt="Building Energy Anomalies dashboard — two weeks of real meter data, January 2020" width="100%">

*Two weeks of real meter data replayed through the live pipeline. Blue: metered demand · orange
dashed: expected demand · shaded: 90 % normal range · red: alerts. Lower panels: residual in σ and
the CUSUM that catches sustained deviations.*

<img src="docs/img/tiles.png" alt="Sustainability tiles" width="100%">

*Excess energy on HIGH alerts, converted with the grid emission factor and tariff (both editable on
the dashboard; the defaults are placeholders). A normal building reads zero.*

## Architecture

<img src="docs/img/architecture.svg" alt="Wattnorm system architecture — meter and weather over MQTT into the scorer, TimescaleDB and Grafana; trainer and MLflow registry supply the model" width="100%">

## Quick start

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

| Service | URL |
|---|---|
| Grafana | http://localhost:3000 (view without login; admin/admin to edit) |
| MLflow | http://localhost:5001 |
| Scorer health | http://localhost:8001/health |

## Connecting a real meter

Publish `{"meter_id": "...", "ts": "<ISO-8601 with offset>", "kw": <float>}` to
`building/<meter_id>/demand` (QoS 1) and weather to `weather/<station_id>/obs`. Nothing downstream
changes; the broker and database are `.env` settings.

## Data

`data/raw/demand.csv` (gitignored) is the 15-minute electric demand of Clark Hall, Cornell University,
in kW (the portal reports building electricity as power; the pipeline converts each slot to kWh as
`kW × 0.25 h`). The export's clock is UTC (`site.demand_tz` in `configs/default.yaml`); calendar
features are computed in `America/New_York`. The data are not redistributed with this repository.

> Cornell University, Facilities and Campus Services — Energy and Sustainability. *Energy Management and Control System (EMCS) Portal*, Clark Hall electric demand, 15-minute interval, January 2015 – May 2021. https://portal.emcs.cornell.edu (accessed 2022 for the original study; re-used here). Data are the property of Cornell University and are used for research and educational purposes.

Weather is the Open-Meteo ERA5 archive for the building's location (`make weather`).

## Research background

Wattnorm builds on my master's research at Western University, published as
[*An ensemble learning framework for anomaly detection in building energy consumption*](https://www.sciencedirect.com/science/article/pii/S0378778817306904)
(Araya, Grolinger, ElYamany, Capretz and Bitsuamlak, *Energy and Buildings* 144, 2017). The paper
combined CCAD-SW, a pattern-based autoencoder classifier over sliding windows, with support vector
regression and random forest predictors in a majority-vote ensemble that, on data from a building in
Brampton, Ontario, raised CCAD-SW's sensitivity by 3.6 % and cut its false-alarm rate by 2.7 %.
Wattnorm takes the prediction-based branch as its core and adds what a deployment needs: calibrated
bands, adaptation to a building that changes, a promotion gate, and a dashboard.

## Documentation

- [`docs/writeup.md`](docs/writeup.md) — methodology, data work, evaluation, and the defects found along the way
- [`docs/runbook.md`](docs/runbook.md) — operating guide for building managers
- [`docs/PLAN.md`](docs/PLAN.md) — design, phases and decisions

The Python package is `smartbuilding` (`src/smartbuilding/`).

## License

MIT — see `LICENSE`. The Clark Hall data are Cornell University's and are not covered by it.
