# Smartbuildings — real-time contextual energy anomaly detection

Learns what a building *should* draw for the current time and weather, scores every 15-minute
reading against it, and raises direction-aware alerts on Grafana — too high = waste, too low =
failure — with the avoidable kWh translated to CO₂e and cost.

```
make setup      # uv sync + pre-commit
make test       # unit tests
make up         # mosquitto + timescaledb + mlflow + grafana + scorer
make train      # train + calibrate in the stack → MLflow @staging
make eval       # evaluate vs baseline on the held-out year; promotes @production on PASS
make backfill   # load the whole 2020-01 → 2021-05 history fast (background)
make live       # human-paced demo: one building day per ~15 min   (START=… INJECT=sustained_offset)
make retrain    # trailing-window retrain with promotion gate  (AS_OF=YYYY-MM-DD)
```

Grafana: http://localhost:3000 (admin/admin) · MLflow: http://localhost:5001 · scorer: http://localhost:8001/health

Design, phases and decisions: `docs/PLAN.md`. Data files live in `data/raw/` (gitignored):
`demand.csv` (15-min kW, 2015–2021) plus weather from Open-Meteo (`make weather` fetches it).
