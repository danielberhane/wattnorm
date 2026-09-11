# Smartbuildings — real-time contextual energy anomaly detection

Learns the *expected* electric demand of a building for any (time, weather) context, scores each
15-min reading against it, and raises direction-aware alerts (too high = waste, too low = failure)
on Grafana with excess kWh → CO₂e/$. Replays historical data over MQTT now; a real meter later.
Design plan: `docs/PLAN.md`.

## Design goals (in priority order — check changes against these)
1. Simplicity / YAGNI — one model, one DB, one dashboard. Don't add a component or abstraction until something needs it.
2. Maintainability — logic lives in `src/smartbuilding/`, tested, typed, configured in one place.
3. Ease of use — everything runs through `make` targets; alerts are readable by a non-engineer.
4. Reproducibility — temporal splits only (never shuffle time series), seeded evaluation, pinned deps, versioned models.
5. Clear boundaries — data → features → model → scorer → DB → Grafana. The MQTT payload is the only contract a meter must meet.
6. Config over code — `.env` + `configs/default.yaml`; no hardcoded paths, hosts, or factors.
7. Fail-safe defaults — degrade (stale weather → seasonal means), dedupe alerts, make rollback one command.
8. Observable by construction — every score row carries `model_version`; health is SQL on `scores`.
9. Extensible without rework — new model/meter/broker plugs in behind existing interfaces.
10. Secrets never in the repo — `secrets/` and `.env` are gitignored and must not be read or printed.

## Layout
- `src/smartbuilding/` — the package. `data/` (loaders, DST, grid, splits), `features.py`, `model.py`, `eval.py`, `mqtt.py`, `simulator.py`, `scorer.py`, `db.py`, `retrain.py`.
- `tests/` — pytest, one file per module. `-m integration` needs the compose stack.
- `configs/default.yaml` — all tunables. `.env.example` — runtime env.
- `deploy/` — compose assets (mosquitto, init.sql, Grafana provisioning, Dockerfile).
- `data/raw/` (gitignored) — demand.csv, weather CSVs. `data/processed/` — parquet caches.
- `legacy/` — the original notebooks/scrapers/dash demos. Read for reference; never import from.

## Commands
- `make setup` — `uv sync --all-extras` + pre-commit install
- `make lint` / `make test` — ruff check+format / pytest (unit)
- `make train` / `make eval` — train + register model / evaluate vs baseline on 2019
- `make up` / `make down` / `make replay [START=… INJECT=…]` — compose stack / simulator
- `make retrain [AS_OF=…]` — trailing-window retrain with promotion gate

## Working rules
- **Verify after every code change.** After editing `src/` or `tests/`, run `uv run pytest -q` (or the relevant test file) and confirm it passes before moving on. The ruff hook runs automatically on edit — fix anything it reports immediately.
- TDD: write the failing test first, then the minimum implementation, then refactor.
- Time series discipline: train < calibrate < test in time; fit imputers/scalers/thresholds on train/calibration only.
- Timestamps: store UTC (`ts`), compute calendar features in `America/New_York`. Handle DST via `localize()`; never build a 15-min grid in naive local time.
- Units: weather is metric internally (`temp_c`, `pressure_hpa`, `wind_kph`, `precip_mm`); demand is `kw`.
- Don't add dependencies, services, alert types, or config keys not in `docs/PLAN.md` without saying why.
- Python 3.12, `uv` for everything (`uv run …`, never bare `python`). Ruff for lint+format (line length 100).
- Never read, print, or commit anything under `secrets/` or `.env`.
