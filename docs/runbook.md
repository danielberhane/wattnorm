# Runbook — Building Energy Anomalies

## URLs (local stack)
Dashboard http://localhost:3000/d/smartbuilding-overview (viewer needs no login; admin/admin to edit) ·
MLflow http://localhost:5001 · scorer http://localhost:8001/health

## What an alert means and what to do
| Alert | Meaning | First action |
|---|---|---|
| **SUSTAINED_HIGH** | Consumption has been above the expected band for ≥ ~1 h given time + weather | Check HVAC / lighting schedules, equipment left running, occupancy events. "Excess kWh / CO₂ / $" quantify the waste so far. |
| **SUSTAINED_LOW** | Consumption well below expectation for ≥ ~1 h | Equipment or sub-metering failure, unplanned closure, or a real efficiency gain. If it persists > 3 days the system adopts it as the new normal. |
| **SPIKE_HIGH / SPIKE_LOW** | One 15-min reading > 5σ from expected | Usually a transient load or a meter glitch. If it continues it becomes SUSTAINED automatically. |
| **STUCK** | 8+ identical consecutive readings (2 h) | Metering / communications fault. Check the gateway. |

"Why (top drivers)" in the alerts table lists the three features that most shaped the *expectation* (e.g. `sin_tod +38 kW` = time of day pushed expected demand up). It explains the baseline, not the cause of the anomaly.

Acknowledge an alert: `UPDATE alerts SET acknowledged = true WHERE id = <id>;` (a button can be added to the table later).

## System health row
- **Since last reading** grows → simulator/meter → mosquitto → scorer chain is broken. `docker compose logs scorer`.
- **Band coverage** < 0.80 for a day → model is stale or the building changed regime → `make retrain`.
- **Stale-weather share** > 0 → weather publisher down; expectations fall back to seasonal means (wider error).

## Operations
- Start/stop: `make up` / `make down` (`make down VOLUMES=1` wipes data; `deploy/init.sql` only runs on a fresh volume).
- Backfill history: `make backfill` (background, ~40 min). Live-paced demo: `make live START=2020-04-01 [INJECT=sustained_offset]`.
- Train / evaluate / promote: `make train` → `make eval` (promotes `@production` on PASS). Scorer hot-swaps within 10 min or on `curl -X POST localhost:8001/reload-model`.
- Nightly retrain: `make retrain [AS_OF=YYYY-MM-DD]` — trailing 24 months, gate vs baseline **and** current production; add to cron: `0 2 * * * cd <repo> && make retrain`.
- Rollback: in MLflow set alias `production` back to the previous version (UI → Models → smartbuilding-detector → aliases), then `curl -X POST localhost:8001/reload-model`.
- Truncating tables while the scorer runs is safe: a reading > 1 day older than the last one resets that meter's state.

## Connecting a real meter
Publish to `building/<meter_id>/demand` payload `{"meter_id": "...", "ts": "<ISO-8601 with offset>", "kw": <float>}` (QoS 1), and weather to `weather/<station_id>/obs`
(`temp_c, dewpoint_c, rh, wind_kph, gust_kph, pressure_hpa, precip_mm`). Nothing downstream changes.
Set `MQTT_HOST/PORT/TLS` in `.env` to point at a managed broker (e.g. AWS IoT Core) if needed.
Cold start: the band is trustworthy after ~4–6 weeks of readings spanning some temperature range; until then expect wider bands and treat alerts as advisory.
Change the dashboard default time range back to `now-24h` for live use.

## Known limits
- Demand units assumed kW; emission factor (0.25 kg/kWh) and tariff ($0.14/kWh) are placeholders — set real values in `.env` (used for the alerts table) **and** the dashboard text boxes (used for the tiles).
- A frozen model degrades over a year (2019 coverage 0.81 with the 2018-calibrated model); retraining is part of the system, not optional.
