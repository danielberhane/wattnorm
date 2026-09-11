---
name: stack
description: Operate the local Docker Compose stack (mosquitto, timescaledb, mlflow, grafana, scorer, simulator). Use to start/stop the stack, run a replay, check health, inspect alerts in the DB, or debug a container.
---

# Stack operations

- Start: `make up` (builds the image, starts core services). Grafana http://localhost:3000 (admin/admin), MLflow http://localhost:5000, scorer http://localhost:8000/health.
- Stop: `make down` (keeps volumes) · `make down VOLUMES=1` wipes data.
- Replay: `make replay` (defaults: 2020-01-01, 60×) · `make replay START=2019-06-01 INJECT=sustained_offset`.
- Health: `curl -s localhost:8000/health | jq` → model version + last score ts.
- Alerts: `docker compose exec timescaledb psql -U postgres -d smartbuilding -c "select alert_type, started_at, ended_at, peak_z, excess_kwh from alerts order by started_at desc limit 20"`.
- Logs: `docker compose logs -f scorer` (or simulator / grafana).

Debug order when nothing shows on the dashboard: simulator publishing? (`docker compose logs simulator | tail`) → scorer receiving? (`logs scorer`) → rows in `readings`/`scores`? → Grafana datasource test.
Never change `deploy/init.sql` for an existing volume without `make down VOLUMES=1`; the DDL only runs on first init.
