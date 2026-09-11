.PHONY: setup lint test train train-local eval eval-local up down replay retrain logs
START ?= 2020-01-01
INJECT ?=
AS_OF ?=
VOLUMES ?=

setup:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff format --check src tests scripts
	uv run ruff check src tests scripts

test:
	uv run pytest -q -m "not integration"

train:  ## train inside the stack so the model lands in the stack's MLflow registry
	docker compose --profile train run --rm trainer

train-local:  ## quick local experiment against sqlite:///mlruns.db
	MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run python scripts/train.py --config configs/default.yaml

eval:  ## evaluate the staging model in the stack; promotes to @production on PASS
	docker compose --profile train run --rm trainer evaluate --model models:/smartbuilding-detector@staging --baseline --promote

eval-local:
	MLFLOW_TRACKING_URI=sqlite:///mlruns.db uv run python scripts/evaluate.py --config configs/default.yaml --baseline

up:
	docker compose up -d --build

down:
	docker compose down $(if $(VOLUMES),-v,)

replay:
	docker compose --profile replay run --rm simulator --start $(START) $(if $(INJECT),--inject $(INJECT),)

retrain:
	docker compose --profile train run --rm trainer $(if $(AS_OF),--as-of $(AS_OF),)

logs:
	docker compose logs -f --tail=50 scorer

weather:  ## fetch the Open-Meteo archive to data/processed
	uv run python -m smartbuilding.data.openmeteo --start 2015-01-01 --end 2021-05-31
