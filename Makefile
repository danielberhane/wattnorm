.PHONY: setup lint test train eval up down replay retrain
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

train:
	uv run python scripts/train.py --config configs/default.yaml

eval:
	uv run python scripts/evaluate.py --config configs/default.yaml --baseline --promote

up:
	docker compose up -d --build

down:
	docker compose down $(if $(VOLUMES),-v,)

replay:
	docker compose --profile replay run --rm simulator --start $(START) $(if $(INJECT),--inject $(INJECT),)

retrain:
	docker compose --profile train run --rm trainer $(if $(AS_OF),--as-of $(AS_OF),)
