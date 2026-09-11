---
name: verify
description: Run the full local verification for this project (ruff lint+format check, unit tests, and — if the compose stack is up — integration tests). Use after any code change, before claiming work is done, and before committing.
---

# Verify

Run, in order, and stop at the first failure:

1. `uv run ruff format --check src tests scripts` — if it fails, run `uv run ruff format src tests scripts` and re-check.
2. `uv run ruff check src tests scripts`
3. `uv run pytest -q -m "not integration"`
4. If `docker compose ps --status running` lists `timescaledb`, also run `uv run pytest -q -m integration`.

Report the actual output of each step. Never state that tests pass without having seen the pytest summary line in this session.
If a step fails, fix the cause (not the test) and rerun the whole sequence from step 1.
