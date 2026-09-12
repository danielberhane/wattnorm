"""The scorer process: MQTT consumer + tiny HTTP surface (/health, /reload-model).

uv run python -m smartbuilding.service
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException

from smartbuilding.config import Env, load_config
from smartbuilding.model import Detector
from smartbuilding.mqtt import (
    DEMAND_WILDCARD,
    WEATHER_WILDCARD,
    DemandMsg,
    make_client,
    parse_message,
)
from smartbuilding.scorer import ScorerCore

log = logging.getLogger("scorer")


def create_app(core: ScorerCore, loader: Callable[[], Detector]) -> FastAPI:
    app = FastAPI(title="smartbuilding scorer")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **core.health()}

    @app.post("/reload-model")
    def reload_model() -> dict:
        previous = core.detector.model_version
        try:
            detector = loader()
        except Exception as e:  # noqa: BLE001 — surface any registry problem to the caller
            raise HTTPException(status_code=503, detail=f"could not load model: {e}") from e
        core.swap_detector(detector)
        log.info("model reloaded: %s → %s", previous, detector.model_version)
        return {"previous": previous, "current": detector.model_version}

    return app


def _retry(fn: Callable, what: str, attempts: int = 60, wait_s: float = 5.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — dependencies come up in any order under compose
            log.warning("%s not ready (%s), retry %d/%d", what, e, i + 1, attempts)
            time.sleep(wait_s)
    raise RuntimeError(f"{what} never became available")


def start_consumer(core: ScorerCore, env: Env) -> None:
    client = make_client(env, client_id="scorer")

    def on_message(_client, _userdata, message) -> None:
        try:
            msg = parse_message(message.topic, message.payload)
            if isinstance(msg, DemandMsg):
                core.handle_demand(msg)
            elif msg is not None:
                core.handle_weather(msg)
        except Exception:  # noqa: BLE001 — one bad message must not kill the consumer
            log.exception("failed to handle message on %s", message.topic)

    def on_connect(client, _userdata, _flags, reason_code, _properties) -> None:
        log.info("connected to broker (%s); subscribing", reason_code)
        client.subscribe([(DEMAND_WILDCARD, 1), (WEATHER_WILDCARD, 1)])

    client.on_message, client.on_connect = on_message, on_connect
    _retry(lambda: client.connect(env.mqtt_host, env.mqtt_port), "mqtt broker")
    client.loop_start()


def watch_registry(
    core: ScorerCore,
    loader: Callable[[], Detector],
    every_s: float,
    current_version: Callable[[], str] | None = None,
) -> None:
    """Background poll: if the production alias moved, hot-swap without a restart.

    `current_version` is a cheap registry lookup (alias → version string); the full model is
    downloaded only when it differs from the loaded one.
    """

    def loop() -> None:
        while True:
            time.sleep(every_s)
            try:
                if current_version and current_version() == core.detector.model_version:
                    continue
                detector = loader()
                if detector.model_version != core.detector.model_version:
                    core.swap_detector(detector)
                    log.info("hot-swapped model → %s", detector.model_version)
            except Exception as e:  # noqa: BLE001
                log.warning("registry poll failed: %s", e)

    threading.Thread(target=loop, daemon=True, name="registry-watch").start()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from smartbuilding.db import PostgresSink
    from smartbuilding.registry import alias_version, load_detector

    cfg, env = load_config(), Env()
    mlflow.set_tracking_uri(env.mlflow_tracking_uri)
    loader = lambda: load_detector(env.model_uri)  # noqa: E731
    detector = _retry(loader, f"model {env.model_uri}")
    sink = _retry(lambda: PostgresSink(env.db_url), "database")
    core = ScorerCore(
        detector,
        sink,
        station_id=cfg.site.station_id,
        weather_stale_hours=cfg.rules.weather_stale_hours,
        emission_factor=env.emission_factor_kg_per_kwh,
        tariff=env.tariff_usd_per_kwh,
    )
    start_consumer(core, env)
    watch_registry(core, loader, every_s=600, current_version=lambda: alias_version(env.model_uri))
    uvicorn.run(create_app(core, loader), host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
