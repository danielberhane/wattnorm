from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from smartbuilding.mqtt import DemandMsg
from smartbuilding.scorer import MemorySink, ScorerCore
from smartbuilding.service import create_app


def _core(detector):
    return ScorerCore(detector, MemorySink(), "ithaca", 3, 0.25, 0.14)


def test_health_endpoint_reports_core_state(detector):
    core = _core(detector)
    core.handle_demand(DemandMsg(meter_id="bldg-a", ts=datetime(2020, 1, 1, tzinfo=UTC), kw=200.0))
    client = TestClient(create_app(core, loader=lambda: detector))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test"
    assert body["meters"]["bldg-a"]["n_scored"] == 1


def test_reload_model_swaps_detector_via_loader(detector):
    import copy

    core = _core(detector)
    newer = copy.copy(detector)
    newer.model_version = "v9"
    client = TestClient(create_app(core, loader=lambda: newer))
    r = client.post("/reload-model")
    assert r.status_code == 200
    assert r.json() == {"previous": "test", "current": "v9"}
    assert client.get("/health").json()["model_version"] == "v9"


def test_reload_model_reports_loader_failure(detector):
    def boom():
        raise RuntimeError("registry down")

    client = TestClient(create_app(_core(detector), loader=boom))
    r = client.post("/reload-model")
    assert r.status_code == 503
    assert "registry down" in r.json()["detail"]


def test_watch_registry_downloads_only_when_alias_moves(detector, monkeypatch):
    import threading
    import time

    from smartbuilding.service import watch_registry

    core = _core(detector)
    loads = []
    ticks = iter(range(3))
    monkeypatch.setattr(time, "sleep", lambda _s: next(ticks))  # StopIteration ends the loop
    monkeypatch.setattr(threading, "Thread", lambda **kw: _Sync(kw["target"]))
    versions = iter([detector.model_version, detector.model_version, "smartbuilding-detector/99"])

    with pytest.raises(StopIteration):
        watch_registry(
            core,
            lambda: loads.append(1) or detector,
            every_s=0,
            current_version=lambda: next(versions),
        )
    assert len(loads) == 1  # two unchanged polls skipped the download; the third fetched


class _Sync:
    """Stand-in for threading.Thread that runs the target inline on start()."""

    def __init__(self, target):
        self.target = target

    def start(self):
        self.target()
