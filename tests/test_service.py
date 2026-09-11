from datetime import UTC, datetime

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
