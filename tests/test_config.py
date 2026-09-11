from pathlib import Path

from smartbuilding.config import Config, Env, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_config_reads_default_yaml():
    cfg = load_config(ROOT / "configs/default.yaml")
    assert isinstance(cfg, Config)
    assert cfg.site.tz == "America/New_York"
    assert cfg.splits.train == ("2016-01-01", "2017-12-31")
    assert cfg.model.quantiles == [0.05, 0.5, 0.95]
    assert cfg.rules.spike_z == 5.0
    assert cfg.paths.demand_csv == Path("data/raw/demand.csv")


def test_env_reads_runtime_settings_from_environment(monkeypatch):
    monkeypatch.setenv("MQTT_HOST", "broker.local")
    monkeypatch.setenv("EMISSION_FACTOR_KG_PER_KWH", "0.3")
    env = Env()
    assert env.mqtt_host == "broker.local"
    assert env.mqtt_port == 1883
    assert env.emission_factor_kg_per_kwh == 0.3
