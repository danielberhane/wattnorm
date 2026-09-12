"""Configuration: tunables from configs/*.yaml, runtime hosts/secrets from the environment."""

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths(BaseModel):
    demand_csv: Path
    weather_metric_csv: Path
    weather_imperial_dir: Path
    weather_parquet: Path


class Site(BaseModel):
    meter_id: str
    station_id: str
    lat: float
    lon: float
    tz: str
    demand_tz: str = "UTC"  # clock of the demand export; the EMCS portal exports UTC
    freq: str = "15min"


class Splits(BaseModel):
    train: tuple[str, str]
    calibrate: tuple[str, str]
    test: tuple[str, str]
    replay: tuple[str, str]

    def as_bounds(self) -> dict[str, tuple[str, str]]:
        return {"train": self.train, "calibrate": self.calibrate, "test": self.test}


class ModelParams(BaseModel):
    quantiles: list[float]
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    target_coverage: float
    seed: int = 0


class Rules(BaseModel):
    spike_z: float
    cusum_k: float
    cusum_h: float
    stuck_slots: int
    close_after_in_band_slots: int
    weather_stale_hours: int
    level_halflife_days: float


class Config(BaseModel):
    paths: Paths
    site: Site
    splits: Splits
    model: ModelParams
    rules: Rules


def load_config(path: Path | str = "configs/default.yaml") -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))


class Env(BaseSettings):
    """Runtime environment (.env / container env). Anything host- or secret-like lives here."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_tls: bool = False
    db_url: str = "postgresql://postgres:postgres@localhost:5433/smartbuilding"
    mlflow_tracking_uri: str = (
        "sqlite:///mlruns.db"  # local experiments; the stack sets http://mlflow:5000
    )
    model_uri: str = "models:/smartbuilding-detector@production"
    emission_factor_kg_per_kwh: float = 0.25
    tariff_usd_per_kwh: float = 0.14
