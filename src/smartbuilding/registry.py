"""MLflow glue: one function to log + register a Detector, one to load it back by URI.

The payload is the plain directory Detector.save() writes, wrapped as a pyfunc model so MLflow
can version it and Grafana/scorer can refer to it as models:/smartbuilding-detector@production.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.entities.model_registry import ModelVersion

from smartbuilding.model import Detector

MODEL_NAME = "smartbuilding-detector"
ARTIFACT_KEY = "detector"


class DetectorPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.detector = Detector.load(context.artifacts[ARTIFACT_KEY])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        scores, _ = self.detector.score(model_input)
        return scores


def log_and_register(
    detector: Detector,
    params: dict,
    metrics: dict,
    alias: str = "staging",
    run_name: str | None = None,
    tags: dict | None = None,
) -> ModelVersion:
    """Log params/metrics + the saved detector as a run, register it, point `alias` at it."""
    client = mlflow.MlflowClient()
    with mlflow.start_run(run_name=run_name) as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})  # drop NaN
        if tags:
            mlflow.set_tags(tags)
        detector.save(Path(tmp) / ARTIFACT_KEY)
        mlflow.pyfunc.log_model(
            name=ARTIFACT_KEY,
            python_model=DetectorPyfunc(),
            artifacts={ARTIFACT_KEY: str(Path(tmp) / ARTIFACT_KEY)},
            registered_model_name=MODEL_NAME,
        )
    versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
    version = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, alias, version.version)
    return version


def load_detector(model_uri: str) -> Detector:
    """Load a Detector from `models:/name@alias`, `models:/name/3`, or a plain local directory."""
    if Path(model_uri).is_dir():
        return Detector.load(model_uri)
    pyfunc = mlflow.pyfunc.load_model(model_uri)
    detector: Detector = pyfunc.unwrap_python_model().detector
    if model_uri.startswith("models:/"):
        detector.model_version = f"{MODEL_NAME}/{_resolve_version(model_uri)}"
    return detector


def _resolve_version(model_uri: str) -> str:
    ref = model_uri.removeprefix("models:/")
    if "@" in ref:
        name, alias = ref.split("@", 1)
        return mlflow.MlflowClient().get_model_version_by_alias(name, alias).version
    return ref.rsplit("/", 1)[-1]
