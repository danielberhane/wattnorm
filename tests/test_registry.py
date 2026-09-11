import pandas as pd
import pytest

from smartbuilding.registry import MODEL_NAME, load_detector, log_and_register
from tests.conftest import synthetic


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


def test_log_and_register_creates_a_model_version_and_alias(detector, tracking):
    version = log_and_register(
        detector, params={"n_estimators": 60}, metrics={"coverage": 0.9}, alias="staging"
    )
    assert version.name == MODEL_NAME
    assert int(version.version) >= 1
    again = load_detector(f"models:/{MODEL_NAME}@staging")
    df = synthetic("2018-07-01", 1, seed=20)
    a, _ = detector.score(df)
    b, _ = again.score(df)
    pd.testing.assert_frame_equal(a.drop(columns="model_version"), b.drop(columns="model_version"))
    assert again.model_version == f"{MODEL_NAME}/{version.version}"
