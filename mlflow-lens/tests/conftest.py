from __future__ import annotations

import uuid

import mlflow
import pytest


@pytest.fixture()
def experiment_id(tmp_path):
    name = f"/tmp/model-landscape-test-{uuid.uuid4().hex[:8]}"
    exp = mlflow.create_experiment(name, artifact_location=str(tmp_path / "artifacts"))
    yield exp
    mlflow.delete_experiment(exp)


@pytest.fixture()
def cache_path(tmp_path):
    return str(tmp_path / "index.parquet")
