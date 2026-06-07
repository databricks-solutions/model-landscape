import json

import mlflow
import pandas as pd

from mlflow_lens import summary


def test_log_creates_summary_artifact(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run() as run:
        payload = summary.log(
            task="classification",
            primary_metric="auc",
            score=0.91,
            dataset="credit_v3",
            notes="baseline run",
        )

    assert payload["task"] == "classification"
    assert payload["score"] == 0.91
    assert payload["lens_version"] == "0.2.0"
    assert payload["schema_version"] == "1"

    client = mlflow.tracking.MlflowClient()
    stored = client.get_run(run.info.run_id)
    assert stored.data.tags.get("lens.has_summary") == "true"


def test_build_indexes_runs(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    for i in range(3):
        with mlflow.start_run():
            mlflow.log_metric("acc", 0.8 + i * 0.05)

    index = summary.build([experiment_id])
    assert len(index) == 3
    assert "run_id" in index.columns
    assert "metrics" in index.columns


def test_build_is_incremental(experiment_id, cache_path):
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run():
        mlflow.log_metric("acc", 0.8)

    idx1 = summary.build([experiment_id], cache_path=cache_path)
    summary.cache(idx1, cache_path)
    assert len(idx1) == 1

    with mlflow.start_run():
        mlflow.log_metric("acc", 0.9)

    idx2 = summary.build([experiment_id], cache_path=cache_path)
    assert len(idx2) == 2


def test_cache_and_load_roundtrip(experiment_id, cache_path):
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run():
        mlflow.log_metric("f1", 0.85)

    index = summary.build([experiment_id])
    summary.cache(index, cache_path)
    loaded = summary.load(cache_path)

    assert len(loaded) == 1
    assert loaded.iloc[0]["run_id"] == index.iloc[0]["run_id"]


def test_load_returns_empty_for_missing_path():
    result = summary.load("/nonexistent/path/index.parquet")
    assert isinstance(result, pd.DataFrame)
    assert result.empty
