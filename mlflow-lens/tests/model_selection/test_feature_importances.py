from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from mlflow_lens.model_selection import feature_importances


def test_feature_importances_returns_bar(fitted_rf, cls_dataset):
    _, _, names = cls_dataset
    fig = feature_importances(fitted_rf, names)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"


def test_feature_importances_top_n(fitted_rf, cls_dataset):
    _, _, names = cls_dataset
    fig = feature_importances(fitted_rf, names, top_n=5)
    assert len(fig.data[0].x) == 5


def test_feature_importances_from_values():
    fig = feature_importances.from_values(["a", "b", "c"], [0.5, 0.2, 0.1], top_n=2)
    # sorted desc, top 2
    assert list(fig.data[0].y) == ["a", "b"]


def test_feature_importances_logs_artifacts(experiment_id, fitted_rf, cls_dataset):
    mlflow.set_experiment(experiment_id=experiment_id)
    _, _, names = cls_dataset
    with mlflow.start_run() as run:
        feature_importances(fitted_rf, names, top_n=10, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/feature_importance.json")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "feature_importance"
    assert payload["top_n"] == 10
    assert len(payload["data"]) == 10
    client.download_artifacts(run.info.run_id, "lens/panels/feature_importance.html")
