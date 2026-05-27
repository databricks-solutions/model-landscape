from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from mlflow_lens.regressor import prediction_error


def test_prediction_error_returns_figure(reg_data):
    model, X, y, _, _ = reg_data
    fig = prediction_error(model, X, y)
    assert isinstance(fig, go.Figure)
    # scatter + best-fit + identity
    assert len(fig.data) == 3


def test_prediction_error_from_predictions(reg_predictions):
    y_true, y_pred = reg_predictions
    fig = prediction_error.from_predictions(y_true, y_pred)
    assert isinstance(fig, go.Figure)


def test_prediction_error_logs_artifacts(experiment_id, reg_data):
    mlflow.set_experiment(experiment_id=experiment_id)
    model, X, y, _, _ = reg_data
    with mlflow.start_run() as run:
        prediction_error(model, X, y, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/prediction_error.json")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "prediction_error"
    assert "points" in payload["data"]
    assert "r2" in payload["data"]
    client.download_artifacts(run.info.run_id, "lens/panels/prediction_error.html")
