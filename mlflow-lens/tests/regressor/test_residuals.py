from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go

from mlflow_lens.regressor import residuals


def test_residuals_test_only(reg_data):
    model, X, y, _, _ = reg_data
    fig = residuals(model, X, y)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_residuals_with_train_overlay(reg_data):
    model, X_te, y_te, X_tr, y_tr = reg_data
    fig = residuals(model, X_te, y_te, X_train=X_tr, y_train=y_tr)
    assert len(fig.data) == 2


def test_residuals_logs_artifacts(experiment_id, reg_data):
    mlflow.set_experiment(experiment_id=experiment_id)
    model, X_te, y_te, X_tr, y_tr = reg_data
    with mlflow.start_run() as run:
        residuals(model, X_te, y_te, X_train=X_tr, y_train=y_tr, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/residuals.json")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "residuals"
    assert "test" in payload["data"]
    assert "train" in payload["data"]
    client.download_artifacts(run.info.run_id, "lens/panels/residuals.html")
