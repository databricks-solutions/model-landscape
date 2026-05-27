from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import plotly.graph_objects as go
from mlflow_lens.regressor import alpha_selection


def test_alpha_selection_returns_figure(reg_data, ridge_estimator):
    _, X, y, _, _ = reg_data
    fig = alpha_selection(ridge_estimator, X, y, alphas=np.logspace(-2, 2, 5), cv=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.xaxis.type == "log"


def test_alpha_selection_logs_artifacts(experiment_id, reg_data, ridge_estimator):
    mlflow.set_experiment(experiment_id=experiment_id)
    _, X, y, _, _ = reg_data
    with mlflow.start_run() as run:
        alpha_selection(ridge_estimator, X, y, alphas=np.logspace(-2, 2, 5), cv=3, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/alpha_selection.json")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "alpha_selection"
    assert "best_alpha" in payload["data"]
    assert len(payload["data"]["sweeps"]) == 5
    client.download_artifacts(run.info.run_id, "lens/panels/alpha_selection.html")
