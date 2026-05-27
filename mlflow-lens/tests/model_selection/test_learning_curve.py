from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import plotly.graph_objects as go
from mlflow_lens.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression


def test_learning_curve_returns_figure(cls_dataset):
    X, y, _ = cls_dataset
    fig = learning_curve(
        LogisticRegression(max_iter=200),
        X,
        y,
        cv=3,
        train_sizes=np.linspace(0.3, 1.0, 3),
    )
    assert isinstance(fig, go.Figure)
    # train line + train band + val line + val band
    assert len(fig.data) == 4


def test_learning_curve_logs_artifacts(experiment_id, cls_dataset):
    mlflow.set_experiment(experiment_id=experiment_id)
    X, y, _ = cls_dataset
    with mlflow.start_run() as run:
        learning_curve(
            LogisticRegression(max_iter=200),
            X,
            y,
            cv=3,
            train_sizes=np.linspace(0.3, 1.0, 3),
            log=True,
        )

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/learning_curve.json")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "learning_curve"
    assert isinstance(payload["data"], list)
    assert "train_score_mean" in payload["data"][0]
    client.download_artifacts(run.info.run_id, "lens/panels/learning_curve.html")
