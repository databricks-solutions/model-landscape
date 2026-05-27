from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

from mlflow_lens.model_selection import cv_scores


def test_cv_scores_returns_bar(cls_dataset):
    X, y, _ = cls_dataset
    fig = cv_scores(LogisticRegression(max_iter=200), X, y, cv=3)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    assert len(fig.data[0].x) == 3


def test_cv_scores_from_scores():
    fig = cv_scores.from_scores([0.9, 0.85, 0.88])
    assert len(fig.data[0].x) == 3


def test_cv_scores_logs_artifacts(experiment_id, cls_dataset):
    mlflow.set_experiment(experiment_id=experiment_id)
    X, y, _ = cls_dataset
    with mlflow.start_run() as run:
        cv_scores(LogisticRegression(max_iter=200), X, y, cv=3, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(
        run.info.run_id, "lens/panels/cv_scores.json"
    )
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "cv_scores"
    assert len(payload["data"]["folds"]) == 3
    assert "mean" in payload["data"]
    client.download_artifacts(run.info.run_id, "lens/panels/cv_scores.html")
