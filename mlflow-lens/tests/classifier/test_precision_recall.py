from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from mlflow_lens.classifier import precision_recall


def test_precision_recall_returns_figure(binary_data):
    model, X, y = binary_data
    fig = precision_recall(model, X, y)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_precision_recall_multiclass(multiclass_data):
    model, X, y = multiclass_data
    fig = precision_recall(model, X, y)
    assert len(fig.data) == 3


def test_precision_recall_from_scores_parity(binary_data, binary_scores):
    model, X, y = binary_data
    fig_model = precision_recall(model, X, y)
    y_true, y_score = binary_scores
    fig_scores = precision_recall.from_scores(y_true, y_score, classes=list(model.classes_))
    assert fig_model.data[0].name == fig_scores.data[0].name


def test_precision_recall_logs_artifacts(experiment_id, binary_data):
    mlflow.set_experiment(experiment_id=experiment_id)
    model, X, y = binary_data
    with mlflow.start_run() as run:
        precision_recall(model, X, y, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(
        run.info.run_id, "lens/panels/precision_recall_curve.json"
    )
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "precision_recall_curve"
    assert payload["data"]["n_classes"] == 2
    client.download_artifacts(run.info.run_id, "lens/panels/precision_recall_curve.html")
