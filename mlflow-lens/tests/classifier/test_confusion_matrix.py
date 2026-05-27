from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from mlflow_lens.classifier import confusion_matrix


def test_confusion_matrix_returns_figure(binary_data):
    model, X, y = binary_data
    fig = confusion_matrix(model, X, y)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_confusion_matrix_normalize(binary_data):
    model, X, y = binary_data
    fig = confusion_matrix(model, X, y, normalize="true")
    row_sums = [sum(row) for row in fig.data[0].z]
    for s in row_sums:
        assert abs(s - 1.0) < 1e-6


def test_confusion_matrix_from_predictions_parity(binary_data, binary_predictions):
    model, X, y = binary_data
    fig_model = confusion_matrix(model, X, y)
    y_true, y_pred = binary_predictions
    fig_pred = confusion_matrix.from_predictions(y_true, y_pred, labels=list(model.classes_))
    assert list(fig_model.data[0].z[0]) == list(fig_pred.data[0].z[0])


def test_confusion_matrix_logs_artifacts(experiment_id, binary_data):
    mlflow.set_experiment(experiment_id=experiment_id)
    model, X, y = binary_data
    with mlflow.start_run() as run:
        confusion_matrix(model, X, y, log=True, labels=list(model.classes_))

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/confusion_matrix.json")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "confusion_matrix"
    assert "matrix" in payload["data"]
    assert payload["labels"] == list(model.classes_)
    client.download_artifacts(run.info.run_id, "lens/panels/confusion_matrix.html")
