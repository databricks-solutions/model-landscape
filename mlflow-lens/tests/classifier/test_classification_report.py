from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from mlflow_lens.classifier import classification_report


def test_classification_report_returns_figure(multiclass_data):
    model, X, y = multiclass_data
    fig = classification_report(model, X, y)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "heatmap"


def test_classification_report_from_predictions(binary_predictions):
    y_true, y_pred = binary_predictions
    fig = classification_report.from_predictions(y_true, y_pred)
    assert isinstance(fig, go.Figure)


def test_classification_report_logs_artifacts(experiment_id, binary_data):
    mlflow.set_experiment(experiment_id=experiment_id)
    model, X, y = binary_data
    with mlflow.start_run() as run:
        classification_report(model, X, y, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/classification_report.json")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "classification_report"
    assert isinstance(payload["data"], list)
    assert "precision" in payload["data"][0]
    client.download_artifacts(run.info.run_id, "lens/panels/classification_report.html")
