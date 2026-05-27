from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from mlflow_lens.classifier import class_prediction_error


def test_class_prediction_error_returns_stacked_bars(multiclass_data):
    model, X, y = multiclass_data
    fig = class_prediction_error(model, X, y)
    assert isinstance(fig, go.Figure)
    assert all(t.type == "bar" for t in fig.data)
    assert len(fig.data) == 3
    assert fig.layout.barmode == "stack"


def test_class_prediction_error_logs_artifacts(experiment_id, binary_data):
    mlflow.set_experiment(experiment_id=experiment_id)
    model, X, y = binary_data
    with mlflow.start_run() as run:
        class_prediction_error(model, X, y, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(
        run.info.run_id, "lens/panels/class_prediction_error.json"
    )
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "class_prediction_error"
    assert isinstance(payload["data"], list)
    assert "true" in payload["data"][0]
    client.download_artifacts(run.info.run_id, "lens/panels/class_prediction_error.html")
