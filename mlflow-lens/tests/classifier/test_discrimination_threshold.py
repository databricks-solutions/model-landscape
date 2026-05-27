from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
import pytest
from mlflow_lens.classifier import discrimination_threshold


def test_discrimination_threshold_returns_figure(binary_data):
    model, X, y = binary_data
    fig = discrimination_threshold(model, X, y, n_thresholds=20)
    assert isinstance(fig, go.Figure)
    # 4 metric lines (precision, recall, f1, queue_rate)
    assert len(fig.data) == 4


def test_discrimination_threshold_rejects_multiclass(multiclass_data):
    model, X, y = multiclass_data
    with pytest.raises(ValueError, match="binary classifiers only"):
        discrimination_threshold(model, X, y)


def test_discrimination_threshold_logs_artifacts(experiment_id, binary_data):
    mlflow.set_experiment(experiment_id=experiment_id)
    model, X, y = binary_data
    with mlflow.start_run() as run:
        discrimination_threshold(model, X, y, n_thresholds=20, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(
        run.info.run_id, "lens/panels/discrimination_threshold.json"
    )
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "discrimination_threshold"
    assert "best_threshold" in payload["data"]
    assert 0.0 <= payload["data"]["best_threshold"] <= 1.0
    client.download_artifacts(run.info.run_id, "lens/panels/discrimination_threshold.html")
