from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

from mlflow_lens.model_selection import validation_curve


def test_validation_curve_returns_figure(cls_dataset):
    X, y, _ = cls_dataset
    fig = validation_curve(
        RandomForestClassifier(n_estimators=10, random_state=0),
        X,
        y,
        param_name="max_depth",
        param_range=[2, 4, 8],
        cv=3,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4


def test_validation_curve_logs_artifacts(experiment_id, cls_dataset):
    mlflow.set_experiment(experiment_id=experiment_id)
    X, y, _ = cls_dataset
    with mlflow.start_run() as run:
        validation_curve(
            RandomForestClassifier(n_estimators=10, random_state=0),
            X,
            y,
            param_name="max_depth",
            param_range=[2, 4, 8],
            cv=3,
            log=True,
        )

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(
        run.info.run_id, "lens/panels/validation_curve.json"
    )
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "validation_curve"
    assert payload["data"]["param_name"] == "max_depth"
    assert len(payload["data"]["values"]) == 3
    client.download_artifacts(run.info.run_id, "lens/panels/validation_curve.html")
