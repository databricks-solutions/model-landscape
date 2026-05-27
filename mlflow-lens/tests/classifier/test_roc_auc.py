from __future__ import annotations

import json
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from mlflow_lens.classifier import roc_auc


def test_roc_auc_returns_figure(binary_data):
    model, X, y = binary_data
    fig = roc_auc(model, X, y)
    assert isinstance(fig, go.Figure)
    # at least 1 ROC trace + the chance diagonal
    assert len(fig.data) >= 2


def test_roc_auc_multiclass_includes_micro(multiclass_data):
    model, X, y = multiclass_data
    fig = roc_auc(model, X, y)
    trace_names = [t.name for t in fig.data]
    assert any("micro-average" in (n or "") for n in trace_names)
    # 3 classes + micro + diagonal (legend hidden) = 5 traces
    assert len(fig.data) == 5


def test_roc_auc_from_scores_parity(binary_data, binary_scores):
    model, X, y = binary_data
    fig_model = roc_auc(model, X, y)
    y_true, y_score = binary_scores
    fig_scores = roc_auc.from_scores(y_true, y_score, classes=list(model.classes_))
    # Same number of traces and matching AUCs in legend
    assert len(fig_model.data) == len(fig_scores.data)
    assert fig_model.data[0].name == fig_scores.data[0].name


def test_roc_auc_logs_artifacts(experiment_id, binary_data):
    mlflow.set_experiment(experiment_id=experiment_id)
    model, X, y = binary_data
    with mlflow.start_run() as run:
        roc_auc(model, X, y, log=True)

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/roc_curve.json")
    html_path = client.download_artifacts(run.info.run_id, "lens/panels/roc_curve.html")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "roc_curve"
    assert payload["data"]["n_classes"] == 2
    assert Path(html_path).read_text().lstrip().startswith("<")

    run_data = client.get_run(run.info.run_id)
    assert run_data.data.tags.get("lens.panel.roc_curve") == "true"
    assert run_data.data.tags.get("lens.figure.roc_curve") == "true"
