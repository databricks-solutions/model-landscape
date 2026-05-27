import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow_lens.panels import log_panel


def test_log_panel_confusion_matrix(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    data = [[50, 10], [5, 35]]
    with mlflow.start_run() as run:
        result = log_panel("confusion_matrix", {"matrix": data}, labels=["good", "bad"])

    assert result["type"] == "confusion_matrix"
    assert result["labels"] == ["good", "bad"]
    assert result["data"]["matrix"] == data

    client = mlflow.tracking.MlflowClient()
    path = client.download_artifacts(run.info.run_id, "lens/panels/confusion_matrix.json")
    artifact = json.loads(Path(path).read_text())
    assert artifact["type"] == "confusion_matrix"


def test_log_panel_feature_importance_dataframe(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    df = pd.DataFrame({"feature": ["age", "income"], "importance": [0.6, 0.4]})
    with mlflow.start_run():
        result = log_panel("feature_importance", df, top_n=10)

    assert result["type"] == "feature_importance"
    assert result["top_n"] == 10
    assert isinstance(result["data"], list)
    assert result["data"][0]["feature"] == "age"


def test_log_panel_rejects_unknown_type(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run():
        with pytest.raises(ValueError, match="Unknown panel type"):
            log_panel("not_a_panel", {"x": 1})


def test_log_panel_sets_tags(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run() as run:
        log_panel("roc_curve", [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}])

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run.info.run_id)
    assert run_data.data.tags.get("lens.panel.roc_curve") == "true"
