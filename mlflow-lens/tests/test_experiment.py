import mlflow
from mlflow_lens import experiment


def test_auto_log_context_sets_lens_tags(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run() as run:
        tags = experiment.auto_log_context()

    assert "lens.version" in tags
    assert tags["lens.version"] == "0.1.0"

    client = mlflow.tracking.MlflowClient()
    stored = client.get_run(run.info.run_id)
    assert stored.data.tags["lens.version"] == "0.1.0"


def test_auto_log_context_sets_local_environment(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run():
        tags = experiment.auto_log_context()

    assert tags.get("lens.environment") == "local"


def test_log_flattens_nested_dict(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run() as run:
        flat = experiment.log(
            {
                "data": {"source": "credit_v3", "rows": 50000},
                "model": {"type": "xgboost"},
            }
        )

    assert flat == {
        "data.source": "credit_v3",
        "data.rows": "50000",
        "model.type": "xgboost",
    }

    client = mlflow.tracking.MlflowClient()
    stored = client.get_run(run.info.run_id)
    assert stored.data.params["data.source"] == "credit_v3"
    assert stored.data.params["data.rows"] == "50000"


def test_log_with_prefix(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run():
        flat = experiment.log({"scaler": "standard"}, prefix="preprocessing")

    assert flat == {"preprocessing.scaler": "standard"}
