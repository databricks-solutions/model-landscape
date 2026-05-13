import mlflow

from mlflow_lens.cost import log_cost_context


def test_log_cost_context_local(experiment_id):
    """On local (non-Databricks), should set lens.version but not cluster tags."""
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run() as run:
        tags = log_cost_context()

    assert "lens.version" in tags
    assert "lens.cluster_id" not in tags

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run.info.run_id)
    assert "lens.version" in run_data.data.tags


def test_log_cost_context_databricks(experiment_id, monkeypatch):
    """Simulate Databricks environment to verify cluster tagging."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "16.2")
    monkeypatch.setenv("DB_CLUSTER_ID", "0312-test-cluster")

    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run() as run:
        tags = log_cost_context()

    assert tags["lens.cluster_id"] == "0312-test-cluster"
    assert tags["lens.cost_eligible"] == "true"

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run.info.run_id)
    assert run_data.data.tags["lens.cluster_id"] == "0312-test-cluster"
