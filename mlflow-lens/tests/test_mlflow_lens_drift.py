import mlflow
import numpy as np
import pandas as pd
from mlflow_lens.drift import classify_drift, compute_psi, log_drift


def test_compute_psi_identical_distributions():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, size=5000)
    psi = compute_psi(data, data)
    assert psi < 0.01


def test_compute_psi_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, size=5000)
    shifted = rng.normal(2, 1, size=5000)
    psi = compute_psi(ref, shifted)
    assert psi > 0.25


def test_classify_drift_thresholds():
    assert classify_drift(0.05) == "low"
    assert classify_drift(0.15) == "moderate"
    assert classify_drift(0.30) == "high"


def test_log_drift_creates_artifact(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    rng = np.random.default_rng(42)

    ref_features = pd.DataFrame(
        {
            "age": rng.normal(40, 10, 1000),
            "income": rng.normal(50000, 15000, 1000),
        }
    )
    ref_preds = rng.random(1000)

    with mlflow.start_run(run_name="reference"):
        mlflow.log_metric("auc", 0.9)
        log_drift(
            reference_run_id="none",
            features=ref_features,
            predictions=ref_preds,
            reference_features=ref_features,
            reference_predictions=ref_preds,
        )
        ref_run_id = mlflow.active_run().info.run_id

    cur_features = pd.DataFrame(
        {
            "age": rng.normal(45, 10, 1000),
            "income": rng.normal(40000, 15000, 1000),
        }
    )
    cur_preds = rng.random(1000) + 0.1

    with mlflow.start_run(run_name="drifted") as run:
        result = log_drift(
            reference_run_id=ref_run_id,
            features=cur_features,
            predictions=cur_preds,
            reference_features=ref_features,
            reference_predictions=ref_preds,
        )

    assert result["reference_run_id"] == ref_run_id
    assert len(result["features"]) == 2
    assert all(f["name"] in ("age", "income") for f in result["features"])
    assert result["prediction_shift"] is not None
    assert "psi" in result["prediction_shift"]

    client = mlflow.tracking.MlflowClient()
    stored = client.get_run(run.info.run_id)
    assert stored.data.tags.get("lens.has_drift") == "true"


def test_log_drift_without_predictions(experiment_id):
    mlflow.set_experiment(experiment_id=experiment_id)
    rng = np.random.default_rng(42)

    ref_features = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur_features = pd.DataFrame({"x": rng.normal(1, 1, 500)})

    with mlflow.start_run():
        result = log_drift(
            reference_run_id="none",
            features=cur_features,
            reference_features=ref_features,
        )

    assert len(result["features"]) == 1
    assert result["prediction_shift"] is None


def test_drift_snapshot_roundtrip(experiment_id):
    """Verify that a run's snapshot can be loaded by a subsequent drift call."""
    mlflow.set_experiment(experiment_id=experiment_id)
    rng = np.random.default_rng(42)

    features_v1 = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 500),
            "b": rng.normal(10, 2, 500),
        }
    )

    with mlflow.start_run(run_name="v1") as run_v1:
        log_drift(
            reference_run_id="bootstrap",
            features=features_v1,
            predictions=rng.random(500),
            reference_features=features_v1,
            reference_predictions=rng.random(500),
        )
        v1_id = run_v1.info.run_id

    features_v2 = pd.DataFrame(
        {
            "a": rng.normal(0.5, 1, 500),
            "b": rng.normal(12, 2, 500),
        }
    )

    with mlflow.start_run(run_name="v2"):
        result = log_drift(
            reference_run_id=v1_id,
            features=features_v2,
        )

    assert len(result["features"]) == 2
    assert all(f["psi"] >= 0 for f in result["features"])
