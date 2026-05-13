from model_landscape.domain.models import MLflowLineage, MonitorConfig
from model_landscape.services.inference_contracts import build_inference_contract as build_contract
from model_landscape.services.onboarding import build_default_baseline


def _build_monitor_config_payload(config: MonitorConfig) -> dict:
    return {
        "model_key": config.model_key,
        "display_name": config.display_name,
        "source_table": config.source_table,
        "timestamp_col": config.contract.timestamp_col,
        "model_id_col": config.contract.model_id_col,
        "model_id_value": config.model_id_value or "",
        "prediction_col": config.contract.prediction_col,
        "model_version_col": config.contract.model_version_col or "",
        "model_version_value": config.model_version_value or "",
        "prediction_score_col": config.contract.prediction_score_col or "",
        "label_col": config.contract.label_col or "",
        "entity_id_col": config.contract.entity_id_col or "",
        "feature_columns": list(config.contract.feature_columns),
        "slice_columns": list(config.contract.slice_columns),
        "categorical_columns": list(config.contract.categorical_columns),
        "baseline_kind": config.baseline.kind,
        "baseline_n_days": config.baseline.n_days,
        "baseline_max_comparison_days": config.baseline.max_comparison_days,
        "baseline_start": config.baseline.baseline_start or "",
        "baseline_end": config.baseline.baseline_end or "",
        "problem_type": config.problem_type,
        "labels_table": config.labels_table or "",
        "labels_join_col": config.labels_join_col or "",
        "labels_order_col": config.labels_order_col or "",
        "mlflow_experiment_name": config.mlflow.experiment_name or "",
        "mlflow_experiment_id": config.mlflow.experiment_id or "",
        "mlflow_run_id": config.mlflow.run_id or "",
        "mlflow_registered_model_name": config.mlflow.registered_model_name or "",
        "mlflow_model_version": config.mlflow.model_version or "",
        "created_by": config.created_by,
        "status": "active",
    }


def test_monitor_payload_keeps_all_features_and_categorical_columns() -> None:
    contract = build_contract(
        columns=["event_ts", "model_id", "prediction"] + [f"f{i}" for i in range(20)],
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        feature_columns=[f"f{i}" for i in range(20)],
        categorical_columns=["f18", "f19"],
    )
    payload = _build_monitor_config_payload(
        MonitorConfig(
            model_key="payments_risk_v1",
            display_name="Payments Risk",
            source_table="catalog.schema.inference_logs",
            contract=contract,
            baseline=build_default_baseline(),
            model_id_value="payments_risk_v1",
            model_version_value="2026-03-01",
            labels_order_col="label_timestamp",
            mlflow=MLflowLineage(
                experiment_name="fraud-monitoring",
                experiment_id="111",
                run_id="run-abc",
                registered_model_name="catalog.schema.fraud_model",
                model_version="12",
            ),
        )
    )
    assert len(payload["feature_columns"]) == 20
    assert payload["categorical_columns"] == ["f18", "f19"]
    assert payload["model_id_value"] == "payments_risk_v1"
    assert payload["model_version_value"] == "2026-03-01"
    assert payload["labels_order_col"] == "label_timestamp"
    assert payload["mlflow_experiment_name"] == "fraud-monitoring"
    assert payload["mlflow_experiment_id"] == "111"
    assert payload["mlflow_run_id"] == "run-abc"
    assert payload["mlflow_registered_model_name"] == "catalog.schema.fraud_model"
    assert payload["mlflow_model_version"] == "12"
    assert payload["baseline_kind"] == "rolling"
    assert payload["baseline_start"] == ""
    assert payload["baseline_end"] == ""
