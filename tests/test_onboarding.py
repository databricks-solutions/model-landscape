from model_lens.domain.models import MLflowLineage, MonitorConfig
from model_lens.services.contracts import build_contract
from model_lens.services.onboarding import build_default_baseline, build_monitor_config_payload


def test_monitor_payload_keeps_all_features_and_categorical_columns() -> None:
    contract = build_contract(
        columns=["event_ts", "model_id", "prediction"] + [f"f{i}" for i in range(20)],
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        feature_columns=[f"f{i}" for i in range(20)],
        categorical_columns=["f18", "f19"],
    )
    payload = build_monitor_config_payload(
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
