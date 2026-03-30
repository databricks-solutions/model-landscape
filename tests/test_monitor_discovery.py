from __future__ import annotations

import pandas as pd

from model_lens.domain.models import MLflowDiscovery, MLflowLineage
from model_lens.services.monitor_discovery import MonitorDiscoveryService


class FakeRepository:
    def __init__(self) -> None:
        self.source_schema = pd.DataFrame(
            [
                {"col_name": "event_ts", "data_type": "timestamp"},
                {"col_name": "model_id", "data_type": "string"},
                {"col_name": "model_version", "data_type": "string"},
                {"col_name": "prediction", "data_type": "double"},
                {"col_name": "entity_id", "data_type": "string"},
                {"col_name": "amount", "data_type": "double"},
                {"col_name": "velocity_7d", "data_type": "double"},
                {"col_name": "region", "data_type": "string"},
            ]
        )
        self.source_preview = pd.DataFrame(
            [
                {
                    "event_ts": "2026-01-01T00:00:00",
                    "model_id": "fraud_model_demo",
                    "model_version": "7",
                    "prediction": 0.91,
                    "entity_id": "ent-1",
                    "amount": 120.0,
                    "velocity_7d": 2.4,
                    "region": "na",
                },
                {
                    "event_ts": "2026-01-02T00:00:00",
                    "model_id": "fraud_model_demo",
                    "model_version": "7",
                    "prediction": 0.13,
                    "entity_id": "ent-2",
                    "amount": 83.0,
                    "velocity_7d": 1.1,
                    "region": "eu",
                },
            ]
        )
        self.labels_schema = pd.DataFrame(
            [
                {"col_name": "entity_id", "data_type": "string"},
                {"col_name": "label", "data_type": "int"},
                {"col_name": "label_timestamp", "data_type": "timestamp"},
            ]
        )
        self.labels_preview = pd.DataFrame([{"entity_id": "ent-1", "label": 1, "label_timestamp": "2026-01-03T00:00:00"}])
        self.distinct_values = {
            ("main.demo.inference_logs", "model_id"): ["fraud_model_demo"],
            ("main.demo.inference_logs", "model_version"): ["7"],
        }

    def scan_source_table(self, table_name: str, preview_rows: int = 5):
        del preview_rows
        if table_name == "main.demo.labels":
            columns = [str(value) for value in self.labels_schema["col_name"].tolist()]
            return columns, self.labels_preview.copy(), self.labels_schema.copy()
        columns = [str(value) for value in self.source_schema["col_name"].tolist()]
        return columns, self.source_preview.copy(), self.source_schema.copy()

    def sample_distinct_values(self, table_name: str, column_name: str, limit: int = 20) -> list[str]:
        del limit
        return list(self.distinct_values.get((table_name, column_name), []))


class FakeMLflow:
    def __init__(self, discovery: MLflowDiscovery) -> None:
        self._discovery = discovery

    def discover(self, **kwargs) -> MLflowDiscovery:
        del kwargs
        return self._discovery


def test_source_only_discovery_builds_numeric_feature_contract() -> None:
    repository = FakeRepository()
    service = MonitorDiscoveryService(repository, mlflow=FakeMLflow(MLflowDiscovery()))

    result = service.discover(source_table="main.demo.inference_logs")

    assert result.config.display_name == "Fraud Model Demo"
    assert result.config.model_key == "fraud_model_demo"
    assert result.config.contract.timestamp_col == "event_ts"
    assert result.config.contract.model_id_col == "model_id"
    assert result.config.contract.prediction_col == "prediction"
    assert result.config.contract.feature_columns == ("amount", "velocity_7d")
    assert result.config.contract.slice_columns == ("region",)
    assert result.config.problem_type == "classification"
    assert result.config.model_id_value == "fraud_model_demo"
    assert result.config.model_version_value == "7"
    assert result.requires_review is False
    assert result.confidence == "high"


def test_discovery_uses_mlflow_and_labels_to_fill_scope_and_lineage() -> None:
    repository = FakeRepository()
    repository.distinct_values[("main.demo.inference_logs", "model_id")] = ["fraud_model_demo", "other_model"]
    repository.distinct_values[("main.demo.inference_logs", "model_version")] = ["7", "8"]
    mlflow = FakeMLflow(
        MLflowDiscovery(
            lineage=MLflowLineage(
                experiment_name="/Users/example/fraud_model_demo",
                experiment_id="exp-1",
                run_id="run-1",
                registered_model_name="fraud_model_demo",
                model_version="7",
            ),
            feature_columns=("velocity_7d", "amount", "non_numeric_feature"),
            problem_type="classification",
        )
    )
    service = MonitorDiscoveryService(repository, mlflow=mlflow)

    result = service.discover(
        source_table="main.demo.inference_logs",
        labels_table="main.demo.labels",
        mlflow_experiment_name="/Users/example/fraud_model_demo",
        mlflow_registered_model_name="fraud_model_demo",
    )

    assert result.config.model_id_value == "fraud_model_demo"
    assert result.config.model_version_value == "7"
    assert result.config.labels_table == "main.demo.labels"
    assert result.config.labels_join_col == "entity_id"
    assert result.config.labels_order_col == "label_timestamp"
    assert result.config.contract.label_col == "label"
    assert result.config.contract.feature_columns == ("velocity_7d", "amount")
    assert result.config.mlflow.experiment_id == "exp-1"
    assert result.config.mlflow.run_id == "run-1"
    assert result.config.mlflow.registered_model_name == "fraud_model_demo"
    assert result.requires_review is False


def test_discovery_marks_multi_model_tables_for_review_without_mlflow_scope_hint() -> None:
    repository = FakeRepository()
    repository.distinct_values[("main.demo.inference_logs", "model_id")] = ["fraud_model_demo", "other_model"]
    repository.distinct_values[("main.demo.inference_logs", "model_version")] = ["7", "8"]
    service = MonitorDiscoveryService(repository, mlflow=FakeMLflow(MLflowDiscovery()))

    result = service.discover(source_table="main.demo.inference_logs")

    assert result.config.model_id_value is None
    assert result.requires_review is True
    assert any("model_id" in warning for warning in result.warnings)
