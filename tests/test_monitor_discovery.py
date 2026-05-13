from __future__ import annotations

import pandas as pd

from model_landscape.domain.models import MLflowDiscovery, MLflowLineage
from model_landscape.services.monitor_discovery import MonitorDiscoveryService


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
        self.bounded_samples = {
            "main.demo.inference_logs": pd.DataFrame(
                [
                    {"model_id": "fraud_model_demo", "model_version": "7"},
                    {"model_id": "fraud_model_demo", "model_version": "7"},
                ]
            )
        }
        self.labels_validation = {
            "matched_rows": 1,
            "unmatched_rows": 1,
            "duplicate_join_keys": 1,
            "match_rate_pct": 50.0,
            "distinct_label_values": ("0", "1"),
            "binary_compatible": True,
        }

    def scan_source_table(self, table_name: str, preview_rows: int = 5):
        del preview_rows
        if table_name == "main.demo.labels":
            columns = [str(value) for value in self.labels_schema["col_name"].tolist()]
            return columns, self.labels_preview.copy(), self.labels_schema.copy()
        columns = [str(value) for value in self.source_schema["col_name"].tolist()]
        return columns, self.source_preview.copy(), self.source_schema.copy()

    def sample_bounded_rows(self, table_name: str, columns: list[str] | tuple[str, ...], *, max_total_rows: int = 2000):
        del max_total_rows
        frame = self.bounded_samples.get(table_name, pd.DataFrame())
        selected = [column for column in columns if column in frame.columns]
        return frame[selected].copy() if selected else pd.DataFrame(columns=list(columns))

    def profile_labels_mapping(
        self,
        *,
        source_table: str,
        source_join_col: str,
        labels_table: str,
        labels_join_col: str,
        label_col: str,
        labels_order_col: str | None = None,
    ) -> dict:
        assert source_table == "main.demo.inference_logs"
        assert source_join_col == "entity_id"
        assert labels_table == "main.demo.labels"
        assert labels_join_col == "entity_id"
        assert label_col == "label"
        assert labels_order_col == "label_timestamp"
        return dict(self.labels_validation)


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


def test_discovery_prefers_source_label_when_present_in_inference_table() -> None:
    class _SourceLabelRepository(FakeRepository):
        def __init__(self) -> None:
            super().__init__()
            self.source_schema = pd.DataFrame(
                [
                    {"col_name": "event_ts", "data_type": "timestamp"},
                    {"col_name": "model_id", "data_type": "string"},
                    {"col_name": "prediction", "data_type": "double"},
                    {"col_name": "label", "data_type": "int"},
                    {"col_name": "amount", "data_type": "double"},
                    {"col_name": "velocity_7d", "data_type": "double"},
                ]
            )
            self.source_preview = pd.DataFrame(
                [
                    {
                        "event_ts": "2026-01-01T00:00:00",
                        "model_id": "fraud_model_demo",
                        "prediction": 0.91,
                        "label": 1,
                        "amount": 120.0,
                        "velocity_7d": 2.4,
                    },
                    {
                        "event_ts": "2026-01-02T00:00:00",
                        "model_id": "fraud_model_demo",
                        "prediction": 0.13,
                        "label": 0,
                        "amount": 83.0,
                        "velocity_7d": 1.1,
                    },
                ]
            )

    repository = _SourceLabelRepository()
    service = MonitorDiscoveryService(repository, mlflow=FakeMLflow(MLflowDiscovery()))

    result = service.discover(source_table="main.demo.inference_logs")

    assert result.config.labels_table is None
    assert result.config.contract.label_col == "label"
    assert result.requires_review is False


def test_discovery_supports_table_scoped_monitor_without_model_id_column() -> None:
    class _TableScopedRepository(FakeRepository):
        def __init__(self) -> None:
            super().__init__()
            self.source_schema = pd.DataFrame(
                [
                    {"col_name": "event_ts", "data_type": "timestamp"},
                    {"col_name": "prediction", "data_type": "double"},
                    {"col_name": "label", "data_type": "int"},
                    {"col_name": "amount", "data_type": "double"},
                    {"col_name": "velocity_7d", "data_type": "double"},
                    {"col_name": "region", "data_type": "string"},
                ]
            )
            self.source_preview = pd.DataFrame(
                [
                    {
                        "event_ts": "2026-01-01T00:00:00",
                        "prediction": 0.91,
                        "label": 1,
                        "amount": 120.0,
                        "velocity_7d": 2.4,
                        "region": "na",
                    },
                    {
                        "event_ts": "2026-01-02T00:00:00",
                        "prediction": 0.13,
                        "label": 0,
                        "amount": 83.0,
                        "velocity_7d": 1.1,
                        "region": "eu",
                    },
                ]
            )

    repository = _TableScopedRepository()
    service = MonitorDiscoveryService(repository, mlflow=FakeMLflow(MLflowDiscovery()))

    result = service.discover(source_table="main.demo.inference_logs")

    assert result.config.contract.model_id_col is None
    assert result.config.model_id_value is None
    assert result.config.contract.label_col == "label"
    assert result.requires_review is False
    assert result.confidence == "high"


def test_discovery_uses_mlflow_and_labels_to_fill_scope_and_lineage() -> None:
    repository = FakeRepository()
    repository.bounded_samples["main.demo.inference_logs"] = pd.DataFrame(
        [
            {"model_id": "fraud_model_demo", "model_version": "7"},
            {"model_id": "other_model", "model_version": "8"},
        ]
    )
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
    assert result.label_schema_rows[0]["col_name"] == "entity_id"
    assert result.label_preview_rows[0]["label"] == "1"
    assert result.label_validation["matched_rows"] == 1
    assert result.label_validation["duplicate_join_keys"] == 1
    assert result.config.contract.feature_columns == ("velocity_7d", "amount")
    assert result.config.mlflow.experiment_id == "exp-1"
    assert result.config.mlflow.run_id == "run-1"
    assert result.config.mlflow.registered_model_name == "fraud_model_demo"
    assert result.requires_review is False


def test_discovery_marks_multi_model_tables_for_review_without_mlflow_scope_hint() -> None:
    repository = FakeRepository()
    repository.bounded_samples["main.demo.inference_logs"] = pd.DataFrame(
        [
            {"model_id": "fraud_model_demo", "model_version": "7"},
            {"model_id": "other_model", "model_version": "8"},
        ]
    )
    service = MonitorDiscoveryService(repository, mlflow=FakeMLflow(MLflowDiscovery()))

    result = service.discover(source_table="main.demo.inference_logs")

    assert result.config.model_id_value is None
    assert result.requires_review is True
    assert any("single monitored model ID" in warning for warning in result.warnings)


def test_discovery_prefers_shared_string_join_key_and_string_timestamps() -> None:
    class _GeoComplyRepository(FakeRepository):
        def __init__(self) -> None:
            super().__init__()
            self.source_schema = pd.DataFrame(
                [
                    {"col_name": "event_time", "data_type": "string"},
                    {"col_name": "model_version", "data_type": "string"},
                    {"col_name": "prediction", "data_type": "double"},
                    {"col_name": "unique_hash", "data_type": "string"},
                    {"col_name": "anti_spoof_debug_process_id", "data_type": "bigint"},
                    {"col_name": "velocity_7d", "data_type": "double"},
                    {"col_name": "amount", "data_type": "double"},
                ]
            )
            self.source_preview = pd.DataFrame(
                [
                    {
                        "event_time": "2026-01-17T19:47:22.366Z",
                        "model_version": "gc_prod_aiguardian.ios.ali_ios@5",
                        "prediction": 0.91,
                        "unique_hash": "hash-1",
                        "anti_spoof_debug_process_id": 101,
                        "velocity_7d": 2.4,
                        "amount": 120.0,
                    },
                    {
                        "event_time": "2026-01-17T19:48:22.366Z",
                        "model_version": "gc_prod_aiguardian.ios.ali_ios@5",
                        "prediction": 0.13,
                        "unique_hash": "hash-2",
                        "anti_spoof_debug_process_id": 202,
                        "velocity_7d": 1.1,
                        "amount": 83.0,
                    },
                ]
            )
            self.labels_schema = pd.DataFrame(
                [
                    {"col_name": "unique_hash", "data_type": "string"},
                    {"col_name": "anti_spoof_debug_process_id", "data_type": "bigint"},
                    {"col_name": "label", "data_type": "int"},
                    {"col_name": "label_timestamp", "data_type": "string"},
                ]
            )
            self.labels_preview = pd.DataFrame(
                [
                    {
                        "unique_hash": "hash-9",
                        "anti_spoof_debug_process_id": 101,
                        "label": 1,
                        "label_timestamp": "2026-01-17T20:00:00.000Z",
                    }
                ]
            )
            self.bounded_samples["main.demo.inference_logs"] = pd.DataFrame(
                [{"model_version": "gc_prod_aiguardian.ios.ali_ios@5"}]
            )
            self.labels_validation = {
                "inference_rows": 2,
                "matched_rows": 0,
                "unmatched_rows": 2,
                "duplicate_join_keys": 0,
                "match_rate_pct": 0.0,
                "distinct_label_values": ("0", "1"),
                "binary_compatible": True,
            }

        def profile_labels_mapping(self, **kwargs) -> dict:
            assert kwargs["source_join_col"] == "unique_hash"
            assert kwargs["labels_join_col"] == "unique_hash"
            assert kwargs["labels_order_col"] == "label_timestamp"
            return dict(self.labels_validation)

    repository = _GeoComplyRepository()
    service = MonitorDiscoveryService(repository, mlflow=FakeMLflow(MLflowDiscovery()))

    result = service.discover(
        source_table="main.demo.inference_logs",
        labels_table="main.demo.labels",
    )

    assert result.config.contract.timestamp_col == "event_time"
    assert result.config.contract.model_id_col == "model_version"
    assert result.config.contract.model_version_col is None
    assert result.config.model_id_value == "gc_prod_aiguardian.ios.ali_ios@5"
    assert result.config.labels_join_col == "unique_hash"
    assert result.config.labels_order_col == "label_timestamp"
    assert result.label_validation["matched_rows"] == 0
    assert result.requires_review is True
    assert any("No rows matched between inference and labels tables" in warning for warning in result.warnings)


def test_discovery_keeps_all_numeric_features_for_wide_schemas() -> None:
    class _WideRepository(FakeRepository):
        def __init__(self) -> None:
            super().__init__()
            feature_schema = [{"col_name": f"feature_{index}", "data_type": "double"} for index in range(75)]
            self.source_schema = pd.DataFrame(
                [
                    {"col_name": "event_ts", "data_type": "timestamp"},
                    {"col_name": "model_id", "data_type": "string"},
                    {"col_name": "prediction", "data_type": "double"},
                    *feature_schema,
                ]
            )
            self.source_preview = pd.DataFrame(
                [
                    {
                        "event_ts": "2026-01-01T00:00:00",
                        "model_id": "fraud_model_demo",
                        "prediction": 0.91,
                        **{f"feature_{index}": float(index) for index in range(75)},
                    }
                ]
            )
            self.bounded_samples["main.demo.inference_logs"] = pd.DataFrame(
                [{"model_id": "fraud_model_demo"}]
            )

    repository = _WideRepository()
    service = MonitorDiscoveryService(repository, mlflow=FakeMLflow(MLflowDiscovery()))

    result = service.discover(source_table="main.demo.inference_logs")

    assert len(result.config.contract.feature_columns) == 75
    assert result.requires_review is False


def test_discovery_uses_shared_labels_join_when_entity_id_is_absent() -> None:
    class _SharedJoinRepository(FakeRepository):
        def __init__(self) -> None:
            super().__init__()
            self.source_schema = pd.DataFrame(
                [
                    {"col_name": "event_ts", "data_type": "timestamp"},
                    {"col_name": "model_id", "data_type": "string"},
                    {"col_name": "prediction", "data_type": "double"},
                    {"col_name": "gc_transaction", "data_type": "string"},
                    {"col_name": "amount", "data_type": "double"},
                ]
            )
            self.source_preview = pd.DataFrame(
                [
                    {
                        "event_ts": "2026-01-01T00:00:00",
                        "model_id": "fraud_model_demo",
                        "prediction": 0.91,
                        "gc_transaction": "tx-1",
                        "amount": 120.0,
                    }
                ]
            )
            self.labels_schema = pd.DataFrame(
                [
                    {"col_name": "gc_transaction", "data_type": "string"},
                    {"col_name": "label", "data_type": "int"},
                ]
            )
            self.labels_preview = pd.DataFrame([{"gc_transaction": "tx-1", "label": 1}])
            self.labels_validation = {
                "inference_rows": 1,
                "matched_rows": 1,
                "unmatched_rows": 0,
                "duplicate_join_keys": 0,
                "match_rate_pct": 100.0,
                "distinct_label_values": ("0", "1"),
                "binary_compatible": True,
            }

        def profile_labels_mapping(self, **kwargs) -> dict:
            assert kwargs["source_join_col"] == "gc_transaction"
            assert kwargs["labels_join_col"] == "gc_transaction"
            return dict(self.labels_validation)

    repository = _SharedJoinRepository()
    service = MonitorDiscoveryService(repository, mlflow=FakeMLflow(MLflowDiscovery()))

    result = service.discover(
        source_table="main.demo.inference_logs",
        labels_table="main.demo.labels",
    )

    assert result.config.labels_join_col == "gc_transaction"
    assert result.label_validation["matched_rows"] == 1
    assert result.requires_review is False
