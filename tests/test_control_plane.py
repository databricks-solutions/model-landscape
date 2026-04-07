from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timedelta
from threading import Lock
import time

import pandas as pd

from model_lens.config import settings
from model_lens.domain.models import MLflowLineage, MonitorConfig, RefreshResult
from model_lens.services import refresh_runner as refresh_runner_module
from model_lens.services.contracts import build_contract
from model_lens.services.control_plane import ControlPlaneRepository
from model_lens.services.onboarding import build_default_baseline, build_fixed_baseline
from model_lens.services.refresh_engine import (
    build_daily_feature_profile_rows,
    build_daily_performance_profile_rows,
    build_daily_quality_profile_rows,
)
from model_lens.services.refresh_runner import MonitorRefreshResult, RefreshCounts, run_refresh_cycle
from model_lens.services.spark_refresh import SparkDailyProfiles
from model_lens.services.table_names import TableNames


class FakeWarehouse:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.executed_params: list[tuple[str, tuple]] = []
        self.batch_calls: list[tuple[str, list[tuple]]] = []
        self.queries: list[str] = []
        self.query_param_calls: list[tuple[str, tuple]] = []
        self.distinct_model_ids = 2
        self.duplicate_label_keys = 0
        self.alter_field_already_exists = False
        self.schema_exists = False
        self.existing_tables: set[str] = set()
        self.fail_create_schema = False
        self.fail_create_table = False
        self.comparison_window_rows: list[dict[str, object]] = []
        self.drift_window_rows: list[dict[str, object]] = []
        self.monitor_row = {
            "model_key": "payments_risk_v1",
            "display_name": "Payments Risk",
            "source_table": "catalog.schema.inference_logs",
            "timestamp_col": "event_ts",
            "model_id_col": "model_id",
            "model_id_value": "m1",
            "prediction_col": "prediction",
            "model_version_col": "",
            "model_version_value": "",
            "prediction_score_col": "",
            "label_col": "label",
            "entity_id_col": "entity_id",
            "feature_columns": '["amount","segment"]',
            "slice_columns": '["segment"]',
            "categorical_columns": '["segment"]',
            "baseline_kind": "rolling",
            "baseline_n_days": 7,
            "baseline_start": "",
            "baseline_end": "",
            "baseline_max_comparison_days": 90,
            "problem_type": "classification",
            "labels_table": "",
            "labels_join_col": "",
            "labels_order_col": "",
            "performance_metric_names": '["f1","precision","recall"]',
            "default_performance_metric": "f1",
            "mlflow_experiment_name": "",
            "mlflow_experiment_id": "",
            "mlflow_run_id": "",
            "mlflow_registered_model_name": "",
            "mlflow_model_version": "",
            "created_by": "app",
        }

    def execute(self, sql: str) -> None:
        if "CREATE SCHEMA IF NOT EXISTS" in sql:
            if self.fail_create_schema:
                raise Exception("[PERMISSION_DENIED] Missing CREATE privilege")
            self.schema_exists = True
        if "CREATE TABLE IF NOT EXISTS" in sql:
            if self.fail_create_table:
                raise Exception("[PERMISSION_DENIED] Missing CREATE privilege")
            table_name = sql.split("CREATE TABLE IF NOT EXISTS", 1)[1].split("(", 1)[0].strip()
            self.existing_tables.add(table_name)
        if self.alter_field_already_exists and "ALTER TABLE" in sql and "ADD COLUMNS" in sql:
            raise Exception("[FIELD_ALREADY_EXISTS] Column already exists")
        self.executed.append(sql)

    def execute_params(self, sql: str, params: tuple) -> None:
        self.executed_params.append((sql, params))
        if "INSERT INTO" in sql and "monitor_configs" in sql:
            array_literals = re.findall(r"ARRAY\\(([^)]*)\\)", sql)
            performance_metric_names = '[]'
            if len(array_literals) >= 4:
                values = [
                    token.strip().strip("'")
                    for token in array_literals[3].split(",")
                    if token.strip()
                ]
                performance_metric_names = json.dumps(values)
            self.monitor_row = {
                "model_key": params[0],
                "display_name": params[1],
                "source_table": params[2],
                "timestamp_col": params[3],
                "model_id_col": params[4],
                "model_id_value": params[5],
                "prediction_col": params[6],
                "model_version_col": params[7],
                "model_version_value": params[8],
                "prediction_score_col": params[9],
                "label_col": params[10],
                "entity_id_col": params[11],
                "feature_columns": '["amount","segment"]',
                "slice_columns": '["segment"]',
                "categorical_columns": '["segment"]',
                "baseline_kind": params[12],
                "baseline_n_days": params[13],
                "baseline_start": params[14] or "",
                "baseline_end": params[15] or "",
                "baseline_max_comparison_days": params[16],
                "problem_type": params[17],
                "labels_table": params[18],
                "labels_join_col": params[19],
                "labels_order_col": params[20],
                "performance_metric_names": performance_metric_names,
                "default_performance_metric": params[21],
                "drift_cadence_preset": params[22],
                "performance_cadence_preset": params[23],
                "schedule_enabled": params[24],
                "mlflow_experiment_name": params[25],
                "mlflow_experiment_id": params[26],
                "mlflow_run_id": params[27],
                "mlflow_registered_model_name": params[28],
                "mlflow_model_version": params[29],
                "created_by": params[30],
            }

    def execute_batch(self, insert_template: str, rows: list[tuple], batch_size: int = 200) -> None:
        del batch_size
        self.batch_calls.append((insert_template, rows))

    def query(self, sql: str, cache: bool = False) -> pd.DataFrame:
        del cache
        self.queries.append(sql)
        if sql == "SHOW SCHEMAS IN model_observability LIKE 'control_plane'":
            if self.schema_exists:
                return pd.DataFrame([{"databaseName": "control_plane"}])
            return pd.DataFrame(columns=["databaseName"])
        if sql.startswith("SHOW TABLES IN model_observability.control_plane LIKE '"):
            table_name = sql.split("LIKE '", 1)[1].rsplit("'", 1)[0]
            qualified_name = f"model_observability.control_plane.{table_name}"
            if qualified_name in self.existing_tables:
                return pd.DataFrame([{"tableName": table_name}])
            return pd.DataFrame(columns=["tableName"])
        if "COUNT(*) AS total_rows" in sql and "FROM catalog.schema.inference_logs s" in sql:
            return pd.DataFrame([{
                "total_rows": 21,
                "min_ts": "2026-01-01T00:00:00",
                "max_ts": "2026-01-21T00:00:00",
                "prediction_mean": 0.42,
                "prediction_std": 0.11,
                "label_row_count": 15,
            }])
        if "GROUP BY CAST(s.`event_ts` AS DATE)" in sql:
            return pd.DataFrame([
                {"day_key": "2026-01-20", "row_count": 10},
                {"day_key": "2026-01-21", "row_count": 11},
            ])
        if "ROUND(AVG(CASE WHEN s.`amount` IS NULL" in sql:
            return pd.DataFrame([{"amount": 1.25, "segment": 0.0}])
        if "COUNT(DISTINCT" in sql:
            return pd.DataFrame([{"distinct_model_ids": self.distinct_model_ids}])
        if "duplicate_key_count" in sql:
            return pd.DataFrame([{"duplicate_key_count": self.duplicate_label_keys}])
        if "matched_rows" in sql and "unmatched_rows" in sql:
            return pd.DataFrame([{"inference_rows": 10, "matched_rows": 7, "unmatched_rows": 3}])
        if "AS label_value" in sql:
            return pd.DataFrame([{"label_value": "0"}, {"label_value": "1"}])
        if "AS sampled_value" in sql:
            return pd.DataFrame([{"sampled_value": "m1"}])
        if "WITH latest_window AS" in sql:
            return pd.DataFrame([{
                "model_key": self.monitor_row["model_key"],
                "display_name": self.monitor_row["display_name"],
                "max_psi": 0.34,
                "feature_count": 2,
                "latest_window_end": "2026-01-20",
                "total_rows": 120,
                "latest_data_date": "2026-01-20",
                "last_refresh_at": "2026-01-20T12:00:00+00:00",
                "open_incident_count": 1,
            }])
        if "FROM model_observability.control_plane.incidents" in sql and "WHERE status = 'open'" in sql:
            return pd.DataFrame([{
                "model_key": self.monitor_row["model_key"],
                "feature_name": "amount",
                "metric_name": "psi",
                "severity": "critical",
                "metric_value": 0.34,
                "window_end": "2026-01-20",
                "observed_at": "2026-01-20T12:00:00+00:00",
            }])
        return pd.DataFrame([
            {
                "event_ts": "2026-01-20T00:00:00",
                "model_id": "m1",
                "prediction": 0.9,
                "entity_id": "entity-1",
                "label": 1,
                "amount": 10.0,
            }
        ])

    def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
        self.query_param_calls.append((sql, params))
        if "COUNT(*) AS total_rows" in sql and "FROM catalog.schema.inference_logs s" in sql:
            return pd.DataFrame([{
                "total_rows": 21,
                "min_ts": "2026-01-01T00:00:00",
                "max_ts": "2026-01-21T00:00:00",
                "prediction_mean": 0.42,
                "prediction_std": 0.11,
                "label_row_count": 15,
            }])
        if "GROUP BY CAST(s.`event_ts` AS DATE)" in sql:
            return pd.DataFrame([
                {"day_key": "2026-01-20", "row_count": 10},
                {"day_key": "2026-01-21", "row_count": 11},
            ])
        if "ROUND(AVG(CASE WHEN s.`amount` IS NULL" in sql:
            return pd.DataFrame([{"amount": 1.25, "segment": 0.0}])
        if "FROM model_observability.control_plane.comparison_windows" in sql:
            return pd.DataFrame(self.comparison_window_rows)
        if "FROM model_observability.control_plane.incidents" in sql and "status = 'open'" in sql:
            return pd.DataFrame([{
                "model_key": params[0],
                "feature_name": "amount",
                "metric_name": "psi",
                "severity": "critical",
                "status": "open",
                "metric_value": 0.34,
                "window_end": "2026-01-20",
                "observed_at": "2026-01-20T12:00:00+00:00",
            }])
        if "FROM model_observability.control_plane.drift_metrics" in sql and "SELECT DISTINCT" in sql:
            return pd.DataFrame(self.drift_window_rows)
        if "SELECT * FROM" in sql and "monitor_configs" in sql:
            return pd.DataFrame([self.monitor_row])
        return pd.DataFrame([
            {
                "event_ts": "2026-01-20T00:00:00",
                "model_id": params[0] if params else "m1",
                "prediction": 0.9,
                "entity_id": "entity-1",
                "label": 1,
                "amount": 10.0,
            }
        ])

    def describe_table(self, table_name: str) -> pd.DataFrame:
        del table_name
        return pd.DataFrame([
            {"col_name": "event_ts", "data_type": "timestamp"},
            {"col_name": "model_id", "data_type": "string"},
            {"col_name": "prediction", "data_type": "double"},
            {"col_name": "entity_id", "data_type": "string"},
            {"col_name": "amount", "data_type": "double"},
        ])

    def get_columns(self, table_name: str) -> list[str]:
        if table_name == "catalog.schema.labels":
            return ["entity_id", "label", "label_timestamp"]
        return ["event_ts", "model_id", "prediction", "entity_id", "amount"]


def _monitor_config(
    *,
    with_external_labels: bool = False,
    days: int = 7,
    max_comparison_days: int = 90,
    model_id_value: str | None = "m1",
    labels_order_col: str | None = None,
) -> MonitorConfig:
    contract = build_contract(
        columns=["event_ts", "model_id", "prediction", "entity_id", "label", "amount", "segment"],
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        entity_id_col="entity_id",
        feature_columns=["amount", "segment"],
        categorical_columns=["segment"],
        slice_columns=["segment"],
    )
    return MonitorConfig(
        model_key="payments_risk_v1",
        display_name="Payments Risk",
        source_table="catalog.schema.inference_logs",
        contract=contract,
        baseline=build_default_baseline(days, max_comparison_days=max_comparison_days),
        problem_type="classification",
        model_id_value=model_id_value,
        labels_table="catalog.schema.labels" if with_external_labels else None,
        labels_join_col="entity_id" if with_external_labels else None,
        labels_order_col=labels_order_col if with_external_labels else None,
        mlflow=MLflowLineage(
            experiment_name="fraud_monitoring",
            experiment_id="123",
            run_id="run-123",
            registered_model_name="fraud_model_v1",
            model_version="7",
        ),
        created_by="app",
    )


def test_upsert_monitor_config_keeps_full_feature_and_categorical_metadata() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.upsert_monitor_config(_monitor_config())

    insert_sql, insert_params = warehouse.executed_params[-1]
    assert "ARRAY('amount', 'segment')" in insert_sql
    assert "ARRAY('segment')" in insert_sql
    assert "ARRAY('f1', 'precision', 'recall')" in insert_sql
    assert "CAST(%s AS DATE), CAST(%s AS DATE)" in insert_sql
    assert "VALUES (\n                %s, %s, %s, %s, %s," in insert_sql
    assert insert_params[0] == "payments_risk_v1"
    assert insert_params[1] == "Payments Risk"
    assert insert_params[12] == "rolling"
    assert insert_params[14] is None
    assert insert_params[15] is None
    assert insert_params[21] == "f1"
    assert insert_params[22] == "6h"
    assert insert_params[23] == "disabled"
    assert insert_params[24] is True
    assert insert_params[25] == "fraud_monitoring"
    assert insert_params[29] == "7"


def test_upsert_monitor_config_accepts_hyphenated_feature_names() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))
    contract = build_contract(
        columns=["event_ts", "model_id", "prediction", "entity_id", "label", "amount", "us-central1"],
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        entity_id_col="entity_id",
        feature_columns=["amount", "us-central1"],
        slice_columns=["us-central1"],
    )
    config = replace(_monitor_config(), contract=contract)

    repository.upsert_monitor_config(config)

    insert_sql, _ = warehouse.executed_params[-1]
    assert "ARRAY('amount', 'us-central1')" in insert_sql
    assert "ARRAY('us-central1')" in insert_sql


def test_load_monitor_frame_uses_external_labels_join_and_model_filter() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    frame = repository.load_monitor_frame(_monitor_config(with_external_labels=True, labels_order_col="label_timestamp"))

    assert not frame.empty
    data_query, params = warehouse.query_param_calls[-1]
    assert "LEFT JOIN" in data_query
    assert "ROW_NUMBER() OVER" in data_query
    assert "s.`model_id` = %s" in data_query
    assert "l.`label` AS `label`" in data_query
    assert params == ("m1",)


def test_load_monitor_frame_keeps_source_label_when_no_external_labels_table() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    frame = repository.load_monitor_frame(_monitor_config(with_external_labels=False))

    assert not frame.empty
    assert "label" in frame.columns
    data_query, params = warehouse.query_param_calls[-1]
    assert "LEFT JOIN" not in data_query
    assert "s.`label` AS `label`" in data_query
    assert params == ("m1",)


def test_load_monitor_frame_sampling_caps_rows_per_day_and_total_rows() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    frame = repository.load_monitor_frame(
        _monitor_config(with_external_labels=False),
        start_date="2026-01-01",
        end_date="2026-01-21",
        feature_columns=("amount",),
        sample_rows_per_day=100,
        max_total_rows=500,
    )

    assert not frame.empty
    data_query, params = warehouse.query_param_calls[-1]
    assert "ROW_NUMBER() OVER" in data_query
    assert "model_lens_sample_rank" in data_query
    assert "<= 100" in data_query
    assert "LIMIT 500" in data_query
    assert "CAST(%s AS DATE)" in data_query
    assert params == ("m1", "2026-01-01", "2026-01-21")


def test_load_monitor_frame_uses_shared_labels_join_when_entity_id_is_absent() -> None:
    class _SharedJoinWarehouse(FakeWarehouse):
        def describe_table(self, table_name: str) -> pd.DataFrame:
            del table_name
            return pd.DataFrame([
                {"col_name": "event_ts", "data_type": "timestamp"},
                {"col_name": "model_id", "data_type": "string"},
                {"col_name": "prediction", "data_type": "double"},
                {"col_name": "gc_transaction", "data_type": "string"},
                {"col_name": "amount", "data_type": "double"},
            ])

        def get_columns(self, table_name: str) -> list[str]:
            if table_name == "catalog.schema.labels":
                return ["gc_transaction", "label", "label_timestamp"]
            return ["event_ts", "model_id", "prediction", "gc_transaction", "amount"]

    warehouse = _SharedJoinWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))
    contract = build_contract(
        columns=["event_ts", "model_id", "prediction", "gc_transaction", "label", "amount"],
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        entity_id_col=None,
        feature_columns=["amount"],
    )
    config = MonitorConfig(
        model_key="payments_risk_v1",
        display_name="Payments Risk",
        source_table="catalog.schema.inference_logs",
        contract=contract,
        baseline=build_default_baseline(),
        problem_type="classification",
        model_id_value="m1",
        labels_table="catalog.schema.labels",
        labels_join_col="gc_transaction",
        labels_order_col="label_timestamp",
    )

    frame = repository.load_monitor_frame(config)

    assert not frame.empty
    data_query, params = warehouse.query_param_calls[-1]
    assert "ON s.`gc_transaction` = l.`gc_transaction`" in data_query
    assert params == ("m1",)


def test_scan_source_table_returns_schema_and_preview() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    columns, preview, schema = repository.scan_source_table("catalog.schema.inference_logs")

    assert columns == ["event_ts", "model_id", "prediction", "entity_id", "amount"]
    assert not preview.empty
    assert schema[["col_name", "data_type"]].to_dict("records")[0] == {
        "col_name": "event_ts",
        "data_type": "timestamp",
    }


def test_list_monitor_configs_round_trips_mlflow_lineage() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    config = repository.list_monitor_configs()[0]

    assert config.mlflow.experiment_name is None
    repository.upsert_monitor_config(_monitor_config())
    config = repository.list_monitor_configs()[0]
    assert config.mlflow.experiment_name == "fraud_monitoring"
    assert config.mlflow.experiment_id == "123"
    assert config.mlflow.run_id == "run-123"
    assert config.mlflow.registered_model_name == "fraud_model_v1"
    assert config.mlflow.model_version == "7"


def test_list_monitor_configs_round_trips_fixed_baseline_dates() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.upsert_monitor_config(
        replace(
            _monitor_config(),
            baseline=build_fixed_baseline("2026-01-01", "2026-01-07"),
        )
    )

    config = repository.list_monitor_configs()[0]

    assert config.baseline.kind == "fixed"
    assert config.baseline.n_days == 7
    assert config.baseline.baseline_start == "2026-01-01"
    assert config.baseline.baseline_end == "2026-01-07"


def test_list_monitor_configs_recovers_when_filtered_status_query_returns_empty(caplog) -> None:
    class _FallbackWarehouse(FakeWarehouse):
        def query(self, sql: str, cache: bool = False) -> pd.DataFrame:
            if "SELECT * FROM model_observability.control_plane.monitor_configs ORDER BY updated_at DESC" in sql:
                return pd.DataFrame([self.monitor_row])
            return super().query(sql, cache=cache)

        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "SELECT * FROM model_observability.control_plane.monitor_configs WHERE status = %s" in sql:
                return pd.DataFrame()
            return super().query_params(sql, params)

    warehouse = _FallbackWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    with caplog.at_level("WARNING"):
        configs = repository.list_monitor_configs(status="active")

    assert [config.model_key for config in configs] == ["payments_risk_v1"]
    assert any("recovered 1 configs via unfiltered fallback" in message for message in caplog.messages)


def test_validate_monitor_source_requires_model_id_value_for_shared_tables() -> None:
    warehouse = FakeWarehouse()
    warehouse.distinct_model_ids = 3
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    try:
        repository.validate_monitor_source(_monitor_config(model_id_value=None))
    except ValueError as error:
        assert "Monitored Model ID Value" in str(error)
    else:
        raise AssertionError("expected validation failure")


def test_validate_monitor_source_allows_table_scoped_monitor_without_model_id_column() -> None:
    class _TableScopedWarehouse(FakeWarehouse):
        def describe_table(self, table_name: str) -> pd.DataFrame:
            del table_name
            return pd.DataFrame([
                {"col_name": "event_ts", "data_type": "timestamp"},
                {"col_name": "prediction", "data_type": "double"},
                {"col_name": "label", "data_type": "int"},
                {"col_name": "amount", "data_type": "double"},
            ])

        def get_columns(self, table_name: str) -> list[str]:
            del table_name
            return ["event_ts", "prediction", "label", "amount"]

    warehouse = _TableScopedWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))
    contract = build_contract(
        columns=["event_ts", "prediction", "label", "amount"],
        timestamp_col="event_ts",
        model_id_col=None,
        prediction_col="prediction",
        label_col="label",
        feature_columns=["amount"],
    )
    config = MonitorConfig(
        model_key="payments_risk_v1",
        display_name="Payments Risk",
        source_table="catalog.schema.inference_logs",
        contract=contract,
        baseline=build_default_baseline(),
        problem_type="classification",
        model_id_value=None,
    )

    repository.validate_monitor_source(config)

    assert not any("COUNT(DISTINCT" in sql for sql in warehouse.queries)


def test_validate_monitor_source_requires_label_order_column_when_join_keys_repeat() -> None:
    warehouse = FakeWarehouse()
    warehouse.duplicate_label_keys = 2
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    try:
        repository.validate_monitor_source(_monitor_config(with_external_labels=True, labels_order_col=None))
    except ValueError as error:
        assert "External Labels Order Column" in str(error)
    else:
        raise AssertionError("expected validation failure")


def test_validate_monitor_source_allows_shared_labels_join_without_entity_id_column() -> None:
    class _SharedJoinWarehouse(FakeWarehouse):
        def get_columns(self, table_name: str) -> list[str]:
            if table_name == "catalog.schema.labels":
                return ["gc_transaction", "label"]
            return ["event_ts", "model_id", "prediction", "gc_transaction", "amount"]

    warehouse = _SharedJoinWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))
    contract = build_contract(
        columns=["event_ts", "model_id", "prediction", "gc_transaction", "label", "amount"],
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        entity_id_col=None,
        feature_columns=["amount"],
    )
    config = MonitorConfig(
        model_key="payments_risk_v1",
        display_name="Payments Risk",
        source_table="catalog.schema.inference_logs",
        contract=contract,
        baseline=build_default_baseline(),
        problem_type="classification",
        model_id_value="m1",
        labels_table="catalog.schema.labels",
        labels_join_col="gc_transaction",
    )

    repository.validate_monitor_source(config)


def test_profile_labels_mapping_reports_match_counts_and_binary_values() -> None:
    warehouse = FakeWarehouse()
    warehouse.duplicate_label_keys = 2
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    result = repository.profile_labels_mapping(
        source_table="catalog.schema.inference_logs",
        source_join_col="entity_id",
        labels_table="catalog.schema.labels",
        labels_join_col="entity_id",
        label_col="label",
        labels_order_col="label_timestamp",
    )

    assert result["matched_rows"] == 7
    assert result["unmatched_rows"] == 3
    assert result["duplicate_join_keys"] == 2
    assert result["distinct_label_values"] == ("0", "1")
    assert result["binary_compatible"] is True


def test_get_source_profile_uses_sql_aggregates_instead_of_loading_rows() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    profile = repository.get_source_profile(
        _monitor_config(),
        start_date="2026-01-01",
        end_date="2026-01-21",
    )

    assert profile == {
        "total_rows": 21,
        "min_date": "2026-01-01",
        "max_date": "2026-01-21",
        "prediction_mean": 0.42,
        "prediction_std": 0.11,
        "daily_volume": {"2026-01-20": 10, "2026-01-21": 11},
        "null_rates": {"amount": 1.25, "segment": 0.0},
        "label_row_count": 15,
    }
    assert any("COUNT(*) AS total_rows" in sql for sql, _ in warehouse.query_param_calls)
    assert any("GROUP BY CAST(s.`event_ts` AS DATE)" in sql for sql, _ in warehouse.query_param_calls)
    assert any("ROUND(AVG(CASE WHEN s.`amount` IS NULL" in sql for sql, _ in warehouse.query_param_calls)


def test_ensure_control_plane_ignores_field_already_exists_during_migration() -> None:
    warehouse = FakeWarehouse()
    warehouse.alter_field_already_exists = True
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.ensure_control_plane()

    assert any("CREATE SCHEMA IF NOT EXISTS" in sql for sql in warehouse.executed)
    assert any("CREATE TABLE IF NOT EXISTS" in sql for sql in warehouse.executed)


def test_ensure_control_plane_skips_create_when_schema_and_tables_already_exist() -> None:
    warehouse = FakeWarehouse()
    warehouse.schema_exists = True
    warehouse.existing_tables = {
        "model_observability.control_plane.monitor_configs",
        "model_observability.control_plane.drift_metrics",
        "model_observability.control_plane.quality_metrics",
        "model_observability.control_plane.quality_history",
        "model_observability.control_plane.daily_quality_profiles",
        "model_observability.control_plane.daily_feature_profiles",
        "model_observability.control_plane.performance_metrics",
        "model_observability.control_plane.daily_performance_profiles",
        "model_observability.control_plane.performance_bin_specs",
        "model_observability.control_plane.incidents",
        "model_observability.control_plane.incident_history",
        "model_observability.control_plane.refresh_runs",
        "model_observability.control_plane.comparison_windows",
        "model_observability.control_plane.monitor_runtime_state",
    }
    warehouse.fail_create_schema = True
    warehouse.fail_create_table = True
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.ensure_control_plane()

    assert not any("CREATE SCHEMA IF NOT EXISTS" in sql for sql in warehouse.executed)
    assert not any("CREATE TABLE IF NOT EXISTS" in sql for sql in warehouse.executed)
    assert any(
        "ALTER TABLE model_observability.control_plane.refresh_runs" in sql and "scope" in sql.lower()
        for sql in warehouse.executed
    )


def test_ensure_control_plane_adds_refresh_run_migration_columns() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.ensure_control_plane()

    assert any(
        "ALTER TABLE model_observability.control_plane.refresh_runs" in sql and "scope" in sql.lower()
        for sql in warehouse.executed
    )


def test_get_existing_window_keys_falls_back_to_drift_metrics_when_comparison_windows_are_empty() -> None:
    warehouse = FakeWarehouse()
    warehouse.drift_window_rows = [{
        "baseline_start": "2026-01-01",
        "baseline_end": "2026-01-07",
        "window_start": "2026-01-08",
        "window_end": "2026-01-14",
    }]
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    keys = repository.get_existing_window_keys("payments_risk_v1")

    assert keys == {("2026-01-01", "2026-01-07", "2026-01-08", "2026-01-14")}


def test_get_current_incident_state_reads_open_incidents_for_model() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    state = repository.get_current_incident_state("payments_risk_v1")

    assert state[("payments_risk_v1", "amount", "psi")]["severity"] == "critical"
    assert state[("payments_risk_v1", "amount", "psi")]["status"] == "open"


def test_append_refresh_result_replaces_window_scoped_incident_history_rows() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.append_refresh_result(
        "payments_risk_v1",
        RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[
                {
                    "model_key": "payments_risk_v1",
                    "feature_name": "amount",
                    "metric_name": "psi",
                    "event_type": "opened",
                    "severity": "warning",
                    "status": "open",
                    "metric_value": 0.12,
                    "window_id": "window-1",
                    "window_start": "2026-01-08",
                    "window_end": "2026-01-14",
                    "baseline_start": "2026-01-01",
                    "baseline_end": "2026-01-07",
                    "observed_at": "2026-01-14T00:00:00+00:00",
                }
            ],
            quality_history_rows=[],
            window_rows=[],
        ),
        source_run_id="run-1",
    )

    assert any("DELETE FROM model_observability.control_plane.incident_history" in sql for sql, _ in warehouse.executed_params)
    assert any("INSERT INTO model_observability.control_plane.incident_history" in sql for sql, _ in warehouse.batch_calls)


def test_append_refresh_result_clears_open_incidents_on_recovery_history() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.append_refresh_result(
        "payments_risk_v1",
        RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[
                {
                    "model_key": "payments_risk_v1",
                    "feature_name": "amount",
                    "metric_name": "psi",
                    "event_type": "recovered",
                    "severity": "warning",
                    "status": "closed",
                    "metric_value": 0.0,
                    "window_id": "window-1",
                    "window_start": "2026-01-08",
                    "window_end": "2026-01-14",
                    "baseline_start": "2026-01-01",
                    "baseline_end": "2026-01-07",
                    "observed_at": "2026-01-14T00:00:00+00:00",
                }
            ],
            quality_history_rows=[],
            window_rows=[],
        ),
        source_run_id="run-1",
    )

    assert any("DELETE FROM model_observability.control_plane.incidents" in sql for sql, _ in warehouse.executed_params)


def test_append_refresh_result_replaces_daily_profile_rows_by_profile_date() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.append_refresh_result(
        "payments_risk_v1",
        RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[],
            quality_history_rows=[],
            window_rows=[],
            daily_quality_profile_rows=[
                {
                    "model_key": "payments_risk_v1",
                    "profile_date": "2026-01-20",
                    "row_count": 100,
                    "prediction_mean": 0.42,
                    "prediction_std": 0.11,
                    "null_rates": '{"amount": 0.0}',
                    "label_row_count": 80,
                    "computed_at": "2026-01-20T00:00:00+00:00",
                }
            ],
            daily_feature_profile_rows=[
                {
                    "model_key": "payments_risk_v1",
                    "profile_date": "2026-01-20",
                    "feature_name": "amount",
                    "feature_kind": "numeric",
                    "row_count": 100,
                    "non_null_count": 100,
                    "null_pct": 0.0,
                    "mean": 12.0,
                    "std": 1.2,
                    "min_value": 10.0,
                    "max_value": 14.0,
                    "distribution_json": '{"edges":[10.0,14.0],"counts":[100]}',
                    "computed_at": "2026-01-20T00:00:00+00:00",
                }
            ],
            daily_performance_profile_rows=[
                {
                    "model_key": "payments_risk_v1",
                    "profile_date": "2026-01-20",
                    "feature_name": "amount",
                    "bin_label": "[0, 100)",
                    "metric_name": "f1",
                    "metric_value": 0.84,
                    "row_count": 60,
                    "computed_at": "2026-01-20T00:00:00+00:00",
                }
            ],
        ),
        source_run_id="run-2",
    )

    assert any("DELETE FROM model_observability.control_plane.daily_quality_profiles" in sql for sql, _ in warehouse.executed_params)
    assert any("DELETE FROM model_observability.control_plane.daily_feature_profiles" in sql for sql, _ in warehouse.executed_params)
    assert any("DELETE FROM model_observability.control_plane.daily_performance_profiles" in sql for sql, _ in warehouse.executed_params)
    assert any("INSERT INTO model_observability.control_plane.daily_quality_profiles" in sql for sql, _ in warehouse.batch_calls)
    assert any("INSERT INTO model_observability.control_plane.daily_feature_profiles" in sql for sql, _ in warehouse.batch_calls)
    assert any("INSERT INTO model_observability.control_plane.daily_performance_profiles" in sql for sql, _ in warehouse.batch_calls)


def test_append_refresh_result_rebuilds_quality_summary_from_persisted_daily_profiles() -> None:
    class _QualitySummaryWarehouse(FakeWarehouse):
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM model_observability.control_plane.daily_quality_profiles" in sql:
                return pd.DataFrame([
                    {
                        "model_key": params[0],
                        "profile_date": "2026-01-19",
                        "row_count": 100,
                        "prediction_mean": 0.2,
                        "prediction_std": 0.1,
                        "null_rates": '{"amount": 0.0}',
                        "label_row_count": 90,
                        "computed_at": "2026-01-19T00:00:00+00:00",
                    },
                    {
                        "model_key": params[0],
                        "profile_date": "2026-01-20",
                        "row_count": 50,
                        "prediction_mean": 0.6,
                        "prediction_std": 0.2,
                        "null_rates": '{"amount": 20.0}',
                        "label_row_count": 40,
                        "computed_at": "2026-01-20T00:00:00+00:00",
                    },
                ])
            return super().query_params(sql, params)

    warehouse = _QualitySummaryWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.append_refresh_result(
        "payments_risk_v1",
        RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[],
            quality_history_rows=[],
            window_rows=[],
            daily_quality_profile_rows=[
                {
                    "model_key": "payments_risk_v1",
                    "profile_date": "2026-01-20",
                    "row_count": 50,
                    "prediction_mean": 0.6,
                    "prediction_std": 0.2,
                    "null_rates": '{"amount": 20.0}',
                    "label_row_count": 40,
                    "computed_at": "2026-01-20T00:00:00+00:00",
                }
            ],
        ),
        source_run_id="run-quality",
    )

    quality_insert_rows = next(
        rows
        for sql, rows in warehouse.batch_calls
        if "INSERT INTO model_observability.control_plane.quality_metrics" in sql
    )
    payload = quality_insert_rows[0]
    assert payload[1] == 150
    assert payload[2] == "2026-01-19"
    assert payload[3] == "2026-01-20"
    assert round(float(payload[4]), 4) == 0.3333
    assert json.loads(payload[6]) == {"2026-01-19": 100, "2026-01-20": 50}
    assert json.loads(payload[7]) == {"amount": 6.67}


def test_replace_performance_bin_specs_persists_canonical_edges() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.replace_performance_bin_specs("payments_risk_v1", {"amount": (0.0, 1.5, 3.0)})

    assert any(
        "DELETE FROM model_observability.control_plane.performance_bin_specs" in sql
        for sql, _ in warehouse.executed_params
    )
    insert_rows = next(
        rows
        for sql, rows in warehouse.batch_calls
        if "INSERT INTO model_observability.control_plane.performance_bin_specs" in sql
    )
    assert json.loads(insert_rows[0][2]) == [0.0, 1.5, 3.0]


def test_get_performance_bin_specs_parses_persisted_edges() -> None:
    class _BinSpecWarehouse(FakeWarehouse):
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM model_observability.control_plane.performance_bin_specs" in sql:
                return pd.DataFrame([{
                    "feature_name": "amount",
                    "edges_json": "[0.0, 1.5, 3.0]",
                }])
            return super().query_params(sql, params)

    warehouse = _BinSpecWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    assert repository.get_performance_bin_specs("payments_risk_v1") == {"amount": (0.0, 1.5, 3.0)}


def test_update_refresh_run_metadata_updates_range_and_row_counts() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.update_refresh_run_metadata(
        "run-1",
        data_min_date="2026-01-01",
        data_max_date="2026-01-21",
        range_start="2026-01-15",
        range_end="2026-01-21",
        rows_scanned=123,
        label_rows_scanned=45,
    )

    sql, params = warehouse.executed_params[-1]
    assert "UPDATE model_observability.control_plane.refresh_runs" in sql
    assert params == ("2026-01-01", "2026-01-21", "2026-01-15", "2026-01-21", 123, 45, "run-1")


def test_get_label_watermark_uses_label_signature_for_in_source_labels() -> None:
    class _LabelWatermarkWarehouse(FakeWarehouse):
        def get_columns(self, table_name: str) -> list[str]:
            del table_name
            return ["event_ts", "model_id", "prediction", "entity_id", "label", "amount"]

        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "COUNT(*) AS label_count" in sql:
                return pd.DataFrame([{"watermark": "2026-01-21T00:00:00", "label_count": 5}])
            return super().query_params(sql, params)

    warehouse = _LabelWatermarkWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    watermark = repository.get_label_watermark(
        _monitor_config(with_external_labels=False),
        start_date="2026-01-15",
        end_date="2026-01-21",
    )

    assert watermark == "2026-01-21T00:00:00|5"


def test_archive_monitor_marks_monitor_inactive_without_purging_history() -> None:
    warehouse = FakeWarehouse()
    read_model = FakeReadModel()
    repository = ControlPlaneRepository(
        warehouse=warehouse,
        table_names=TableNames("model_observability", "control_plane"),
        read_model=read_model,
    )

    repository.archive_monitor("payments_risk_v1")

    assert any(
        "UPDATE model_observability.control_plane.monitor_configs" in sql and "status = 'inactive'" in sql
        for sql, _ in warehouse.executed_params
    )
    assert len(read_model.synced) == 1


def test_delete_monitor_purges_all_monitor_scoped_tables() -> None:
    warehouse = FakeWarehouse()
    read_model = FakeReadModel()
    repository = ControlPlaneRepository(
        warehouse=warehouse,
        table_names=TableNames("model_observability", "control_plane"),
        read_model=read_model,
    )

    repository.delete_monitor("payments_risk_v1")

    deleted_tables = {sql.split("DELETE FROM ", 1)[1].split(" WHERE", 1)[0] for sql, _ in warehouse.executed_params}
    assert {
        "model_observability.control_plane.monitor_runtime_state",
        "model_observability.control_plane.refresh_runs",
        "model_observability.control_plane.comparison_windows",
        "model_observability.control_plane.drift_metrics",
        "model_observability.control_plane.quality_metrics",
        "model_observability.control_plane.quality_history",
        "model_observability.control_plane.daily_quality_profiles",
        "model_observability.control_plane.daily_feature_profiles",
        "model_observability.control_plane.performance_metrics",
        "model_observability.control_plane.daily_performance_profiles",
        "model_observability.control_plane.performance_bin_specs",
        "model_observability.control_plane.incidents",
        "model_observability.control_plane.incident_history",
        "model_observability.control_plane.monitor_configs",
    }.issubset(deleted_tables)
    assert len(read_model.synced) == 1


class StubRepository:
    def __init__(self) -> None:
        self.config = _monitor_config(days=7)
        self.replaced: list[str] = []
        self.appended: list[str] = []
        self.started_runs: list[dict[str, str | None]] = []
        self.completed_runs: list[dict[str, str | int]] = []

    def ensure_control_plane(self) -> None:
        return None

    def list_monitor_configs(self, status: str = "active") -> list[MonitorConfig]:
        assert status == "active"
        return [self.config]

    def load_monitor_frame(self, config: MonitorConfig) -> pd.DataFrame:
        assert config.model_key == self.config.model_key
        start = datetime(2026, 1, 1)
        return pd.DataFrame({
            "event_ts": [start + timedelta(days=index) for index in range(5)],
            "model_id": ["m1"] * 5,
            "prediction": [0.1 * index for index in range(5)],
            "label": [0, 1, 0, 1, 0],
            "entity_id": [f"entity-{index}" for index in range(5)],
            "amount": [10.0 + index for index in range(5)],
            "segment": [float(index % 2) for index in range(5)],
        })

    def get_existing_window_keys(self, model_key: str) -> set[tuple[str, str, str, str]]:
        assert model_key == self.config.model_key
        return set()

    def get_current_incident_state(self, model_key: str) -> dict[tuple[str, str, str], dict[str, object]]:
        assert model_key == self.config.model_key
        return {}

    def start_refresh_run(
        self,
        *,
        model_key: str,
        requested_mode: str,
        run_kind: str,
        data_min_date: str | None = None,
        data_max_date: str | None = None,
    ) -> str:
        self.started_runs.append({
            "model_key": model_key,
            "requested_mode": requested_mode,
            "run_kind": run_kind,
            "data_min_date": data_min_date,
            "data_max_date": data_max_date,
        })
        return f"run-{len(self.started_runs)}"

    def complete_refresh_run(self, run_id: str, **kwargs) -> None:
        payload = {"run_id": run_id}
        payload.update(kwargs)
        self.completed_runs.append(payload)

    def replace_all_refresh_results(self, model_key: str, result, source_run_id: str | None = None) -> None:
        del result
        del source_run_id
        self.replaced.append(model_key)

    def append_refresh_result(self, model_key: str, result, source_run_id: str | None = None) -> None:
        del result
        del source_run_id
        self.appended.append(model_key)


def _runtime_state(model_key: str, **overrides: object):
    payload: dict[str, object] = {
        "model_key": model_key,
        "bootstrap_status": "pending",
        "last_drift_refresh_at": None,
        "last_performance_refresh_at": None,
        "next_drift_due_at": None,
        "next_performance_due_at": None,
        "last_label_watermark": None,
        "last_run_status": None,
        "last_run_error": None,
        "last_run_started_at": None,
        "last_run_completed_at": None,
        "backoff_until": None,
        "consecutive_failures": 0,
    }
    payload.update(overrides)
    return type("State", (), payload)()


class BootstrapSelectionSkipRepository(StubRepository):
    def __init__(self) -> None:
        super().__init__()
        self.runtime_states: dict[str, object] = {}
        self.running_bootstraps: set[str] = set()

    def get_monitor_runtime_state(self, model_key: str):
        return self.runtime_states.get(model_key)

    def get_latest_refresh_run(self, model_key: str, scope: str, statuses: tuple[str, ...] | None = None):
        if scope == "bootstrap" and statuses == ("running",) and model_key in self.running_bootstraps:
            return {"run_id": f"running-{model_key}"}
        return None


def test_run_refresh_cycle_skips_replacing_results_when_not_enough_data() -> None:
    repository = StubRepository()

    counts = run_refresh_cycle(repository)

    assert counts.models == 0
    assert counts.drift_rows == 0
    assert counts.quality_rows == 0
    assert counts.performance_rows == 0
    assert counts.incident_rows == 0
    assert repository.replaced == []
    assert repository.appended == []
    assert repository.started_runs[0]["run_kind"] == "backfill"
    assert repository.completed_runs[0]["status"] == "skipped"


def test_run_refresh_cycle_bootstrap_logs_skip_reason_for_backoff(capsys) -> None:
    repository = BootstrapSelectionSkipRepository()
    repository.runtime_states[repository.config.model_key] = _runtime_state(
        repository.config.model_key,
        backoff_until="2026-12-31T00:00:00+00:00",
        last_run_status="failed",
        consecutive_failures=1,
    )

    counts = run_refresh_cycle(repository, mode="auto", scope="bootstrap")

    captured = capsys.readouterr()
    assert counts.models == 0
    assert repository.started_runs == []
    assert "requested_scope=bootstrap" in captured.out
    assert f"model_key={repository.config.model_key}" in captured.out
    assert "reason=recent_failure_backoff" in captured.out
    assert "stage=bootstrap candidates=0 selected=0" in captured.out


class MultiWindowRepository(StubRepository):
    def __init__(self) -> None:
        super().__init__()
        self.existing_window_keys: set[tuple[str, str, str, str]] = set()

    def load_monitor_frame(self, config: MonitorConfig) -> pd.DataFrame:
        assert config.model_key == self.config.model_key
        start = datetime(2026, 1, 1)
        rows: list[dict[str, object]] = []
        for day in range(21):
            regime = 0 if day < 7 else 1 if day < 14 else 2
            for offset in range(2):
                rows.append({
                    "event_ts": start + timedelta(days=day, hours=offset),
                    "model_id": "m1",
                    "prediction": [0.15, 0.35, 0.55, 0.85][regime + offset if regime < 2 else 2 + offset],
                    "label": 0 if regime < 2 else 1,
                    "entity_id": f"entity-{day}-{offset}",
                    "amount": float((day * 2) + offset + (regime * 4)),
                    "segment": float(offset),
                })
        return pd.DataFrame(rows)

    def get_existing_window_keys(self, model_key: str) -> set[tuple[str, str, str, str]]:
        assert model_key == self.config.model_key
        return set(self.existing_window_keys)


class BoundedWindowRepository(MultiWindowRepository):
    def __init__(self) -> None:
        super().__init__()
        self.window_load_calls: list[dict[str, object]] = []

    def get_source_date_range(self, config: MonitorConfig) -> tuple[str | None, str | None]:
        assert config.model_key == self.config.model_key
        return "2026-01-01", "2026-01-21"

    def get_source_profile(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object]:
        assert config.model_key == self.config.model_key
        return {
            "total_rows": 420,
            "min_date": start_date or "2026-01-01",
            "max_date": end_date or "2026-01-21",
            "prediction_mean": 0.42,
            "prediction_std": 0.11,
            "daily_volume": {"2026-01-20": 20, "2026-01-21": 20},
            "null_rates": {"amount": 0.0, "segment": 0.0},
            "label_row_count": 210,
        }

    def load_monitor_frame(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        feature_columns: tuple[str, ...] | None = None,
        sample_rows_per_day: int | None = None,
        max_total_rows: int | None = None,
    ) -> pd.DataFrame:
        self.window_load_calls.append({
            "start_date": start_date,
            "end_date": end_date,
            "feature_columns": feature_columns,
            "sample_rows_per_day": sample_rows_per_day,
            "max_total_rows": max_total_rows,
        })
        frame = super().load_monitor_frame(config)
        if start_date:
            frame = frame[pd.to_datetime(frame["event_ts"]).dt.date >= pd.Timestamp(start_date).date()]
        if end_date:
            frame = frame[pd.to_datetime(frame["event_ts"]).dt.date <= pd.Timestamp(end_date).date()]
        selected = ["event_ts", "model_id", "prediction", "label", "entity_id"]
        selected.extend(list(feature_columns or ()))
        deduped = list(dict.fromkeys(selected))
        return frame[deduped].reset_index(drop=True)


def test_run_refresh_cycle_auto_backfills_when_no_existing_windows() -> None:
    repository = MultiWindowRepository()

    counts = run_refresh_cycle(repository, mode="auto")

    assert counts.models == 1
    assert counts.drift_rows == 8 * 2 * 3
    assert counts.quality_rows == 1
    assert counts.performance_rows > 0
    assert repository.replaced == [repository.config.model_key]
    assert repository.appended == []
    assert repository.started_runs[0]["run_kind"] == "backfill"
    assert repository.completed_runs[0]["status"] == "completed"
    assert repository.completed_runs[0]["window_count"] == 8


def test_run_refresh_cycle_auto_appends_only_new_windows_when_history_exists() -> None:
    repository = MultiWindowRepository()
    repository.existing_window_keys = {
        ("2026-01-01", "2026-01-07", "2026-01-08", "2026-01-14"),
        ("2026-01-02", "2026-01-08", "2026-01-09", "2026-01-15"),
        ("2026-01-03", "2026-01-09", "2026-01-10", "2026-01-16"),
        ("2026-01-04", "2026-01-10", "2026-01-11", "2026-01-17"),
        ("2026-01-05", "2026-01-11", "2026-01-12", "2026-01-18"),
        ("2026-01-06", "2026-01-12", "2026-01-13", "2026-01-19"),
        ("2026-01-07", "2026-01-13", "2026-01-14", "2026-01-20"),
    }

    counts = run_refresh_cycle(repository, mode="auto")

    assert counts.models == 1
    assert counts.drift_rows == 2 * 3
    assert counts.quality_rows == 1
    assert repository.replaced == []
    assert repository.appended == [repository.config.model_key]
    assert repository.started_runs[0]["run_kind"] == "incremental"
    assert repository.completed_runs[0]["status"] == "completed"
    assert repository.completed_runs[0]["window_count"] == 1


def test_run_refresh_cycle_uses_bounded_window_loads_with_sampling_caps() -> None:
    repository = BoundedWindowRepository()

    counts = run_refresh_cycle(repository, mode="auto")

    assert counts.models == 1
    assert repository.window_load_calls
    assert len(repository.window_load_calls) == 1
    assert all(call["start_date"] for call in repository.window_load_calls)
    assert all(call["end_date"] for call in repository.window_load_calls)
    assert all(call["feature_columns"] == repository.config.contract.feature_columns for call in repository.window_load_calls)
    assert all(call["sample_rows_per_day"] is not None for call in repository.window_load_calls)
    assert all(call["max_total_rows"] is not None for call in repository.window_load_calls)


class PerformanceRepairRepository(BoundedWindowRepository):
    def __init__(self) -> None:
        super().__init__()
        self.config = replace(self.config, performance_cadence_preset="daily_7d_repair")

    def get_monitor_runtime_state(self, model_key: str):
        assert model_key == self.config.model_key
        return type("State", (), {
            "model_key": model_key,
            "bootstrap_status": "completed",
            "last_drift_refresh_at": "2026-01-20T00:00:00+00:00",
            "last_performance_refresh_at": "2026-01-20T00:00:00+00:00",
            "next_drift_due_at": "2026-01-20T00:00:00+00:00",
            "next_performance_due_at": "2026-01-20T00:00:00+00:00",
            "last_label_watermark": None,
            "last_run_status": "completed",
            "last_run_error": None,
            "last_run_started_at": "2026-01-20T00:00:00+00:00",
            "last_run_completed_at": "2026-01-20T00:00:00+00:00",
            "backoff_until": None,
            "consecutive_failures": 0,
        })()


def test_run_refresh_cycle_performance_repair_extends_range_for_baseline_lookback() -> None:
    repository = PerformanceRepairRepository()

    counts = run_refresh_cycle(repository, mode="auto", scope="performance_repair")

    assert counts.models == 1
    assert len(repository.window_load_calls) == 1
    assert repository.window_load_calls[0]["start_date"] == "2026-01-02"
    assert repository.window_load_calls[0]["end_date"] == "2026-01-21"


class SparkProfileRepository(BoundedWindowRepository):
    def __init__(self) -> None:
        super().__init__()
        self.spark_profile_calls: list[dict[str, object]] = []
        self.spark_derivation_calls: list[dict[str, object]] = []

    def build_daily_profiles(
        self,
        config: MonitorConfig,
        *,
        start_date: str,
        end_date: str,
        computed_at: str,
        include_drift_quality: bool,
        include_performance: bool,
        existing_bin_specs: dict[str, tuple[float, ...]] | None = None,
    ) -> SparkDailyProfiles:
        self.spark_profile_calls.append({
            "model_key": config.model_key,
            "start_date": start_date,
            "end_date": end_date,
            "computed_at": computed_at,
            "include_drift_quality": include_drift_quality,
            "include_performance": include_performance,
            "existing_bin_specs": existing_bin_specs or {},
        })
        range_frame = super().load_monitor_frame(
            config,
            start_date=start_date,
            end_date=end_date,
            feature_columns=config.contract.feature_columns,
            sample_rows_per_day=settings.refresh_sample_rows_per_day,
            max_total_rows=settings.refresh_max_rows_per_window,
        )
        return SparkDailyProfiles(
            daily_quality_profile_rows=build_daily_quality_profile_rows(
                config=config,
                inference_df=range_frame,
                computed_at=computed_at,
            ),
            daily_feature_profile_rows=build_daily_feature_profile_rows(
                config=config,
                inference_df=range_frame,
                computed_at=computed_at,
            ),
            daily_performance_profile_rows=[],
            performance_bin_specs=existing_bin_specs or {},
        )

    def derive_refresh_result_from_daily_profile_rows(
        self,
        *,
        config: MonitorConfig,
        metadata_list: list[dict[str, str]],
        current_daily_quality_profile_rows: list[dict[str, object]],
        current_daily_feature_profile_rows: list[dict[str, object]],
        current_daily_performance_profile_rows: list[dict[str, object]],
        derivation_start: str,
        derivation_end: str,
        computed_at: str,
        prior_open_incidents: dict[tuple[str, str, str], dict[str, object]] | None = None,
        include_drift_quality: bool = True,
        include_performance: bool = True,
    ) -> RefreshResult:
        del prior_open_incidents
        self.spark_derivation_calls.append({
            "model_key": config.model_key,
            "window_count": len(metadata_list),
            "quality_rows": len(current_daily_quality_profile_rows),
            "feature_rows": len(current_daily_feature_profile_rows),
            "performance_rows": len(current_daily_performance_profile_rows),
            "derivation_start": derivation_start,
            "derivation_end": derivation_end,
            "computed_at": computed_at,
            "include_drift_quality": include_drift_quality,
            "include_performance": include_performance,
        })
        metadata = metadata_list[-1]
        return RefreshResult(
            drift_rows=[{
                "model_key": config.model_key,
                "feature_name": "amount",
                "metric_name": "psi",
                "metric_value": 0.2,
                "window_id": metadata["window_id"],
                "window_start": metadata["window_start"],
                "window_end": metadata["window_end"],
                "baseline_start": metadata["baseline_start"],
                "baseline_end": metadata["baseline_end"],
                "ref_mean": 10.0,
                "cur_mean": 12.0,
                "ref_std": 1.0,
                "cur_std": 1.5,
                "ref_null_pct": 0.0,
                "cur_null_pct": 0.0,
                "ref_count": 10,
                "cur_count": 10,
                "computed_at": computed_at,
            }],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[],
            quality_history_rows=[{
                "model_key": config.model_key,
                "window_id": metadata["window_id"],
                "window_start": metadata["window_start"],
                "window_end": metadata["window_end"],
                "baseline_start": metadata["baseline_start"],
                "baseline_end": metadata["baseline_end"],
                "row_count": 20,
                "prediction_mean": 0.4,
                "prediction_std": 0.1,
                "null_rates": json.dumps({"amount": 0.0, "segment": 0.0}),
                "computed_at": computed_at,
            }],
            window_rows=[{
                "window_id": metadata["window_id"],
                "model_key": config.model_key,
                "window_grain": metadata["window_grain"],
                "window_start": metadata["window_start"],
                "window_end": metadata["window_end"],
                "baseline_start": metadata["baseline_start"],
                "baseline_end": metadata["baseline_end"],
                "baseline_kind": metadata["baseline_kind"],
                "created_at": computed_at,
            }],
            daily_quality_profile_rows=[dict(row) for row in current_daily_quality_profile_rows],
            daily_feature_profile_rows=[dict(row) for row in current_daily_feature_profile_rows],
            daily_performance_profile_rows=[dict(row) for row in current_daily_performance_profile_rows],
        )

    def get_daily_quality_profile_rows(self, *args, **kwargs):
        raise AssertionError("Spark derivation path should not load persisted daily quality rows through pandas")

    def get_daily_feature_profile_rows(self, *args, **kwargs):
        raise AssertionError("Spark derivation path should not load persisted daily feature rows through pandas")

    def get_daily_performance_profile_rows(self, *args, **kwargs):
        raise AssertionError("Spark derivation path should not load persisted daily performance rows through pandas")

    def load_monitor_frame(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        feature_columns: tuple[str, ...] | None = None,
        sample_rows_per_day: int | None = None,
        max_total_rows: int | None = None,
    ) -> pd.DataFrame:
        raise AssertionError("Spark-backed refresh path should not load pandas source frames")


def test_run_refresh_cycle_prefers_repository_spark_profiles_over_pandas_window_loads() -> None:
    repository = SparkProfileRepository()

    counts = run_refresh_cycle(repository, mode="auto")

    assert counts.models == 1
    assert len(repository.spark_profile_calls) == 1
    assert len(repository.spark_derivation_calls) == 1
    assert repository.spark_profile_calls[0]["start_date"] == "2025-10-11"
    assert repository.spark_profile_calls[0]["end_date"] == "2026-01-21"
    assert repository.spark_derivation_calls[0]["derivation_start"] == "2025-10-11"
    assert repository.spark_derivation_calls[0]["derivation_end"] == "2026-01-21"


class LabelSignaturePerformanceRepairRepository(PerformanceRepairRepository):
    def __init__(self) -> None:
        super().__init__()
        self._runtime_state = type("State", (), {
            "model_key": self.config.model_key,
            "bootstrap_status": "completed",
            "last_drift_refresh_at": "2026-01-20T00:00:00+00:00",
            "last_performance_refresh_at": "2026-01-20T00:00:00+00:00",
            "next_drift_due_at": "2026-01-20T00:00:00+00:00",
            "next_performance_due_at": "2026-01-20T00:00:00+00:00",
            "last_label_watermark": "2026-01-21T00:00:00|4",
            "last_run_status": "completed",
            "last_run_error": None,
            "last_run_started_at": "2026-01-20T00:00:00+00:00",
            "last_run_completed_at": "2026-01-20T00:00:00+00:00",
            "backoff_until": None,
            "consecutive_failures": 0,
        })()

    def get_monitor_runtime_state(self, model_key: str):
        assert model_key == self.config.model_key
        return self._runtime_state

    def get_label_watermark(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str | None:
        assert config.model_key == self.config.model_key
        assert start_date is not None
        assert end_date is not None
        return "2026-01-21T00:00:00|5"


def test_run_refresh_cycle_performance_repair_uses_label_signature_not_only_max_timestamp() -> None:
    repository = LabelSignaturePerformanceRepairRepository()

    counts = run_refresh_cycle(repository, mode="auto", scope="performance_repair")

    assert counts.models == 1
    assert repository.completed_runs[-1]["status"] == "completed"


class RefreshMetadataRepository(BoundedWindowRepository):
    def __init__(self) -> None:
        super().__init__()
        self.metadata_updates: list[dict[str, object]] = []

    def update_refresh_run_metadata(self, run_id: str, **kwargs) -> None:
        payload = {"run_id": run_id}
        payload.update(kwargs)
        self.metadata_updates.append(payload)


def test_run_refresh_cycle_updates_refresh_run_metadata_after_profile_discovery() -> None:
    repository = RefreshMetadataRepository()

    counts = run_refresh_cycle(repository, mode="auto")

    assert counts.models == 1
    assert repository.metadata_updates
    update = repository.metadata_updates[0]
    assert update["data_min_date"] == "2026-01-01"
    assert update["data_max_date"] == "2026-01-21"
    assert update["range_start"] == "2025-10-11"
    assert update["range_end"] == "2026-01-21"
    assert update["rows_scanned"] == 420
    assert update["label_rows_scanned"] == 210


class PersistedDailyFactRepository(BoundedWindowRepository):
    def __init__(self) -> None:
        super().__init__()
        self.persisted_quality_calls: list[tuple[str | None, str | None]] = []
        self.persisted_feature_calls: list[tuple[str | None, str | None]] = []
        self.persisted_performance_calls: list[tuple[str | None, str | None]] = []
        self.config = replace(self.config, performance_cadence_preset="daily_7d_repair")
        start = datetime(2026, 1, 8)
        baseline_rows: list[dict[str, object]] = []
        current_rows: list[dict[str, object]] = []
        for day in range(14):
            target = baseline_rows if day < 7 else current_rows
            for offset in range(2):
                target.append({
                    "event_ts": start + timedelta(days=day, hours=offset),
                    "model_id": "m1",
                    "prediction": [0.2, 0.3, 0.7, 0.8][(0 if day < 7 else 2) + offset],
                    "label": 0 if day < 7 else 1,
                    "entity_id": f"entity-{day}-{offset}",
                    "amount": float((day * 3) + offset),
                    "segment": float(offset),
                })
        self._baseline_frame = pd.DataFrame(baseline_rows)
        self._current_frame = pd.DataFrame(current_rows)
        computed_at = "2026-01-22T00:00:00Z"
        self._persisted_quality_rows = build_daily_quality_profile_rows(
            config=self.config,
            inference_df=self._baseline_frame,
            computed_at=computed_at,
        )
        self._persisted_feature_rows = build_daily_feature_profile_rows(
            config=self.config,
            inference_df=self._baseline_frame,
            computed_at=computed_at,
        )
        self._persisted_performance_rows = build_daily_performance_profile_rows(
            config=self.config,
            inference_df=pd.concat([self._baseline_frame, self._current_frame], ignore_index=True),
            computed_at=computed_at,
        )

    def get_monitor_runtime_state(self, model_key: str):
        assert model_key == self.config.model_key
        return type("State", (), {
            "model_key": model_key,
            "bootstrap_status": "completed",
            "last_drift_refresh_at": "2026-01-15T00:00:00+00:00",
            "last_performance_refresh_at": "2026-01-15T00:00:00+00:00",
            "next_drift_due_at": "2026-01-20T00:00:00+00:00",
            "next_performance_due_at": "2026-01-20T00:00:00+00:00",
            "last_label_watermark": None,
            "last_run_status": "completed",
            "last_run_error": None,
            "last_run_started_at": "2026-01-15T00:00:00+00:00",
            "last_run_completed_at": "2026-01-15T00:00:00+00:00",
            "backoff_until": None,
            "consecutive_failures": 0,
        })()

    def get_source_profile(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object]:
        assert config.model_key == self.config.model_key
        return {
            "total_rows": 140,
            "min_date": "2026-01-08",
            "max_date": "2026-01-21",
            "prediction_mean": 0.52,
            "prediction_std": 0.18,
            "daily_volume": {"2026-01-15": 2, "2026-01-21": 2},
            "null_rates": {"amount": 0.0, "segment": 0.0},
            "label_row_count": 14,
        }

    def load_monitor_frame(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        feature_columns: tuple[str, ...] | None = None,
        sample_rows_per_day: int | None = None,
        max_total_rows: int | None = None,
    ) -> pd.DataFrame:
        self.window_load_calls.append({
            "start_date": start_date,
            "end_date": end_date,
            "feature_columns": feature_columns,
            "sample_rows_per_day": sample_rows_per_day,
            "max_total_rows": max_total_rows,
        })
        frame = self._current_frame.copy()
        selected = ["event_ts", "model_id", "prediction", "label", "entity_id"]
        selected.extend(list(feature_columns or ()))
        deduped = list(dict.fromkeys(selected))
        return frame[deduped].reset_index(drop=True)

    def get_daily_quality_profile_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, object]]:
        assert model_key == self.config.model_key
        self.persisted_quality_calls.append((start_date, end_date))
        return list(self._persisted_quality_rows)

    def get_daily_feature_profile_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, object]]:
        assert model_key == self.config.model_key
        self.persisted_feature_calls.append((start_date, end_date))
        return list(self._persisted_feature_rows)

    def get_daily_performance_profile_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, object]]:
        assert model_key == self.config.model_key
        self.persisted_performance_calls.append((start_date, end_date))
        return [
            row
            for row in self._persisted_performance_rows
            if str(row.get("profile_date") or "") <= "2026-01-14"
        ]


def test_run_refresh_cycle_reuses_persisted_daily_profiles_for_incremental_derivation() -> None:
    repository = PersistedDailyFactRepository()

    counts = run_refresh_cycle(repository, mode="auto", scope="drift_quality")

    assert counts.models == 1
    assert counts.drift_rows > 0
    assert len(repository.window_load_calls) == 1
    assert repository.persisted_quality_calls == [("2026-01-08", "2026-01-21")]
    assert repository.persisted_feature_calls == [("2026-01-08", "2026-01-21")]


class ParallelSchedulerRepository(BoundedWindowRepository):
    def __init__(self) -> None:
        super().__init__()
        self.configs = [
            replace(
                self.config,
                model_key="payments_risk_v1",
                model_id_value="m1",
                performance_cadence_preset="daily_7d_repair",
            ),
            replace(
                self.config,
                model_key="payments_risk_v2",
                display_name="Payments Risk 2",
                model_id_value="m2",
                performance_cadence_preset="daily_7d_repair",
            ),
            replace(
                self.config,
                model_key="payments_risk_v3",
                display_name="Payments Risk 3",
                model_id_value="m3",
                performance_cadence_preset="daily_7d_repair",
            ),
        ]
        self.fork_calls = 0
        self.sync_calls = 0
        self._active_start_calls = 0
        self.max_parallel_start_calls = 0
        self._start_lock = Lock()

    def list_monitor_configs(self, status: str = "active") -> list[MonitorConfig]:
        assert status == "active"
        return self.configs

    def get_source_date_range(self, config: MonitorConfig) -> tuple[str | None, str | None]:
        assert any(candidate.model_key == config.model_key for candidate in self.configs)
        return "2026-01-01", "2026-01-21"

    def get_source_profile(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object]:
        assert any(candidate.model_key == config.model_key for candidate in self.configs)
        return {
            "total_rows": 420,
            "min_date": start_date or "2026-01-01",
            "max_date": end_date or "2026-01-21",
            "prediction_mean": 0.42,
            "prediction_std": 0.11,
            "daily_volume": {"2026-01-20": 20, "2026-01-21": 20},
            "null_rates": {"amount": 0.0, "segment": 0.0},
            "label_row_count": 210,
        }

    def load_monitor_frame(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        feature_columns: tuple[str, ...] | None = None,
        sample_rows_per_day: int | None = None,
        max_total_rows: int | None = None,
    ) -> pd.DataFrame:
        self.window_load_calls.append({
            "start_date": start_date,
            "end_date": end_date,
            "feature_columns": feature_columns,
            "sample_rows_per_day": sample_rows_per_day,
            "max_total_rows": max_total_rows,
        })
        start = datetime(2026, 1, 1)
        rows: list[dict[str, object]] = []
        for day in range(21):
            regime = 0 if day < 7 else 1 if day < 14 else 2
            for offset in range(2):
                rows.append({
                    "event_ts": start + timedelta(days=day, hours=offset),
                    "model_id": config.model_id_value or config.model_key,
                    "prediction": [0.15, 0.35, 0.55, 0.85][regime + offset if regime < 2 else 2 + offset],
                    "label": 0 if regime < 2 else 1,
                    "entity_id": f"{config.model_key}-entity-{day}-{offset}",
                    "amount": float((day * 2) + offset + (regime * 4)),
                    "segment": float(offset),
                })
        frame = pd.DataFrame(rows)
        if start_date:
            frame = frame[pd.to_datetime(frame["event_ts"]).dt.date >= pd.Timestamp(start_date).date()]
        if end_date:
            frame = frame[pd.to_datetime(frame["event_ts"]).dt.date <= pd.Timestamp(end_date).date()]
        selected = ["event_ts", "model_id", "prediction", "label", "entity_id"]
        selected.extend(list(feature_columns or ()))
        deduped = list(dict.fromkeys(selected))
        return frame[deduped].reset_index(drop=True)

    def get_monitor_runtime_state(self, model_key: str):
        return type("State", (), {
            "model_key": model_key,
            "bootstrap_status": "completed",
            "last_drift_refresh_at": "2026-01-20T00:00:00+00:00",
            "last_performance_refresh_at": "2026-01-20T00:00:00+00:00",
            "next_drift_due_at": "2026-01-20T00:00:00+00:00",
            "next_performance_due_at": "2026-01-20T00:00:00+00:00",
            "last_label_watermark": None,
            "last_run_status": "completed",
            "last_run_error": None,
        })()

    def get_existing_window_keys(self, model_key: str) -> set[tuple[str, str, str, str]]:
        del model_key
        return set()

    def get_current_incident_state(self, model_key: str) -> dict[tuple[str, str, str], dict[str, object]]:
        assert any(candidate.model_key == model_key for candidate in self.configs)
        return {}

    def start_refresh_run(
        self,
        *,
        model_key: str,
        requested_mode: str,
        run_kind: str,
        scope: str = "bootstrap",
        data_min_date: str | None = None,
        data_max_date: str | None = None,
        **_: object,
    ) -> str:
        with self._start_lock:
            self._active_start_calls += 1
            self.max_parallel_start_calls = max(self.max_parallel_start_calls, self._active_start_calls)
        time.sleep(0.05)
        with self._start_lock:
            self._active_start_calls -= 1
        self.started_runs.append({
            "model_key": model_key,
            "requested_mode": requested_mode,
            "run_kind": run_kind,
            "scope": scope,
            "data_min_date": data_min_date,
            "data_max_date": data_max_date,
        })
        return f"run-{len(self.started_runs)}"

    def fork_for_worker(self):
        self.fork_calls += 1
        return self

    def _sync_read_model(self) -> None:
        self.sync_calls += 1


class SparkSerialSchedulerRepository(ParallelSchedulerRepository):
    def recommended_max_parallel_refresh_workers(self) -> int:
        return 1


class ParallelBootstrapSelectionRepository(ParallelSchedulerRepository):
    def get_monitor_runtime_state(self, model_key: str):
        return _runtime_state(model_key)

    def get_latest_refresh_run(self, model_key: str, scope: str, statuses: tuple[str, ...] | None = None):
        del model_key, scope, statuses
        return None


def test_run_refresh_cycle_scheduler_parallelizes_across_monitors_with_one_scope_each() -> None:
    repository = ParallelSchedulerRepository()
    original = settings.max_parallel_refresh_workers
    object.__setattr__(settings, "max_parallel_refresh_workers", 2)
    try:
        counts = run_refresh_cycle(repository, mode="auto", scope="scheduler")
    finally:
        object.__setattr__(settings, "max_parallel_refresh_workers", original)

    assert counts.models == 3
    assert repository.fork_calls == 3
    assert repository.max_parallel_start_calls == 2
    assert repository.sync_calls == 1
    assert len(repository.started_runs) == 3
    assert {run["model_key"] for run in repository.started_runs} == {
        "payments_risk_v1",
        "payments_risk_v2",
        "payments_risk_v3",
    }
    assert {run["scope"] for run in repository.started_runs} == {"drift_quality"}


def test_run_refresh_cycle_bootstrap_clamps_non_positive_bootstrap_limit(capsys) -> None:
    repository = ParallelBootstrapSelectionRepository()
    original = settings.max_bootstraps_per_run
    object.__setattr__(settings, "max_bootstraps_per_run", 0)
    try:
        counts = run_refresh_cycle(repository, mode="auto", scope="bootstrap")
    finally:
        object.__setattr__(settings, "max_bootstraps_per_run", original)

    captured = capsys.readouterr()
    assert len(repository.started_runs) == 1
    assert counts.models == 1
    assert "reason=non_positive_limit_clamped" in captured.out
    assert "configured_limit=0 effective_limit=1" in captured.out
    assert "reason=max_bootstraps_per_run_reached" in captured.out


def test_run_refresh_cycle_scheduler_isolates_unexpected_worker_failure_in_parallel(monkeypatch) -> None:
    repository = ParallelSchedulerRepository()
    original_worker_cap = settings.max_parallel_refresh_workers
    original_execute_target = refresh_runner_module._execute_target

    def _patched_execute_target(repository_arg, target, *, requested_mode: str):
        del repository_arg
        del requested_mode
        if target.config.model_key == "payments_risk_v2":
            raise RuntimeError("boom")
        return MonitorRefreshResult(
            model_key=target.config.model_key,
            scope=target.scope,
            status="completed",
            counts=RefreshCounts(models=1, drift_rows=1, quality_rows=0, performance_rows=0, incident_rows=0),
        )

    monkeypatch.setattr(refresh_runner_module, "_execute_target", _patched_execute_target)
    object.__setattr__(settings, "max_parallel_refresh_workers", 2)
    try:
        counts = run_refresh_cycle(repository, mode="auto", scope="scheduler")
    finally:
        object.__setattr__(settings, "max_parallel_refresh_workers", original_worker_cap)
        monkeypatch.setattr(refresh_runner_module, "_execute_target", original_execute_target)

    assert counts.models == 2
    assert len(counts.results) == 3
    failure = next(result for result in counts.results if result.model_key == "payments_risk_v2")
    assert failure.status == "failed"
    assert failure.error == "boom"


def test_run_refresh_cycle_scheduler_isolates_unexpected_worker_failure_in_serial(monkeypatch) -> None:
    repository = ParallelSchedulerRepository()
    original_worker_cap = settings.max_parallel_refresh_workers
    original_execute_target = refresh_runner_module._execute_target

    def _patched_execute_target(repository_arg, target, *, requested_mode: str):
        del repository_arg
        del requested_mode
        if target.config.model_key == "payments_risk_v1":
            raise RuntimeError("serial boom")
        return MonitorRefreshResult(
            model_key=target.config.model_key,
            scope=target.scope,
            status="completed",
            counts=RefreshCounts(models=1, drift_rows=1, quality_rows=0, performance_rows=0, incident_rows=0),
        )

    monkeypatch.setattr(refresh_runner_module, "_execute_target", _patched_execute_target)
    object.__setattr__(settings, "max_parallel_refresh_workers", 1)
    try:
        counts = run_refresh_cycle(repository, mode="auto", scope="scheduler")
    finally:
        object.__setattr__(settings, "max_parallel_refresh_workers", original_worker_cap)
        monkeypatch.setattr(refresh_runner_module, "_execute_target", original_execute_target)

    assert counts.models == 2
    assert len(counts.results) == 3
    failure = next(result for result in counts.results if result.model_key == "payments_risk_v1")
    assert failure.status == "failed"


def test_run_refresh_cycle_clamps_parallelism_when_repository_requests_serial_workers() -> None:
    repository = SparkSerialSchedulerRepository()
    original = settings.max_parallel_refresh_workers
    object.__setattr__(settings, "max_parallel_refresh_workers", 3)
    try:
        counts = run_refresh_cycle(repository, mode="auto", scope="scheduler")
    finally:
        object.__setattr__(settings, "max_parallel_refresh_workers", original)

    assert counts.models == 3
    assert repository.fork_calls == 0
    assert repository.max_parallel_start_calls == 1


class FailingSourceRangeRepository(StubRepository):
    def get_source_date_range(self, config: MonitorConfig) -> tuple[str | None, str | None]:
        assert config.model_key == self.config.model_key
        raise ValueError("source lookup failed")


def test_run_refresh_cycle_records_failed_run_when_source_range_lookup_errors() -> None:
    repository = FailingSourceRangeRepository()

    counts = run_refresh_cycle(repository, mode="auto")

    assert counts.models == 0
    assert repository.started_runs
    assert repository.completed_runs[-1]["status"] == "failed"
    assert "source lookup failed" in str(repository.completed_runs[-1]["error_message"])


class StaleRunningRepository(StubRepository):
    def __init__(self) -> None:
        super().__init__()
        self.runtime_state = type("State", (), {
            "model_key": self.config.model_key,
            "bootstrap_status": "completed",
            "last_drift_refresh_at": "2026-01-20T00:00:00+00:00",
            "last_performance_refresh_at": None,
            "next_drift_due_at": "2026-01-20T00:00:00+00:00",
            "next_performance_due_at": None,
            "last_label_watermark": None,
            "last_run_status": "running",
            "last_run_error": None,
            "last_run_started_at": "2026-01-20T00:00:00+00:00",
            "last_run_completed_at": None,
            "backoff_until": None,
            "consecutive_failures": 0,
        })()
        self.stale_runs = [{"run_id": "stale-run-1", "model_key": self.config.model_key}]
        self.upserted_states: list[object] = []

    def get_monitor_runtime_state(self, model_key: str):
        assert model_key == self.config.model_key
        return self.runtime_state

    def upsert_monitor_runtime_state(self, state) -> None:
        self.runtime_state = state
        self.upserted_states.append(state)

    def get_stale_running_refresh_runs(self, started_before: str) -> list[dict[str, object]]:
        assert started_before
        return list(self.stale_runs)

    def get_latest_refresh_run(self, model_key: str, scope: str, statuses: tuple[str, ...] | None = None):
        del model_key, scope, statuses
        return None


def test_run_refresh_cycle_reconciles_stale_running_runs_before_target_selection() -> None:
    repository = StaleRunningRepository()

    counts = run_refresh_cycle(repository, mode="auto")

    assert counts.models == 0
    assert repository.completed_runs[0]["run_id"] == "stale-run-1"
    assert repository.completed_runs[0]["status"] == "failed"
    assert "stale-run timeout" in str(repository.completed_runs[0]["error_message"]).lower()
    assert repository.runtime_state.last_run_status == "failed"
    assert repository.runtime_state.backoff_until is not None


class FakeReadModel:
    def __init__(self) -> None:
        self.configured = True
        self.ensured = 0
        self.synced: list[dict] = []

    def ensure_schema(self) -> None:
        self.ensured += 1

    def replace_dashboard_projection(self, *, configs, summary, incidents) -> None:
        self.ensure_schema()
        self.synced.append({
            "configs": configs,
            "summary": summary,
            "incidents": incidents,
        })

    def get_monitor_summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "model_key": "payments_risk_v1",
            "display_name": "Payments Risk",
            "max_psi": 0.51,
            "feature_count": 2,
            "latest_window_end": "2026-01-21",
            "total_rows": 140,
            "latest_data_date": "2026-01-21",
            "last_refresh_at": "2026-01-21T12:00:00+00:00",
            "open_incident_count": 2,
        }])

    def get_open_incidents(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "model_key": "payments_risk_v1",
            "feature_name": "amount",
            "metric_name": "psi",
            "severity": "critical",
            "metric_value": 0.51,
            "window_end": "2026-01-21",
            "observed_at": "2026-01-21T12:00:00+00:00",
        }])


class EmptyReadModel(FakeReadModel):
    def get_monitor_summary(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_open_incidents(self) -> pd.DataFrame:
        return pd.DataFrame()


def test_upsert_monitor_config_syncs_lakebase_projection() -> None:
    warehouse = FakeWarehouse()
    read_model = FakeReadModel()
    repository = ControlPlaneRepository(
        warehouse=warehouse,
        table_names=TableNames("model_observability", "control_plane"),
        read_model=read_model,
    )

    repository.upsert_monitor_config(_monitor_config())

    assert read_model.ensured >= 1
    assert len(read_model.synced) == 1
    assert read_model.synced[0]["configs"][0].model_key == "payments_risk_v1"
    assert read_model.synced[0]["summary"].iloc[0]["max_psi"] == 0.34


def test_summary_and_incidents_prefer_lakebase_projection_when_available() -> None:
    warehouse = FakeWarehouse()
    read_model = FakeReadModel()
    repository = ControlPlaneRepository(
        warehouse=warehouse,
        table_names=TableNames("model_observability", "control_plane"),
        read_model=read_model,
    )

    summary = repository.get_monitor_summary()
    incidents = repository.get_open_incidents()

    assert summary.iloc[0]["max_psi"] == 0.51
    assert incidents.iloc[0]["severity"] == "critical"


def test_summary_and_incidents_fall_back_to_warehouse_when_lakebase_projection_is_empty() -> None:
    warehouse = FakeWarehouse()
    read_model = EmptyReadModel()
    repository = ControlPlaneRepository(
        warehouse=warehouse,
        table_names=TableNames("model_observability", "control_plane"),
        read_model=read_model,
    )

    summary = repository.get_monitor_summary()
    incidents = repository.get_open_incidents()

    assert summary.iloc[0]["max_psi"] == 0.34
    assert incidents.iloc[0]["severity"] == "critical"
