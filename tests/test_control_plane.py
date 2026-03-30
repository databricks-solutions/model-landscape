from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd

from model_lens.domain.models import MLflowLineage, MonitorConfig, RefreshResult
from model_lens.services.contracts import build_contract
from model_lens.services.control_plane import ControlPlaneRepository
from model_lens.services.onboarding import build_default_baseline, build_fixed_baseline
from model_lens.services.refresh_runner import run_refresh_cycle
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
            "mlflow_experiment_name": "",
            "mlflow_experiment_id": "",
            "mlflow_run_id": "",
            "mlflow_registered_model_name": "",
            "mlflow_model_version": "",
            "created_by": "app",
        }

    def execute(self, sql: str) -> None:
        if self.alter_field_already_exists and "ALTER TABLE" in sql and "ADD COLUMNS" in sql:
            raise Exception("[FIELD_ALREADY_EXISTS] Column already exists")
        self.executed.append(sql)

    def execute_params(self, sql: str, params: tuple) -> None:
        self.executed_params.append((sql, params))
        if "INSERT INTO" in sql and "monitor_configs" in sql:
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
                "mlflow_experiment_name": params[21],
                "mlflow_experiment_id": params[22],
                "mlflow_run_id": params[23],
                "mlflow_registered_model_name": params[24],
                "mlflow_model_version": params[25],
                "created_by": params[26],
            }

    def execute_batch(self, insert_template: str, rows: list[tuple], batch_size: int = 200) -> None:
        del batch_size
        self.batch_calls.append((insert_template, rows))

    def query(self, sql: str, cache: bool = False) -> pd.DataFrame:
        del cache
        self.queries.append(sql)
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
    assert "CAST(%s AS DATE), CAST(%s AS DATE)" in insert_sql
    assert "VALUES (\n                %s, %s, %s, %s, %s," in insert_sql
    assert insert_params[0] == "payments_risk_v1"
    assert insert_params[1] == "Payments Risk"
    assert insert_params[12] == "rolling"
    assert insert_params[14] is None
    assert insert_params[15] is None
    assert insert_params[21] == "fraud_monitoring"
    assert insert_params[25] == "7"


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


def test_ensure_control_plane_ignores_field_already_exists_during_migration() -> None:
    warehouse = FakeWarehouse()
    warehouse.alter_field_already_exists = True
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.ensure_control_plane()

    assert any("CREATE SCHEMA IF NOT EXISTS" in sql for sql in warehouse.executed)
    assert any("CREATE TABLE IF NOT EXISTS" in sql for sql in warehouse.executed)


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
    assert repository.started_runs[0]["run_kind"] == "skipped"
    assert repository.completed_runs[0]["status"] == "skipped"


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
