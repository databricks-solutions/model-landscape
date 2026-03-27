from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from model_lens.domain.models import MonitorConfig
from model_lens.services.contracts import build_contract
from model_lens.services.control_plane import ControlPlaneRepository
from model_lens.services.onboarding import build_default_baseline
from model_lens.services.refresh_runner import run_refresh_cycle
from model_lens.services.table_names import TableNames


class FakeWarehouse:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.executed_params: list[tuple[str, tuple]] = []
        self.batch_calls: list[tuple[str, list[tuple]]] = []
        self.queries: list[str] = []
        self.monitor_row = {
            "model_key": "payments_risk_v1",
            "display_name": "Payments Risk",
            "source_table": "catalog.schema.inference_logs",
            "timestamp_col": "event_ts",
            "model_id_col": "model_id",
            "prediction_col": "prediction",
            "model_version_col": "",
            "prediction_score_col": "",
            "label_col": "label",
            "entity_id_col": "entity_id",
            "feature_columns": '["amount","segment"]',
            "slice_columns": '["segment"]',
            "categorical_columns": '["segment"]',
            "baseline_kind": "first_n_days",
            "baseline_n_days": 7,
            "baseline_max_comparison_days": 90,
            "problem_type": "classification",
            "labels_table": "",
            "labels_join_col": "",
            "created_by": "app",
        }

    def execute(self, sql: str) -> None:
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
                "prediction_col": params[5],
                "model_version_col": params[6],
                "prediction_score_col": params[7],
                "label_col": params[8],
                "entity_id_col": params[9],
                "feature_columns": '["amount","segment"]',
                "slice_columns": '["segment"]',
                "categorical_columns": '["segment"]',
                "baseline_kind": params[10],
                "baseline_n_days": params[11],
                "baseline_max_comparison_days": params[12],
                "problem_type": params[13],
                "labels_table": params[14],
                "labels_join_col": params[15],
                "created_by": params[16],
            }

    def execute_batch(self, insert_template: str, rows: list[tuple], batch_size: int = 200) -> None:
        del batch_size
        self.batch_calls.append((insert_template, rows))

    def query(self, sql: str, cache: bool = False) -> pd.DataFrame:
        del cache
        self.queries.append(sql)
        if "MIN(CAST" in sql:
            return pd.DataFrame([{"min_date": "2026-01-01", "max_date": "2026-01-20"}])
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
                "event_ts": "2026-01-10T00:00:00",
                "model_id": "m1",
                "prediction": 0.9,
                "entity_id": "entity-1",
                "label": 1,
                "amount": 10.0,
            }
        ])

    def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
        del params
        if "SELECT * FROM" in sql and "monitor_configs" in sql:
            return pd.DataFrame([self.monitor_row])
        return pd.DataFrame()

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
        del table_name
        return ["event_ts", "model_id", "prediction", "entity_id", "amount"]


def _monitor_config(
    *,
    with_external_labels: bool = False,
    days: int = 7,
    max_comparison_days: int = 90,
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
        labels_table="catalog.schema.labels" if with_external_labels else None,
        labels_join_col="entity_id" if with_external_labels else None,
        created_by="app",
    )


def test_upsert_monitor_config_keeps_full_feature_and_categorical_metadata() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    repository.upsert_monitor_config(_monitor_config())

    insert_sql, insert_params = warehouse.executed_params[-1]
    assert "ARRAY('amount', 'segment')" in insert_sql
    assert "ARRAY('segment')" in insert_sql
    assert insert_params[0] == "payments_risk_v1"
    assert insert_params[1] == "Payments Risk"


def test_load_monitor_frame_uses_external_labels_join_and_comparison_window() -> None:
    warehouse = FakeWarehouse()
    repository = ControlPlaneRepository(warehouse=warehouse, table_names=TableNames("model_observability", "control_plane"))

    frame = repository.load_monitor_frame(_monitor_config(with_external_labels=True, max_comparison_days=10))

    assert not frame.empty
    data_query = warehouse.queries[-1]
    assert "LEFT JOIN catalog.schema.labels l" in data_query
    assert "BETWEEN '2026-01-01' AND '2026-01-17'" in data_query
    assert "l.`label` AS `label`" in data_query


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


class StubRepository:
    def __init__(self) -> None:
        self.config = _monitor_config(days=7)
        self.replaced: list[str] = []

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

    def replace_refresh_result(self, model_key: str, result) -> None:
        del result
        self.replaced.append(model_key)


def test_run_refresh_cycle_skips_replacing_results_when_not_enough_data() -> None:
    repository = StubRepository()

    counts = run_refresh_cycle(repository)

    assert counts.models == 0
    assert counts.drift_rows == 0
    assert counts.quality_rows == 0
    assert counts.performance_rows == 0
    assert counts.incident_rows == 0
    assert repository.replaced == []


class FakeReadModel:
    def __init__(self) -> None:
        self.configured = True
        self.ensured = 0
        self.synced: list[dict] = []

    def ensure_schema(self) -> None:
        self.ensured += 1

    def replace_dashboard_projection(self, *, configs, summary, incidents) -> None:
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
