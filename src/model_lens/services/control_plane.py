from __future__ import annotations

import logging
from datetime import timedelta
from functools import lru_cache
from typing import Any

import pandas as pd

from model_lens.config import settings
from model_lens.domain.models import BaselinePolicy, InferenceContract, MonitorConfig, RefreshResult
from model_lens.services.lakebase import LakebaseConnection, LakebaseReadModel
from model_lens.services.schema import ddl
from model_lens.services.sql_utils import array_literal, parse_string_array, quote_column, validate_identifier
from model_lens.services.table_names import TableNames
from model_lens.services.warehouse import WarehouseConnection, get_warehouse


logger = logging.getLogger(__name__)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


class ControlPlaneRepository:
    def __init__(
        self,
        warehouse: WarehouseConnection,
        table_names: TableNames,
        read_model: LakebaseReadModel | None = None,
    ):
        self._warehouse = warehouse
        self._table_names = table_names
        self._read_model = read_model

    @property
    def table_names(self) -> TableNames:
        return self._table_names

    def ensure_control_plane(self) -> None:
        self._warehouse.execute(f"CREATE CATALOG IF NOT EXISTS {self._table_names.catalog}")
        self._warehouse.execute(f"CREATE SCHEMA IF NOT EXISTS {self._table_names.namespace}")
        for statement in ddl(self._table_names).values():
            self._warehouse.execute(statement)
        if self._read_model and self._read_model.configured:
            self._read_model.ensure_schema()

    def scan_source_table(self, table_name: str, preview_rows: int = 5) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
        validate_identifier(table_name)
        preview_rows = max(1, min(preview_rows, 20))
        schema = self._warehouse.describe_table(table_name)
        columns = [str(value) for value in schema.get("col_name", pd.Series(dtype=str)).tolist()]
        preview = self._warehouse.query(f"SELECT * FROM {table_name} LIMIT {preview_rows}")
        return columns, preview, schema

    def upsert_monitor_config(self, config: MonitorConfig) -> None:
        self.ensure_control_plane()
        now = pd.Timestamp.utcnow().isoformat()
        feature_columns = array_literal(list(config.contract.feature_columns))
        slice_columns = array_literal(list(config.contract.slice_columns))
        categorical_columns = array_literal(list(config.contract.categorical_columns))
        self._warehouse.execute_params(
            f"DELETE FROM {self._table_names.monitor_configs} WHERE model_key = %s",
            (config.model_key,),
        )
        self._warehouse.execute_params(
            f"""
            INSERT INTO {self._table_names.monitor_configs} (
                model_key, display_name, source_table,
                timestamp_col, model_id_col, prediction_col,
                model_version_col, prediction_score_col, label_col, entity_id_col,
                feature_columns, slice_columns, categorical_columns,
                baseline_kind, baseline_n_days, baseline_max_comparison_days,
                problem_type, labels_table, labels_join_col, created_by, status,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                {feature_columns}, {slice_columns}, {categorical_columns},
                %s, %s, %s,
                %s, %s, %s, %s, %s,
                CAST(%s AS TIMESTAMP), CAST(%s AS TIMESTAMP)
            )
            """,
            (
                config.model_key,
                config.display_name,
                config.source_table,
                config.contract.timestamp_col,
                config.contract.model_id_col,
                config.contract.prediction_col,
                config.contract.model_version_col or "",
                config.contract.prediction_score_col or "",
                config.contract.label_col or "",
                config.contract.entity_id_col or "",
                config.baseline.kind,
                config.baseline.n_days,
                config.baseline.max_comparison_days,
                config.problem_type,
                config.labels_table or "",
                config.labels_join_col or "",
                config.created_by,
                "active",
                now,
                now,
            ),
        )
        self._sync_read_model()

    def list_monitor_configs(self, status: str = "active") -> list[MonitorConfig]:
        self.ensure_control_plane()
        if status:
            frame = self._warehouse.query_params(
                f"SELECT * FROM {self._table_names.monitor_configs} WHERE status = %s ORDER BY updated_at DESC",
                (status,),
            )
        else:
            frame = self._warehouse.query(f"SELECT * FROM {self._table_names.monitor_configs} ORDER BY updated_at DESC")
        return [self._row_to_monitor_config(row) for _, row in frame.iterrows()]

    def _row_to_monitor_config(self, row: pd.Series) -> MonitorConfig:
        contract = InferenceContract(
            timestamp_col=_as_text(row.get("timestamp_col")),
            model_id_col=_as_text(row.get("model_id_col")),
            prediction_col=_as_text(row.get("prediction_col")),
            model_version_col=_as_text(row.get("model_version_col")) or None,
            prediction_score_col=_as_text(row.get("prediction_score_col")) or None,
            label_col=_as_text(row.get("label_col")) or None,
            entity_id_col=_as_text(row.get("entity_id_col")) or None,
            feature_columns=parse_string_array(row.get("feature_columns")),
            slice_columns=parse_string_array(row.get("slice_columns")),
            categorical_columns=parse_string_array(row.get("categorical_columns")),
        )
        baseline = BaselinePolicy(
            kind=_as_text(row.get("baseline_kind")) or "first_n_days",
            n_days=int(row.get("baseline_n_days", 7) or 7),
            max_comparison_days=int(row.get("baseline_max_comparison_days", 90) or 90),
        )
        return MonitorConfig(
            model_key=_as_text(row.get("model_key")),
            display_name=_as_text(row.get("display_name")),
            source_table=_as_text(row.get("source_table")),
            contract=contract,
            baseline=baseline,
            problem_type=_as_text(row.get("problem_type")) or "classification",
            labels_table=_as_text(row.get("labels_table")) or None,
            labels_join_col=_as_text(row.get("labels_join_col")) or None,
            created_by=_as_text(row.get("created_by")) or "app",
        )

    def load_monitor_frame(self, config: MonitorConfig) -> pd.DataFrame:
        validate_identifier(config.source_table)
        ts_col = validate_identifier(config.contract.timestamp_col)

        bounds = self._warehouse.query(
            f"""
            SELECT
                MIN(CAST({quote_column(ts_col)} AS DATE)) AS min_date,
                MAX(CAST({quote_column(ts_col)} AS DATE)) AS max_date
            FROM {config.source_table}
            """
        )
        if bounds.empty or pd.isna(bounds.iloc[0]["min_date"]) or pd.isna(bounds.iloc[0]["max_date"]):
            return pd.DataFrame()

        min_date = pd.to_datetime(bounds.iloc[0]["min_date"]).date()
        max_date = pd.to_datetime(bounds.iloc[0]["max_date"]).date()
        baseline_end = min_date + timedelta(days=max(config.baseline.n_days - 1, 0))
        comparison_end = min(max_date, baseline_end + timedelta(days=config.baseline.max_comparison_days))

        select_columns = [
            config.contract.timestamp_col,
            config.contract.model_id_col,
            config.contract.prediction_col,
        ]
        optional_source_columns = [
            config.contract.model_version_col,
            config.contract.prediction_score_col,
            config.contract.entity_id_col,
        ]
        for column in optional_source_columns:
            if column:
                select_columns.append(column)
        for column in config.contract.feature_columns:
            select_columns.append(column)
        source_label_column = None
        if config.contract.label_col and not config.labels_table:
            source_label_column = config.contract.label_col
            select_columns.append(config.contract.label_col)

        deduped_columns = list(dict.fromkeys(select_columns))
        source_projection = ", ".join(
            f"s.{quote_column(validate_identifier(column))} AS {quote_column(validate_identifier(column))}"
            for column in deduped_columns
        )

        label_projection = ""
        join_sql = ""
        if config.labels_table and config.contract.label_col and config.contract.entity_id_col and config.labels_join_col:
            validate_identifier(config.labels_table)
            label_projection = (
                ", "
                f"l.{quote_column(validate_identifier(config.contract.label_col))} AS {quote_column(validate_identifier(config.contract.label_col))}"
            )
            join_sql = (
                f" LEFT JOIN {config.labels_table} l"
                f" ON s.{quote_column(validate_identifier(config.contract.entity_id_col))}"
                f" = l.{quote_column(validate_identifier(config.labels_join_col))}"
            )

        query = f"""
            SELECT {source_projection}{label_projection}
            FROM {config.source_table} s
            {join_sql}
            WHERE CAST(s.{quote_column(ts_col)} AS DATE) BETWEEN '{min_date}' AND '{comparison_end}'
            ORDER BY s.{quote_column(ts_col)}
        """
        frame = self._warehouse.query(query)
        if source_label_column and source_label_column not in frame.columns:
            frame[source_label_column] = pd.NA
        return frame

    def replace_refresh_result(self, model_key: str, result: RefreshResult) -> None:
        window_end = ""
        if result.drift_rows:
            window_end = _as_text(result.drift_rows[0].get("window_end"))
        elif result.performance_rows:
            window_end = _as_text(result.performance_rows[0].get("window_end"))

        if window_end:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.drift_metrics} WHERE model_key = %s AND window_end = CAST(%s AS DATE)",
                (model_key, window_end),
            )
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.performance_metrics} WHERE model_key = %s AND window_end = CAST(%s AS DATE)",
                (model_key, window_end),
            )
        self._warehouse.execute_params(
            f"DELETE FROM {self._table_names.quality_metrics} WHERE model_key = %s",
            (model_key,),
        )
        self._warehouse.execute_params(
            f"DELETE FROM {self._table_names.incidents} WHERE model_key = %s",
            (model_key,),
        )

        self._insert_drift_rows(result.drift_rows)
        self._insert_quality_rows(result.quality_rows)
        self._insert_performance_rows(result.performance_rows)
        self._insert_incident_rows(result.incident_rows)
        self._sync_read_model()

    def _insert_drift_rows(self, rows: list[dict]) -> None:
        payload = [
            (
                row["model_key"],
                row["feature_name"],
                row["metric_name"],
                row["metric_value"],
                row["window_start"],
                row["window_end"],
                row["baseline_start"],
                row["baseline_end"],
                row["ref_mean"],
                row["cur_mean"],
                row["ref_std"],
                row["cur_std"],
                row["ref_null_pct"],
                row["cur_null_pct"],
                row["ref_count"],
                row["cur_count"],
                row["computed_at"],
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.drift_metrics} (
                model_key, feature_name, metric_name, metric_value,
                window_start, window_end, baseline_start, baseline_end,
                ref_mean, cur_mean, ref_std, cur_std,
                ref_null_pct, cur_null_pct, ref_count, cur_count, computed_at
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_quality_rows(self, rows: list[dict]) -> None:
        payload = [
            (
                row["model_key"],
                row["total_rows"],
                row["min_date"],
                row["max_date"],
                row["prediction_mean"],
                row["prediction_std"],
                row["daily_volume"],
                row["null_rates"],
                row["computed_at"],
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.quality_metrics} (
                model_key, total_rows, min_date, max_date,
                prediction_mean, prediction_std, daily_volume, null_rates, computed_at
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_performance_rows(self, rows: list[dict]) -> None:
        payload = [
            (
                row["model_key"],
                row["feature_name"],
                row["bin_label"],
                row["baseline_metric"],
                row["current_metric"],
                row["delta"],
                row["volume_pct"],
                row["contribution"],
                row["metric_name"],
                row["window_start"],
                row["window_end"],
                row["computed_at"],
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.performance_metrics} (
                model_key, feature_name, bin_label,
                baseline_metric, current_metric, delta,
                volume_pct, contribution, metric_name,
                window_start, window_end, computed_at
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_incident_rows(self, rows: list[dict]) -> None:
        payload = [
            (
                row["model_key"],
                row["feature_name"],
                row["metric_name"],
                row["severity"],
                row["status"],
                row["metric_value"],
                row["window_end"],
                row["observed_at"],
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.incidents} (
                model_key, feature_name, metric_name,
                severity, status, metric_value, window_end, observed_at
            ) VALUES
            """.strip(),
            payload,
        )

    def _get_monitor_summary_from_warehouse(self) -> pd.DataFrame:
        self.ensure_control_plane()
        return self._warehouse.query(
            f"""
            WITH latest_window AS (
                SELECT model_key, MAX(window_end) AS latest_window_end
                FROM {self._table_names.drift_metrics}
                GROUP BY model_key
            ),
            latest_quality AS (
                SELECT q.*
                FROM {self._table_names.quality_metrics} q
                INNER JOIN (
                    SELECT model_key, MAX(computed_at) AS latest_computed_at
                    FROM {self._table_names.quality_metrics}
                    GROUP BY model_key
                ) latest
                    ON q.model_key = latest.model_key
                   AND q.computed_at = latest.latest_computed_at
            ),
            open_incidents AS (
                SELECT model_key, COUNT(*) AS open_incident_count
                FROM {self._table_names.incidents}
                WHERE status = 'open'
                GROUP BY model_key
            )
            SELECT
                c.model_key,
                c.display_name,
                COALESCE(MAX(CASE WHEN d.metric_name = 'psi' THEN d.metric_value END), 0) AS max_psi,
                COALESCE(SUM(CASE WHEN d.metric_name = 'psi' THEN 1 ELSE 0 END), 0) AS feature_count,
                MAX(d.window_end) AS latest_window_end,
                q.total_rows,
                q.max_date AS latest_data_date,
                q.computed_at AS last_refresh_at,
                COALESCE(i.open_incident_count, 0) AS open_incident_count
            FROM {self._table_names.monitor_configs} c
            LEFT JOIN latest_window lw ON c.model_key = lw.model_key
            LEFT JOIN {self._table_names.drift_metrics} d
                ON c.model_key = d.model_key
               AND d.window_end = lw.latest_window_end
            LEFT JOIN latest_quality q
                ON c.model_key = q.model_key
            LEFT JOIN open_incidents i
                ON c.model_key = i.model_key
            WHERE c.status = 'active'
            GROUP BY
                c.model_key,
                c.display_name,
                q.total_rows,
                q.max_date,
                q.computed_at,
                i.open_incident_count
            ORDER BY max_psi DESC, c.display_name
            """
        )

    def _get_open_incidents_from_warehouse(self) -> pd.DataFrame:
        self.ensure_control_plane()
        return self._warehouse.query(
            f"""
            SELECT model_key, feature_name, metric_name, severity, metric_value, window_end, observed_at
            FROM {self._table_names.incidents}
            WHERE status = 'open'
            ORDER BY
                CASE severity WHEN 'critical' THEN 0 ELSE 1 END,
                observed_at DESC
            """
        )

    def _sync_read_model(self) -> None:
        if not self._read_model or not self._read_model.configured:
            return
        try:
            self._read_model.replace_dashboard_projection(
                configs=self.list_monitor_configs(status="active"),
                summary=self._get_monitor_summary_from_warehouse(),
                incidents=self._get_open_incidents_from_warehouse(),
            )
        except Exception as error:
            logger.warning("Lakebase read-model sync failed: %s", error)

    def get_monitor_summary(self) -> pd.DataFrame:
        if self._read_model and self._read_model.configured:
            try:
                return self._read_model.get_monitor_summary()
            except Exception as error:
                logger.warning("Lakebase summary read failed, falling back to warehouse: %s", error)
        return self._get_monitor_summary_from_warehouse()

    def get_open_incidents(self) -> pd.DataFrame:
        if self._read_model and self._read_model.configured:
            try:
                return self._read_model.get_open_incidents()
            except Exception as error:
                logger.warning("Lakebase incidents read failed, falling back to warehouse: %s", error)
        return self._get_open_incidents_from_warehouse()


def _build_read_model(
    *,
    use_lakebase_read_model: bool | None = None,
    lakebase_instance_name: str | None = None,
    lakebase_database_name: str | None = None,
    lakebase_host: str | None = None,
    lakebase_port: int | None = None,
    lakebase_pguser: str | None = None,
    lakebase_password: str | None = None,
    lakebase_sslmode: str | None = None,
    lakebase_schema: str | None = None,
) -> LakebaseReadModel | None:
    enabled = settings.use_lakebase_read_model if use_lakebase_read_model is None else use_lakebase_read_model
    if not enabled and not any((
        lakebase_instance_name,
        lakebase_database_name,
        lakebase_host,
        settings.lakebase_instance_name,
        settings.lakebase_database_name,
        settings.lakebase_host,
    )):
        return None
    connection = LakebaseConnection(
        instance_name=lakebase_instance_name if lakebase_instance_name is not None else settings.lakebase_instance_name,
        database_name=lakebase_database_name if lakebase_database_name is not None else settings.lakebase_database_name,
        host=lakebase_host if lakebase_host is not None else settings.lakebase_host,
        port=lakebase_port if lakebase_port is not None else settings.lakebase_port,
        user=lakebase_pguser if lakebase_pguser is not None else settings.lakebase_pguser,
        password=lakebase_password if lakebase_password is not None else settings.lakebase_password,
        sslmode=lakebase_sslmode if lakebase_sslmode is not None else settings.lakebase_sslmode,
    )
    if not connection.configured:
        return None
    return LakebaseReadModel(
        connection=connection,
        schema=lakebase_schema or settings.lakebase_schema,
    )


@lru_cache(maxsize=1)
def get_default_repository() -> ControlPlaneRepository:
    table_names = TableNames(
        catalog=settings.control_plane_catalog,
        schema=settings.control_plane_schema,
    )
    return ControlPlaneRepository(
        warehouse=get_warehouse(),
        table_names=table_names,
        read_model=_build_read_model(),
    )


def build_repository(
    *,
    warehouse_id: str = "",
    catalog: str | None = None,
    schema: str | None = None,
    use_lakebase_read_model: bool | None = None,
    lakebase_instance_name: str | None = None,
    lakebase_database_name: str | None = None,
    lakebase_host: str | None = None,
    lakebase_port: int | None = None,
    lakebase_pguser: str | None = None,
    lakebase_password: str | None = None,
    lakebase_sslmode: str | None = None,
    lakebase_schema: str | None = None,
) -> ControlPlaneRepository:
    return ControlPlaneRepository(
        warehouse=get_warehouse(warehouse_id=warehouse_id or settings.sql_warehouse_id),
        table_names=TableNames(
            catalog=catalog or settings.control_plane_catalog,
            schema=schema or settings.control_plane_schema,
        ),
        read_model=_build_read_model(
            use_lakebase_read_model=use_lakebase_read_model,
            lakebase_instance_name=lakebase_instance_name,
            lakebase_database_name=lakebase_database_name,
            lakebase_host=lakebase_host,
            lakebase_port=lakebase_port,
            lakebase_pguser=lakebase_pguser,
            lakebase_password=lakebase_password,
            lakebase_sslmode=lakebase_sslmode,
            lakebase_schema=lakebase_schema,
        ),
    )
