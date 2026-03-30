from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd

from model_lens.config import settings
from model_lens.domain.models import BaselinePolicy, InferenceContract, MLflowLineage, MonitorConfig, RefreshResult
from model_lens.services.lakebase import LakebaseConnection, LakebaseReadModel
from model_lens.services.schema import ddl, monitor_config_migration_columns
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

    def ensure_control_plane(self, *, create_catalog: bool = False) -> None:
        if create_catalog:
            self._warehouse.execute(f"CREATE CATALOG IF NOT EXISTS {self._table_names.catalog}")
        self._warehouse.execute(f"CREATE SCHEMA IF NOT EXISTS {self._table_names.namespace}")
        for statement in ddl(self._table_names).values():
            self._warehouse.execute(statement)
        self._ensure_monitor_config_columns()
        if self._read_model and self._read_model.configured:
            self._read_model.ensure_schema()

    def _ensure_monitor_config_columns(self) -> None:
        try:
            schema = self._warehouse.describe_table(self._table_names.monitor_configs)
        except Exception:
            return
        existing = {str(value) for value in schema.get("col_name", pd.Series(dtype=str)).tolist()}
        for column_name, data_type in monitor_config_migration_columns().items():
            if column_name in existing:
                continue
            self._warehouse.execute(
                f"ALTER TABLE {self._table_names.monitor_configs} "
                f"ADD COLUMNS ({validate_identifier(column_name)} {data_type})"
            )

    def scan_source_table(self, table_name: str, preview_rows: int = 5) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
        validate_identifier(table_name)
        preview_rows = max(1, min(preview_rows, 20))
        schema = self._warehouse.describe_table(table_name)
        columns = [str(value) for value in schema.get("col_name", pd.Series(dtype=str)).tolist()]
        preview = self._warehouse.query(f"SELECT * FROM {table_name} LIMIT {preview_rows}")
        return columns, preview, schema

    def sample_distinct_values(self, table_name: str, column_name: str, limit: int = 20) -> list[str]:
        validate_identifier(table_name)
        column_name = validate_identifier(column_name)
        limit = max(1, min(limit, 50))
        frame = self._warehouse.query(
            f"""
            SELECT DISTINCT {quote_column(column_name)} AS sampled_value
            FROM {table_name}
            WHERE {quote_column(column_name)} IS NOT NULL
            LIMIT {limit}
            """
        )
        if frame.empty or "sampled_value" not in frame.columns:
            return []
        values: list[str] = []
        for value in frame["sampled_value"].tolist():
            text = str(value).strip()
            if text:
                values.append(text)
        return values

    def upsert_monitor_config(self, config: MonitorConfig) -> None:
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
                timestamp_col, model_id_col, model_id_value, prediction_col,
                model_version_col, model_version_value, prediction_score_col, label_col, entity_id_col,
                feature_columns, slice_columns, categorical_columns,
                baseline_kind, baseline_n_days, baseline_max_comparison_days,
                problem_type, labels_table, labels_join_col, labels_order_col,
                mlflow_experiment_name, mlflow_experiment_id, mlflow_run_id,
                mlflow_registered_model_name, mlflow_model_version,
                created_by, status,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                {feature_columns}, {slice_columns}, {categorical_columns},
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s,
                CAST(%s AS TIMESTAMP), CAST(%s AS TIMESTAMP)
            )
            """,
            (
                config.model_key,
                config.display_name,
                config.source_table,
                config.contract.timestamp_col,
                config.contract.model_id_col,
                config.model_id_value or "",
                config.contract.prediction_col,
                config.contract.model_version_col or "",
                config.model_version_value or "",
                config.contract.prediction_score_col or "",
                config.contract.label_col or "",
                config.contract.entity_id_col or "",
                config.baseline.kind,
                config.baseline.n_days,
                config.baseline.max_comparison_days,
                config.problem_type,
                config.labels_table or "",
                config.labels_join_col or "",
                config.labels_order_col or "",
                config.mlflow.experiment_name or "",
                config.mlflow.experiment_id or "",
                config.mlflow.run_id or "",
                config.mlflow.registered_model_name or "",
                config.mlflow.model_version or "",
                config.created_by,
                "active",
                now,
                now,
            ),
        )
        self._sync_read_model()

    def list_monitor_configs(self, status: str = "active") -> list[MonitorConfig]:
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
            kind=_as_text(row.get("baseline_kind")) or "rolling_n_days",
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
            model_id_value=_as_text(row.get("model_id_value")) or None,
            model_version_value=_as_text(row.get("model_version_value")) or None,
            labels_table=_as_text(row.get("labels_table")) or None,
            labels_join_col=_as_text(row.get("labels_join_col")) or None,
            labels_order_col=_as_text(row.get("labels_order_col")) or None,
            mlflow=MLflowLineage(
                experiment_name=_as_text(row.get("mlflow_experiment_name")) or None,
                experiment_id=_as_text(row.get("mlflow_experiment_id")) or None,
                run_id=_as_text(row.get("mlflow_run_id")) or None,
                registered_model_name=_as_text(row.get("mlflow_registered_model_name")) or None,
                model_version=_as_text(row.get("mlflow_model_version")) or None,
            ),
            created_by=_as_text(row.get("created_by")) or "app",
        )

    def validate_monitor_source(self, config: MonitorConfig) -> None:
        validate_identifier(config.source_table)
        if config.model_version_value and not config.contract.model_version_col:
            raise ValueError("Model Version Value requires a mapped Model Version Column.")

        if not config.model_id_value:
            distinct_models = self._warehouse.query(
                f"""
                SELECT COUNT(DISTINCT {quote_column(validate_identifier(config.contract.model_id_col))}) AS distinct_model_ids
                FROM {config.source_table}
                """
            )
            distinct_count = 0 if distinct_models.empty else int(distinct_models.iloc[0]["distinct_model_ids"] or 0)
            if distinct_count > 1:
                raise ValueError(
                    "Source table contains multiple model_id values. Set Monitored Model ID Value so this monitor targets one model."
                )

        if not config.labels_table:
            return

        if not (config.contract.label_col and config.contract.entity_id_col and config.labels_join_col):
            raise ValueError(
                "External labels require Label Column, Entity ID Column, and External Labels Join Column."
            )

        label_columns = set(self._warehouse.get_columns(config.labels_table))
        required_columns = {
            config.contract.label_col,
            config.labels_join_col,
        }
        if config.labels_order_col:
            required_columns.add(config.labels_order_col)
        missing = sorted(column for column in required_columns if column and column not in label_columns)
        if missing:
            raise ValueError(f"External labels table is missing required columns: {missing}")

        if config.labels_order_col:
            return

        duplicate_keys = self._warehouse.query(
            f"""
            SELECT COUNT(*) AS duplicate_key_count
            FROM (
                SELECT {quote_column(validate_identifier(config.labels_join_col))}
                FROM {config.labels_table}
                GROUP BY {quote_column(validate_identifier(config.labels_join_col))}
                HAVING COUNT(*) > 1
            ) duplicate_keys
            """
        )
        duplicate_count = 0 if duplicate_keys.empty else int(duplicate_keys.iloc[0]["duplicate_key_count"] or 0)
        if duplicate_count > 0:
            raise ValueError(
                "External labels table contains duplicate join keys. Provide External Labels Order Column so Model Lens can pick the latest label."
            )

    def load_monitor_frame(self, config: MonitorConfig) -> pd.DataFrame:
        self.validate_monitor_source(config)
        ts_col = validate_identifier(config.contract.timestamp_col)

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
            label_col = validate_identifier(config.contract.label_col)
            join_col = validate_identifier(config.labels_join_col)
            label_projection = f", l.{quote_column(label_col)} AS {quote_column(label_col)}"
            if config.labels_order_col:
                order_col = validate_identifier(config.labels_order_col)
                join_source = f"""
                    (
                        SELECT {quote_column(join_col)}, {quote_column(label_col)}
                        FROM (
                            SELECT
                                {quote_column(join_col)},
                                {quote_column(label_col)},
                                ROW_NUMBER() OVER (
                                    PARTITION BY {quote_column(join_col)}
                                    ORDER BY {quote_column(order_col)} DESC
                                ) AS {quote_column("model_lens_label_rank")}
                            FROM {config.labels_table}
                        ) ranked_labels
                        WHERE {quote_column("model_lens_label_rank")} = 1
                    ) l
                """
            else:
                join_source = f"{config.labels_table} l"
            join_sql = (
                f" LEFT JOIN {join_source}"
                f" ON s.{quote_column(validate_identifier(config.contract.entity_id_col))}"
                f" = l.{quote_column(join_col)}"
            )

        filters: list[str] = []
        params: list[object] = []
        if config.model_id_value:
            filters.append(f"s.{quote_column(validate_identifier(config.contract.model_id_col))} = %s")
            params.append(config.model_id_value)
        if config.model_version_value and config.contract.model_version_col:
            filters.append(f"s.{quote_column(validate_identifier(config.contract.model_version_col))} = %s")
            params.append(config.model_version_value)
        where_sql = f" WHERE {' AND '.join(filters)}" if filters else ""

        query = f"""
            SELECT {source_projection}{label_projection}
            FROM {config.source_table} s
            {join_sql}
            {where_sql}
            ORDER BY s.{quote_column(ts_col)}
        """
        frame = self._warehouse.query_params(query, tuple(params)) if params else self._warehouse.query(query)
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
                summary = self._read_model.get_monitor_summary()
                if not summary.empty:
                    return summary
                logger.warning("Lakebase summary read returned no rows, falling back to warehouse.")
            except Exception as error:
                logger.warning("Lakebase summary read failed, falling back to warehouse: %s", error)
        return self._get_monitor_summary_from_warehouse()

    def get_open_incidents(self) -> pd.DataFrame:
        if self._read_model and self._read_model.configured:
            try:
                incidents = self._read_model.get_open_incidents()
                if not incidents.empty:
                    return incidents
                logger.warning("Lakebase incidents read returned no rows, falling back to warehouse.")
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
