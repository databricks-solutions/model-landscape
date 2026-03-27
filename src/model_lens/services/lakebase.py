from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from model_lens.domain.models import MonitorConfig
from model_lens.services.sql_utils import validate_identifier


logger = logging.getLogger(__name__)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.to_pydatetime().replace(tzinfo=timezone.utc)
        return value.to_pydatetime()
    return value


class LakebaseConnection:
    def __init__(
        self,
        *,
        instance_name: str = "",
        database_name: str = "",
        host: str = "",
        port: int = 5432,
        user: str = "",
        password: str = "",
        sslmode: str = "require",
    ) -> None:
        self._instance_name = instance_name.strip()
        self._database_name = database_name.strip()
        self._host = host.strip()
        self._port = int(port or 5432)
        self._user = user.strip()
        self._password = password.strip()
        self._sslmode = sslmode.strip() or "require"

    @property
    def configured(self) -> bool:
        return bool(self._database_name and (self._host or self._instance_name))

    def _resolve_host(self) -> str:
        if self._host:
            return self._host
        if not self._instance_name:
            return ""
        from databricks.sdk import WorkspaceClient

        workspace = WorkspaceClient()
        instance = workspace.database.get_database_instance(self._instance_name)
        return (instance.read_write_dns or instance.read_only_dns or "").strip()

    def _resolve_user(self) -> str:
        if self._user:
            return self._user
        from databricks.sdk import WorkspaceClient

        workspace = WorkspaceClient()
        current_user = workspace.current_user.me()
        return (current_user.user_name or "").strip()

    def _resolve_password(self) -> str:
        if self._password:
            return self._password
        if not self._instance_name:
            return ""
        from databricks.sdk import WorkspaceClient

        workspace = WorkspaceClient()
        credential = workspace.database.generate_database_credential(instance_names=[self._instance_name])
        return (credential.token or "").strip()

    def _connection_kwargs(self) -> dict[str, Any]:
        host = self._resolve_host()
        user = self._resolve_user()
        if not host or not self._database_name or not user:
            raise RuntimeError(
                "Lakebase connection is not fully configured. "
                "Set LAKEBASE_INSTANCE_NAME/LAKEBASE_DATABASE_NAME or PGHOST/PGDATABASE/PGUSER."
            )
        kwargs: dict[str, Any] = {
            "host": host,
            "port": self._port,
            "dbname": self._database_name,
            "user": user,
            "sslmode": self._sslmode,
            "autocommit": True,
        }
        password = self._resolve_password()
        if password:
            kwargs["password"] = password
        return kwargs

    def _get_connection(self):
        import psycopg

        kwargs = self._connection_kwargs()
        try:
            return psycopg.connect(**kwargs)
        except Exception as error:
            if "password" in kwargs:
                raise
            logger.warning("Lakebase passwordless connection failed: %s", error)
            password = self._resolve_password()
            if not password:
                raise
            kwargs["password"] = password
            return psycopg.connect(**kwargs)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)

    def execute_many(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(sql, rows)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                if cursor.description is None:
                    return pd.DataFrame()
                rows = cursor.fetchall()
                columns = [str(column.name) for column in cursor.description]
        return pd.DataFrame(rows, columns=columns)


class LakebaseReadModel:
    def __init__(self, connection: LakebaseConnection, schema: str) -> None:
        self._connection = connection
        self._schema = validate_identifier(schema)

    @property
    def configured(self) -> bool:
        return self._connection.configured

    @property
    def inventory_table(self) -> str:
        return f'"{self._schema}"."monitor_inventory"'

    @property
    def summary_table(self) -> str:
        return f'"{self._schema}"."monitor_summary"'

    @property
    def incidents_table(self) -> str:
        return f'"{self._schema}"."open_incidents"'

    def ensure_schema(self) -> None:
        if not self.configured:
            return
        self._connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}"')
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.inventory_table} (
                model_key TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                source_table TEXT NOT NULL,
                feature_count INTEGER NOT NULL,
                labels_table TEXT NOT NULL,
                feature_columns TEXT[] NOT NULL,
                categorical_columns TEXT[] NOT NULL,
                slice_columns TEXT[] NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.summary_table} (
                model_key TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                max_psi DOUBLE PRECISION,
                feature_count INTEGER,
                latest_window_end DATE,
                total_rows BIGINT,
                latest_data_date DATE,
                last_refresh_at TIMESTAMPTZ,
                open_incident_count INTEGER NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.incidents_table} (
                model_key TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                metric_value DOUBLE PRECISION,
                window_end DATE,
                observed_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (model_key, feature_name, metric_name)
            )
            """
        )

    def replace_dashboard_projection(
        self,
        *,
        configs: list[MonitorConfig],
        summary: pd.DataFrame,
        incidents: pd.DataFrame,
    ) -> None:
        if not self.configured:
            return
        self.ensure_schema()
        synced_at = datetime.now(timezone.utc)
        inventory_rows = [
            (
                config.model_key,
                config.display_name,
                config.source_table,
                len(config.contract.feature_columns),
                config.labels_table or "",
                list(config.contract.feature_columns),
                list(config.contract.categorical_columns),
                list(config.contract.slice_columns),
                synced_at,
            )
            for config in configs
        ]
        summary_rows = [
            (
                _clean_text(row.get("model_key")),
                _clean_text(row.get("display_name")),
                _normalize_scalar(row.get("max_psi")),
                _normalize_scalar(row.get("feature_count")),
                _normalize_scalar(row.get("latest_window_end")),
                _normalize_scalar(row.get("total_rows")),
                _normalize_scalar(row.get("latest_data_date")),
                _normalize_scalar(row.get("last_refresh_at")),
                int(row.get("open_incident_count") or 0),
                synced_at,
            )
            for _, row in summary.iterrows()
        ]
        incident_rows = [
            (
                _clean_text(row.get("model_key")),
                _clean_text(row.get("feature_name")),
                _clean_text(row.get("metric_name")),
                _clean_text(row.get("severity")),
                _normalize_scalar(row.get("metric_value")),
                _normalize_scalar(row.get("window_end")),
                _normalize_scalar(row.get("observed_at")),
                synced_at,
            )
            for _, row in incidents.iterrows()
        ]
        self._connection.execute(f"DELETE FROM {self.inventory_table}")
        self._connection.execute(f"DELETE FROM {self.summary_table}")
        self._connection.execute(f"DELETE FROM {self.incidents_table}")
        self._connection.execute_many(
            f"""
            INSERT INTO {self.inventory_table} (
                model_key, display_name, source_table, feature_count, labels_table,
                feature_columns, categorical_columns, slice_columns, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            inventory_rows,
        )
        self._connection.execute_many(
            f"""
            INSERT INTO {self.summary_table} (
                model_key, display_name, max_psi, feature_count, latest_window_end,
                total_rows, latest_data_date, last_refresh_at, open_incident_count, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            summary_rows,
        )
        self._connection.execute_many(
            f"""
            INSERT INTO {self.incidents_table} (
                model_key, feature_name, metric_name, severity, metric_value,
                window_end, observed_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            incident_rows,
        )

    def get_monitor_summary(self) -> pd.DataFrame:
        if not self.configured:
            return pd.DataFrame()
        return self._connection.query(
            f"""
            SELECT
                model_key, display_name, max_psi, feature_count, latest_window_end,
                total_rows, latest_data_date, last_refresh_at, open_incident_count
            FROM {self.summary_table}
            ORDER BY max_psi DESC NULLS LAST, display_name
            """
        )

    def get_open_incidents(self) -> pd.DataFrame:
        if not self.configured:
            return pd.DataFrame()
        return self._connection.query(
            f"""
            SELECT model_key, feature_name, metric_name, severity, metric_value, window_end, observed_at
            FROM {self.incidents_table}
            ORDER BY
                CASE severity WHEN 'critical' THEN 0 ELSE 1 END,
                observed_at DESC
            """
        )
