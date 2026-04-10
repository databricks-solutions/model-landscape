from __future__ import annotations

import decimal
import logging
import os
import threading

import pandas as pd

from model_lens.services.sql_utils import validate_identifier


logger = logging.getLogger(__name__)


class WarehouseConnection:
    def __init__(self, warehouse_id: str = "", host: str = ""):
        from databricks.sdk.core import Config

        self._cfg = Config()
        self._warehouse_id = warehouse_id or os.getenv("SQL_WAREHOUSE_ID", "")
        self._http_path = f"/sql/1.0/warehouses/{self._warehouse_id}"
        raw_host = host or self._cfg.host or ""
        self._host = raw_host.replace("https://", "").replace("http://", "").rstrip("/")
        self._conn = None
        self._cache: dict[str, pd.DataFrame] = {}

    def _invalidate_cache(self) -> None:
        self._cache.clear()

    @property
    def configured(self) -> bool:
        return bool(self._warehouse_id)

    def _get_connection(self):
        if not self._warehouse_id:
            raise RuntimeError("SQL_WAREHOUSE_ID is not configured")
        if self._conn is None:
            from databricks import sql
            from databricks.sdk import WorkspaceClient

            workspace = WorkspaceClient()
            host = (workspace.config.host or "").replace("https://", "").replace("http://", "").rstrip("/") or self._host
            self._host = host

            try:
                self._conn = sql.connect(
                    server_hostname=host,
                    http_path=self._http_path,
                    credentials_provider=lambda: workspace.config.authenticate,
                )
                with self._conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            except Exception as error:
                logger.warning("credentials_provider connection failed: %s", error)
                self._conn = None
                headers = workspace.config.authenticate()
                token = headers.get("Authorization", "").replace("Bearer ", "")
                if not token:
                    raise RuntimeError("No Databricks auth token available for SQL connection") from error
                self._conn = sql.connect(
                    server_hostname=host,
                    http_path=self._http_path,
                    access_token=token,
                )
        return self._conn

    def _retry(self, fn):
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                return fn(cursor)
        except Exception:
            self._conn = None
            conn = self._get_connection()
            with conn.cursor() as cursor:
                return fn(cursor)

    def _to_dataframe(self, cursor) -> pd.DataFrame:
        if cursor.description is None:
            return pd.DataFrame()
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        frame = pd.DataFrame(rows, columns=columns)
        for column in frame.columns:
            if len(frame) > 0 and frame[column].dtype == object and isinstance(frame[column].iloc[0], decimal.Decimal):
                frame[column] = frame[column].astype(float)
        return frame

    def query(self, sql: str, cache: bool = False) -> pd.DataFrame:
        if cache and sql in self._cache:
            return self._cache[sql].copy()

        def run(cursor):
            cursor.execute(sql)
            return self._to_dataframe(cursor)

        frame = self._retry(run)
        if cache:
            self._cache[sql] = frame.copy()
        return frame

    def _resolve_params(self, sql: str, params: tuple) -> tuple[str, dict]:
        if not params or "%s" not in sql:
            return sql, {}
        parts = sql.split("%s")
        if len(parts) - 1 != len(params):
            raise ValueError("Parameter count mismatch")
        param_dict: dict[str, object] = {}
        rebuilt = [parts[0]]
        for index, value in enumerate(params):
            key = f"p{index}"
            param_dict[key] = value
            rebuilt.append(f":{key}")
            rebuilt.append(parts[index + 1])
        return "".join(rebuilt), param_dict

    def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
        resolved_sql, resolved = self._resolve_params(sql, params)

        def run(cursor):
            cursor.execute(resolved_sql, parameters=resolved)
            return self._to_dataframe(cursor)

        return self._retry(run)

    def execute(self, sql: str) -> None:
        self._retry(lambda cursor: cursor.execute(sql))
        self._invalidate_cache()

    def execute_params(self, sql: str, params: tuple) -> None:
        resolved_sql, resolved = self._resolve_params(sql, params)
        self._retry(lambda cursor: cursor.execute(resolved_sql, parameters=resolved))
        self._invalidate_cache()

    def execute_batch(self, insert_template: str, rows: list[tuple], batch_size: int = 200) -> None:
        if not rows:
            return
        for index in range(0, len(rows), batch_size):
            batch = rows[index:index + batch_size]
            param_dict: dict[str, object] = {}
            placeholders: list[str] = []
            for row_index, row in enumerate(batch):
                names: list[str] = []
                for col_index, value in enumerate(row):
                    key = f"p{row_index}_{col_index}"
                    param_dict[key] = value
                    names.append(f":{key}")
                placeholders.append("(" + ", ".join(names) + ")")
            sql = f"{insert_template} {', '.join(placeholders)}"
            self._retry(lambda cursor, statement=sql, parameters=param_dict: cursor.execute(statement, parameters=parameters))
        self._invalidate_cache()

    def describe_table(self, table_name: str, *, cache: bool = True) -> pd.DataFrame:
        validate_identifier(table_name)
        frame = self.query(f"DESCRIBE TABLE {table_name}", cache=cache)
        if frame.empty or "col_name" not in frame.columns:
            return pd.DataFrame(columns=["col_name", "data_type"])
        frame = frame[~frame["col_name"].astype(str).str.startswith("#", na=False)].copy()
        frame = frame[frame["col_name"].astype(str).str.strip() != ""]
        keep = [column for column in ("col_name", "data_type", "comment") if column in frame.columns]
        return frame[keep].reset_index(drop=True)

    def get_columns(self, table_name: str, *, cache: bool = True) -> list[str]:
        frame = self.describe_table(table_name, cache=cache)
        if frame.empty:
            return []
        return [str(value) for value in frame["col_name"].tolist()]


_warehouse_instance = None
_warehouse_lock = threading.Lock()


def get_warehouse(warehouse_id: str = "") -> WarehouseConnection:
    global _warehouse_instance
    if warehouse_id:
        return WarehouseConnection(warehouse_id=warehouse_id)
    if _warehouse_instance is None:
        with _warehouse_lock:
            if _warehouse_instance is None:
                _warehouse_instance = WarehouseConnection()
    return _warehouse_instance
