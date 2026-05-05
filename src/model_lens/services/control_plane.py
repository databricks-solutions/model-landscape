from __future__ import annotations

import json
import logging
import re
from datetime import timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from model_lens.config import settings
from model_lens.domain.models import (
    BaselinePolicy,
    InferenceContract,
    MLflowLineage,
    MonitorConfig,
    MonitorRuntimeState,
    RefreshResult,
)
from model_lens.services.inference_contracts import build_inference_contract
from model_lens.services.class_filters import NEGATIVE_CLASS_TOKENS, POSITIVE_CLASS_TOKENS
from model_lens.services.lakebase import LakebaseConnection, LakebaseReadModel
from model_lens.services.schema import (
    comparison_window_migration_columns,
    daily_class_feature_profile_migration_columns,
    daily_class_quality_profile_migration_columns,
    daily_feature_profile_migration_columns,
    daily_label_metric_migration_columns,
    daily_performance_profile_migration_columns,
    daily_quality_profile_migration_columns,
    ddl,
    drift_metric_migration_columns,
    incident_history_migration_columns,
    incident_migration_columns,
    monitor_config_migration_columns,
    performance_metric_migration_columns,
    performance_bin_spec_migration_columns,
    quality_history_migration_columns,
    quality_metric_migration_columns,
    refresh_run_migration_columns,
    runtime_state_migration_columns,
)
from model_lens.services.sql_utils import array_literal, parse_string_array, quote_column, validate_identifier
from model_lens.services.table_names import TableNames
from model_lens.services.warehouse import WarehouseConnection, get_warehouse


logger = logging.getLogger(__name__)

_MAX_DAILY_HISTORY_DAYS = 400


class PermanentDeleteUnsupportedError(RuntimeError):
    """Raised when the configured control-plane tables cannot safely support permanent delete."""


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _as_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _sql_binary_indicator(expression: str) -> str:
    positive_tokens = ", ".join(f"'{token}'" for token in sorted(POSITIVE_CLASS_TOKENS))
    negative_tokens = ", ".join(f"'{token}'" for token in sorted(NEGATIVE_CLASS_TOKENS))
    numeric = f"CAST({expression} AS DOUBLE)"
    text = f"LOWER(TRIM(CAST({expression} AS STRING)))"
    return (
        "CASE "
        f"WHEN {text} IN ({positive_tokens}) THEN 1 "
        f"WHEN {text} IN ({negative_tokens}) THEN 0 "
        f"WHEN {numeric} = 1.0 THEN 1 "
        f"WHEN {numeric} = 0.0 THEN 0 "
        "ELSE NULL END"
    )


def _sql_probability_indicator(expression: str) -> str:
    numeric = f"CAST({expression} AS DOUBLE)"
    return (
        "CASE "
        f"WHEN {numeric} IS NOT NULL AND {numeric} BETWEEN 0.0 AND 1.0 "
        f"THEN CASE WHEN {numeric} >= 0.5 THEN 1 ELSE 0 END "
        "ELSE NULL END"
    )


def _sql_binary_prediction_expression(prediction_expression: str, *, score_expression: str | None = None) -> str:
    parts = [_sql_binary_indicator(prediction_expression)]
    if score_expression:
        parts.append(_sql_probability_indicator(score_expression))
    parts.append(_sql_probability_indicator(prediction_expression))
    return "COALESCE(" + ", ".join(parts) + ")"


def _safe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): inner for key, inner in value.items()}
    text = str(value).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): inner for key, inner in loaded.items()}


def _combine_weighted_mean_std(parts: list[tuple[int, float | None, float | None]]) -> tuple[float | None, float | None]:
    valid_parts = [(count, mean, std) for count, mean, std in parts if count > 0 and mean is not None]
    if not valid_parts:
        return None, None
    total_count = sum(count for count, _, _ in valid_parts)
    if total_count <= 0:
        return None, None
    combined_mean = sum(count * float(mean) for count, mean, _ in valid_parts) / total_count
    if total_count <= 1:
        return combined_mean, None
    total_ss = 0.0
    for count, mean, std in valid_parts:
        local_ss = 0.0
        if std is not None and count > 1:
            local_ss = (count - 1) * (float(std) ** 2)
        total_ss += local_ss + (count * ((float(mean) - combined_mean) ** 2))
    combined_std = (total_ss / (total_count - 1)) ** 0.5 if total_ss > 0 else 0.0
    return combined_mean, combined_std


def _non_empty_text_bounds(values: list[object] | tuple[object, ...]) -> tuple[str | None, str | None]:
    normalized = sorted(str(value).strip() for value in values if str(value).strip())
    if not normalized:
        return None, None
    return normalized[0], normalized[-1]


def _aggregate_quality_summary_rows(model_key: str, rows: list[dict[str, Any]], computed_at: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    normalized_rows = [dict(row) for row in rows]
    dated_rows = [
        row for row in normalized_rows
        if _as_text(row.get("profile_date")).strip()
    ]
    if not dated_rows:
        return []
    total_rows = sum(int(row.get("row_count", 0) or 0) for row in dated_rows)
    if total_rows <= 0:
        return []
    profile_dates = sorted(_as_text(row.get("profile_date")).strip() for row in dated_rows if _as_text(row.get("profile_date")).strip())
    daily_volume = {
        _as_text(row.get("profile_date")).strip(): int(row.get("row_count", 0) or 0)
        for row in sorted(dated_rows, key=lambda item: _as_text(item.get("profile_date")).strip())
    }
    null_totals: dict[str, float] = {}
    for row in dated_rows:
        row_count = int(row.get("row_count", 0) or 0)
        if row_count <= 0:
            continue
        for feature_name, null_pct in _safe_json_dict(row.get("null_rates")).items():
            null_value = _as_float(null_pct)
            if null_value is None:
                continue
            null_totals[str(feature_name)] = null_totals.get(str(feature_name), 0.0) + (null_value * row_count)
    null_rates = {
        feature_name: round(weighted_total / total_rows, 2)
        for feature_name, weighted_total in sorted(null_totals.items())
    }
    prediction_mean, prediction_std = _combine_weighted_mean_std([
        (
            int(row.get("row_count", 0) or 0),
            _as_float(row.get("prediction_mean")),
            _as_float(row.get("prediction_std")),
        )
        for row in dated_rows
    ])
    return [{
        "model_key": model_key,
        "total_rows": total_rows,
        "min_date": profile_dates[0],
        "max_date": profile_dates[-1],
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
        "daily_volume": json.dumps(daily_volume),
        "null_rates": json.dumps(null_rates),
        "computed_at": computed_at,
    }]


def _is_field_already_exists_error(error: Exception) -> bool:
    message = str(error).upper()
    return "FIELD_ALREADY_EXISTS" in message or "ALREADY EXISTS" in message


def _show_pattern_literal(value: str) -> str:
    return re.escape(value).replace("'", "''")


def _resolve_source_labels_join_col(
    source_columns: list[str] | tuple[str, ...],
    entity_id_col: str | None,
    labels_join_col: str | None,
) -> str | None:
    normalized_columns = {str(column).strip() for column in source_columns if str(column).strip()}
    shared_join_col = _as_text(labels_join_col).strip()
    if shared_join_col and shared_join_col in normalized_columns:
        return validate_identifier(shared_join_col)
    entity_join_col = _as_text(entity_id_col).strip()
    if entity_join_col and entity_join_col in normalized_columns:
        return validate_identifier(entity_join_col)
    return None


def _monitor_status_filter(status: str | list[str] | tuple[str, ...] | None) -> tuple[str, tuple[object, ...]]:
    if status in (None, "", "all"):
        return "", ()
    if isinstance(status, str):
        return " WHERE status = %s", (status,)
    statuses = tuple(str(value).strip().lower() for value in status if str(value).strip())
    if not statuses:
        return "", ()
    placeholders = ", ".join(["%s"] * len(statuses))
    return f" WHERE status IN ({placeholders})", statuses


def _normalized_status_values(status: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if status in (None, "", "all"):
        return ()
    if isinstance(status, str):
        normalized = str(status).strip().lower()
        return (normalized,) if normalized else ()
    values = tuple(str(value).strip().lower() for value in status if str(value).strip())
    return tuple(dict.fromkeys(values))


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

    def fork_for_worker(self) -> ControlPlaneRepository:
        warehouse = WarehouseConnection(
            warehouse_id=getattr(self._warehouse, "_warehouse_id", ""),
            host=getattr(self._warehouse, "_host", ""),
        )
        return ControlPlaneRepository(
            warehouse=warehouse,
            table_names=self._table_names,
            read_model=None,
        )

    def ensure_control_plane(self, *, create_catalog: bool = False) -> None:
        if create_catalog:
            self._warehouse.execute(f"CREATE CATALOG IF NOT EXISTS {self._table_names.catalog}")
        self._ensure_schema_exists()
        for table_name, statement in self._ddl_statements().items():
            self._ensure_table_exists(table_name, statement)
        self._ensure_monitor_config_columns()
        self._ensure_refresh_run_columns()
        self._backfill_legacy_published_refresh_runs()
        self._ensure_runtime_state_columns()
        self._ensure_drift_metric_columns()
        self._ensure_performance_metric_columns()
        self._ensure_quality_metric_columns()
        self._ensure_quality_history_columns()
        self._ensure_performance_bin_spec_columns()
        self._ensure_incident_columns()
        self._ensure_incident_history_columns()
        self._ensure_daily_quality_profile_columns()
        self._ensure_daily_class_quality_profile_columns()
        self._ensure_daily_feature_profile_columns()
        self._ensure_daily_class_feature_profile_columns()
        self._ensure_daily_performance_profile_columns()
        self._ensure_daily_label_metric_columns()
        self._ensure_comparison_window_columns()
        self._queue_missing_daily_label_metric_backfills()
        if self._read_model and self._read_model.configured:
            self._read_model.ensure_schema()

    def _ddl_statements(self) -> dict[str, str]:
        return {
            getattr(self._table_names, logical_name): statement
            for logical_name, statement in ddl(self._table_names).items()
        }

    def _schema_exists(self) -> bool:
        schema_name = _show_pattern_literal(self._table_names.schema)
        try:
            schemas = self._warehouse.query(
                f"SHOW SCHEMAS IN {self._table_names.catalog} LIKE '{schema_name}'"
            )
        except Exception:
            return False
        return not schemas.empty

    def _table_exists(self, table_name: str) -> bool:
        table_basename = _show_pattern_literal(table_name.rsplit(".", 1)[-1])
        try:
            tables = self._warehouse.query(
                f"SHOW TABLES IN {self._table_names.namespace} LIKE '{table_basename}'"
            )
        except Exception:
            return False
        return not tables.empty

    def _ensure_schema_exists(self) -> None:
        if self._schema_exists():
            return
        statement = f"CREATE SCHEMA IF NOT EXISTS {self._table_names.namespace}"
        try:
            self._warehouse.execute(statement)
        except Exception:
            if self._schema_exists():
                return
            raise

    def _ensure_table_exists(self, table_name: str, create_statement: str) -> None:
        if self._table_exists(table_name):
            return
        try:
            self._warehouse.execute(create_statement)
        except Exception:
            if self._table_exists(table_name):
                return
            raise

    def _describe_table(self, table_name: str, *, cache: bool = True) -> pd.DataFrame:
        describe = getattr(self._warehouse, "describe_table")
        try:
            return describe(table_name, cache=cache)
        except TypeError:
            return describe(table_name)

    def _existing_table_columns(self, table_name: str, *, cache: bool = True) -> set[str]:
        schema = self._describe_table(table_name, cache=cache)
        return {
            str(value).strip().lower()
            for value in schema.get("col_name", pd.Series(dtype=str)).tolist()
            if str(value).strip()
        }

    def _get_table_columns(self, table_name: str, *, cache: bool = True) -> list[str]:
        get_columns = getattr(self._warehouse, "get_columns")
        try:
            return [str(value) for value in get_columns(table_name, cache=cache)]
        except TypeError:
            return [str(value) for value in get_columns(table_name)]

    def _ensure_table_columns(self, table_name: str, migration_columns: dict[str, str]) -> None:
        try:
            existing = self._existing_table_columns(table_name, cache=False)
        except Exception:
            return
        for column_name, data_type in migration_columns.items():
            normalized_name = column_name.strip().lower()
            if normalized_name in existing:
                continue
            try:
                existing = self._existing_table_columns(table_name, cache=False)
            except Exception:
                pass
            if normalized_name in existing:
                continue
            try:
                self._warehouse.execute(
                    f"ALTER TABLE {table_name} "
                    f"ADD COLUMNS ({validate_identifier(column_name)} {data_type})"
                )
            except Exception as error:
                try:
                    existing = self._existing_table_columns(table_name, cache=False)
                except Exception:
                    pass
                if normalized_name in existing or _is_field_already_exists_error(error):
                    existing.add(normalized_name)
                    continue
                raise
            existing.add(normalized_name)

    def _ensure_monitor_config_columns(self) -> None:
        self._ensure_table_columns(self._table_names.monitor_configs, monitor_config_migration_columns())

    def _ensure_refresh_run_columns(self) -> None:
        self._ensure_table_columns(self._table_names.refresh_runs, refresh_run_migration_columns())

    def _backfill_legacy_published_refresh_runs(self) -> None:
        try:
            self._warehouse.execute(
                f"""
                UPDATE {self._table_names.refresh_runs}
                SET
                    generation_id = CASE
                        WHEN generation_id IS NULL OR generation_id = '' THEN run_id
                        ELSE generation_id
                    END,
                    published_at = CASE
                        WHEN published_at IS NULL THEN completed_at
                        ELSE published_at
                    END
                WHERE status = 'completed'
                  AND completed_at IS NOT NULL
                  AND (
                      generation_id IS NULL
                      OR generation_id = ''
                      OR published_at IS NULL
                  )
                """
            )
        except Exception:
            # Old workspaces should still come up even if this additive backfill is not permitted.
            return

    def _ensure_runtime_state_columns(self) -> None:
        self._ensure_table_columns(self._table_names.monitor_runtime_state, runtime_state_migration_columns())

    def _ensure_drift_metric_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.drift_metrics,
            drift_metric_migration_columns(),
        )

    def _ensure_performance_metric_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.performance_metrics,
            performance_metric_migration_columns(),
        )

    def _ensure_daily_performance_profile_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.daily_performance_profiles,
            daily_performance_profile_migration_columns(),
        )

    def _ensure_daily_quality_profile_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.daily_quality_profiles,
            daily_quality_profile_migration_columns(),
        )

    def _ensure_daily_class_quality_profile_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.daily_class_quality_profiles,
            daily_class_quality_profile_migration_columns(),
        )

    def _ensure_daily_feature_profile_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.daily_feature_profiles,
            daily_feature_profile_migration_columns(),
        )

    def _ensure_daily_class_feature_profile_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.daily_class_feature_profiles,
            daily_class_feature_profile_migration_columns(),
        )

    def _ensure_daily_label_metric_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.daily_label_metrics,
            daily_label_metric_migration_columns(),
        )

    def _ensure_comparison_window_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.comparison_windows,
            comparison_window_migration_columns(),
        )

    def _monitor_needs_daily_label_metric_backfill(self, model_key: str) -> bool:
        generation_id = self.get_latest_published_generation_id(model_key)
        performance_filters = ["model_key = %s"]
        performance_params: list[object] = [model_key]
        daily_label_filters = ["model_key = %s"]
        daily_label_params: list[object] = [model_key]
        if generation_id:
            performance_filters.append("source_run_id = %s")
            performance_params.append(generation_id)
            daily_label_filters.append("source_run_id = %s")
            daily_label_params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT
                CASE WHEN EXISTS (
                    SELECT 1
                    FROM {self._table_names.performance_metrics}
                    WHERE {' AND '.join(performance_filters)}
                    LIMIT 1
                ) THEN 1 ELSE 0 END AS has_performance_rows,
                CASE WHEN EXISTS (
                    SELECT 1
                    FROM {self._table_names.daily_label_metrics}
                    WHERE {' AND '.join(daily_label_filters)}
                    LIMIT 1
                ) THEN 1 ELSE 0 END AS has_daily_label_rows
            """,
            tuple([*performance_params, *daily_label_params]),
        )
        if frame.empty or not {"has_performance_rows", "has_daily_label_rows"}.issubset(frame.columns):
            return False
        row = frame.iloc[0]
        has_performance_rows = bool(int(row.get("has_performance_rows", 0) or 0))
        has_daily_label_rows = bool(int(row.get("has_daily_label_rows", 0) or 0))
        return has_performance_rows and not has_daily_label_rows

    def needs_daily_label_metric_backfill(self, model_key: str) -> bool:
        return self._monitor_needs_daily_label_metric_backfill(model_key)

    def _queue_missing_daily_label_metric_backfills(self) -> None:
        try:
            configs = self.list_monitor_configs(status="active")
        except Exception:
            return
        if not configs:
            return
        queued_at = pd.Timestamp.now(tz=timezone.utc).isoformat()
        for config in configs:
            if (
                not config.schedule_enabled
                or not config.has_labels
                or config.performance_cadence_preset in {"disabled", "manual"}
            ):
                continue
            try:
                needs_backfill = self._monitor_needs_daily_label_metric_backfill(config.model_key)
            except Exception:
                continue
            if not needs_backfill:
                continue
            existing = self.get_monitor_runtime_state(config.model_key)
            self.upsert_monitor_runtime_state(
                MonitorRuntimeState(
                    model_key=config.model_key,
                    bootstrap_status=(
                        existing.bootstrap_status
                        if existing and existing.bootstrap_status
                        else "completed"
                    ),
                    last_drift_refresh_at=existing.last_drift_refresh_at if existing else None,
                    last_performance_refresh_at=None,
                    next_drift_due_at=existing.next_drift_due_at if existing else None,
                    next_performance_due_at=queued_at,
                    last_label_watermark=None,
                    last_run_status=existing.last_run_status if existing else None,
                    last_run_error="Queued performance repair to populate daily labeled facts after schema migration.",
                    last_run_started_at=existing.last_run_started_at if existing else None,
                    last_run_completed_at=existing.last_run_completed_at if existing else None,
                    backoff_until=None,
                    consecutive_failures=0,
                )
            )

    def _ensure_quality_metric_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.quality_metrics,
            quality_metric_migration_columns(),
        )

    def _ensure_quality_history_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.quality_history,
            quality_history_migration_columns(),
        )

    def _ensure_performance_bin_spec_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.performance_bin_specs,
            performance_bin_spec_migration_columns(),
        )

    def _ensure_incident_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.incidents,
            incident_migration_columns(),
        )

    def _ensure_incident_history_columns(self) -> None:
        self._ensure_table_columns(
            self._table_names.incident_history,
            incident_history_migration_columns(),
        )

    def scan_source_table(self, table_name: str, preview_rows: int = 5) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
        validate_identifier(table_name)
        preview_rows = max(1, min(preview_rows, 20))
        schema = self._describe_table(table_name)
        columns = [str(value) for value in schema.get("col_name", pd.Series(dtype=str)).tolist()]
        preview = self._warehouse.query(f"SELECT * FROM {table_name} LIMIT {preview_rows}")
        return columns, preview, schema

    def sample_bounded_rows(
        self,
        table_name: str,
        columns: list[str] | tuple[str, ...],
        *,
        max_total_rows: int = 2000,
    ) -> pd.DataFrame:
        validate_identifier(table_name)
        selected_columns = [
            validate_identifier(str(column).strip())
            for column in dict.fromkeys(columns)
            if str(column).strip()
        ]
        if not selected_columns or max_total_rows <= 0:
            return pd.DataFrame(columns=selected_columns)
        projection_sql = ", ".join(quote_column(column) for column in selected_columns)
        hash_order_terms = ", ".join(
            f"COALESCE(CAST({quote_column(column)} AS STRING), '')"
            for column in selected_columns
        )
        return self._warehouse.query(
            f"""
            SELECT {projection_sql}
            FROM {table_name}
            ORDER BY xxhash64({hash_order_terms})
            LIMIT {int(max_total_rows)}
            """
        )

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

    def _source_filters(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        alias: str = "s",
    ) -> tuple[list[str], list[object]]:
        prefix = f"{alias}." if alias else ""
        filters: list[str] = []
        params: list[object] = []
        if config.model_id_value and config.contract.model_id_col:
            filters.append(f"{prefix}{quote_column(validate_identifier(config.contract.model_id_col))} = %s")
            params.append(config.model_id_value)
        if config.model_version_value and config.contract.model_version_col:
            filters.append(f"{prefix}{quote_column(validate_identifier(config.contract.model_version_col))} = %s")
            params.append(config.model_version_value)
        if start_date:
            filters.append(f"CAST({prefix}{quote_column(validate_identifier(config.contract.timestamp_col))} AS DATE) >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append(f"CAST({prefix}{quote_column(validate_identifier(config.contract.timestamp_col))} AS DATE) <= CAST(%s AS DATE)")
            params.append(end_date)
        return filters, params

    def profile_labels_mapping(
        self,
        *,
        source_table: str,
        source_join_col: str,
        labels_table: str,
        labels_join_col: str,
        label_col: str,
        labels_order_col: str | None = None,
    ) -> dict[str, Any]:
        validate_identifier(source_table)
        validate_identifier(labels_table)
        source_join_col = validate_identifier(source_join_col)
        labels_join_col = validate_identifier(labels_join_col)
        label_col = validate_identifier(label_col)
        order_col = validate_identifier(labels_order_col) if labels_order_col else None

        duplicate_keys = self._warehouse.query(
            f"""
            SELECT COUNT(*) AS duplicate_key_count
            FROM (
                SELECT {quote_column(labels_join_col)}
                FROM {labels_table}
                GROUP BY {quote_column(labels_join_col)}
                HAVING COUNT(*) > 1
            ) duplicate_keys
            """
        )
        duplicate_key_count = 0 if duplicate_keys.empty else int(duplicate_keys.iloc[0]["duplicate_key_count"] or 0)

        join_presence = self._warehouse.query(
            f"""
            SELECT
                COUNT(*) AS inference_rows,
                SUM(CASE WHEN l.{quote_column(labels_join_col)} IS NOT NULL THEN 1 ELSE 0 END) AS matched_rows,
                SUM(CASE WHEN l.{quote_column(labels_join_col)} IS NULL THEN 1 ELSE 0 END) AS unmatched_rows
            FROM {source_table} s
            LEFT JOIN (
                SELECT DISTINCT {quote_column(labels_join_col)}
                FROM {labels_table}
                WHERE {quote_column(labels_join_col)} IS NOT NULL
            ) l
                ON s.{quote_column(source_join_col)} = l.{quote_column(labels_join_col)}
            """
        )
        if join_presence.empty:
            inference_rows = matched_rows = unmatched_rows = 0
        else:
            row = join_presence.iloc[0]
            inference_rows = int(row.get("inference_rows", 0) or 0)
            matched_rows = int(row.get("matched_rows", 0) or 0)
            unmatched_rows = int(row.get("unmatched_rows", 0) or 0)

        label_values_frame = self._warehouse.query(
            f"""
            SELECT DISTINCT CAST({quote_column(label_col)} AS STRING) AS label_value
            FROM {labels_table}
            WHERE {quote_column(label_col)} IS NOT NULL
            LIMIT 10
            """
        )
        distinct_label_values: list[str] = []
        if not label_values_frame.empty and "label_value" in label_values_frame.columns:
            for value in label_values_frame["label_value"].tolist():
                text = str(value).strip()
                if text and text not in distinct_label_values:
                    distinct_label_values.append(text)

        binary_compatible = bool(distinct_label_values) and set(distinct_label_values).issubset({"0", "1"})
        match_rate_pct = round((matched_rows / inference_rows) * 100, 2) if inference_rows else 0.0

        return {
            "inference_rows": inference_rows,
            "matched_rows": matched_rows,
            "unmatched_rows": unmatched_rows,
            "match_rate_pct": match_rate_pct,
            "duplicate_join_keys": duplicate_key_count,
            "distinct_label_values": tuple(distinct_label_values),
            "binary_compatible": binary_compatible,
            "order_column": order_col or "",
        }

    def upsert_monitor_config(self, config: MonitorConfig) -> None:
        now = pd.Timestamp.utcnow().isoformat()
        feature_columns = array_literal(list(config.contract.feature_columns))
        slice_columns = array_literal(list(config.contract.slice_columns))
        categorical_columns = array_literal(list(config.contract.categorical_columns))
        performance_metric_names = array_literal(list(config.performance_metric_names))
        self._warehouse.execute_params(
            f"""
            MERGE INTO {self._table_names.monitor_configs} AS target
            USING (
                SELECT
                    %s AS model_key,
                    %s AS display_name,
                    %s AS source_table,
                    %s AS timestamp_col,
                    %s AS model_id_col,
                    %s AS model_id_value,
                    %s AS prediction_col,
                    %s AS model_version_col,
                    %s AS model_version_value,
                    %s AS prediction_score_col,
                    %s AS label_col,
                    %s AS entity_id_col,
                    {feature_columns} AS feature_columns,
                    {slice_columns} AS slice_columns,
                    {categorical_columns} AS categorical_columns,
                    %s AS baseline_kind,
                    %s AS baseline_n_days,
                    CAST(%s AS DATE) AS baseline_start,
                    CAST(%s AS DATE) AS baseline_end,
                    %s AS baseline_max_comparison_days,
                    %s AS problem_type,
                    %s AS labels_table,
                    %s AS labels_join_col,
                    %s AS labels_order_col,
                    {performance_metric_names} AS performance_metric_names,
                    %s AS default_performance_metric,
                    %s AS performance_binning_mode,
                    %s AS performance_binning_clip_percentile,
                    %s AS drift_cadence_preset,
                    %s AS performance_cadence_preset,
                    %s AS schedule_enabled,
                    CAST(%s AS STRING) AS threshold_overrides,
                    %s AS mlflow_experiment_name,
                    %s AS mlflow_experiment_id,
                    %s AS mlflow_run_id,
                    %s AS mlflow_registered_model_name,
                    %s AS mlflow_model_version,
                    %s AS created_by,
                    %s AS status,
                    CAST(%s AS TIMESTAMP) AS created_at,
                    CAST(%s AS TIMESTAMP) AS updated_at
            ) AS source
            ON target.model_key = source.model_key
            WHEN MATCHED THEN UPDATE SET
                display_name = source.display_name,
                source_table = source.source_table,
                timestamp_col = source.timestamp_col,
                model_id_col = source.model_id_col,
                model_id_value = source.model_id_value,
                prediction_col = source.prediction_col,
                model_version_col = source.model_version_col,
                model_version_value = source.model_version_value,
                prediction_score_col = source.prediction_score_col,
                label_col = source.label_col,
                entity_id_col = source.entity_id_col,
                feature_columns = source.feature_columns,
                slice_columns = source.slice_columns,
                categorical_columns = source.categorical_columns,
                baseline_kind = source.baseline_kind,
                baseline_n_days = source.baseline_n_days,
                baseline_start = source.baseline_start,
                baseline_end = source.baseline_end,
                baseline_max_comparison_days = source.baseline_max_comparison_days,
                problem_type = source.problem_type,
                labels_table = source.labels_table,
                labels_join_col = source.labels_join_col,
                labels_order_col = source.labels_order_col,
                performance_metric_names = source.performance_metric_names,
                default_performance_metric = source.default_performance_metric,
                performance_binning_mode = source.performance_binning_mode,
                performance_binning_clip_percentile = source.performance_binning_clip_percentile,
                drift_cadence_preset = source.drift_cadence_preset,
                performance_cadence_preset = source.performance_cadence_preset,
                schedule_enabled = source.schedule_enabled,
                threshold_overrides = source.threshold_overrides,
                mlflow_experiment_name = source.mlflow_experiment_name,
                mlflow_experiment_id = source.mlflow_experiment_id,
                mlflow_run_id = source.mlflow_run_id,
                mlflow_registered_model_name = source.mlflow_registered_model_name,
                mlflow_model_version = source.mlflow_model_version,
                created_by = source.created_by,
                status = source.status,
                updated_at = source.updated_at
            WHEN NOT MATCHED THEN INSERT (
                model_key, display_name, source_table,
                timestamp_col, model_id_col, model_id_value, prediction_col,
                model_version_col, model_version_value, prediction_score_col, label_col, entity_id_col,
                feature_columns, slice_columns, categorical_columns,
                baseline_kind, baseline_n_days, baseline_start, baseline_end, baseline_max_comparison_days,
                problem_type, labels_table, labels_join_col, labels_order_col,
                performance_metric_names, default_performance_metric,
                performance_binning_mode, performance_binning_clip_percentile,
                drift_cadence_preset, performance_cadence_preset, schedule_enabled, threshold_overrides,
                mlflow_experiment_name, mlflow_experiment_id, mlflow_run_id,
                mlflow_registered_model_name, mlflow_model_version,
                created_by, status,
                created_at, updated_at
            ) VALUES (
                source.model_key, source.display_name, source.source_table,
                source.timestamp_col, source.model_id_col, source.model_id_value, source.prediction_col,
                source.model_version_col, source.model_version_value, source.prediction_score_col, source.label_col, source.entity_id_col,
                source.feature_columns, source.slice_columns, source.categorical_columns,
                source.baseline_kind, source.baseline_n_days, source.baseline_start, source.baseline_end, source.baseline_max_comparison_days,
                source.problem_type, source.labels_table, source.labels_join_col, source.labels_order_col,
                source.performance_metric_names, source.default_performance_metric,
                source.performance_binning_mode, source.performance_binning_clip_percentile,
                source.drift_cadence_preset, source.performance_cadence_preset, source.schedule_enabled, source.threshold_overrides,
                source.mlflow_experiment_name, source.mlflow_experiment_id, source.mlflow_run_id,
                source.mlflow_registered_model_name, source.mlflow_model_version,
                source.created_by, source.status,
                source.created_at, source.updated_at
            )
            """,
            (
                config.model_key,
                config.display_name,
                config.source_table,
                config.contract.timestamp_col,
                config.contract.model_id_col or "",
                config.model_id_value or "",
                config.contract.prediction_col,
                config.contract.model_version_col or "",
                config.model_version_value or "",
                config.contract.prediction_score_col or "",
                config.contract.label_col or "",
                config.contract.entity_id_col or "",
                config.baseline.kind,
                config.baseline.n_days,
                config.baseline.baseline_start or None,
                config.baseline.baseline_end or None,
                config.baseline.max_comparison_days,
                config.problem_type,
                config.labels_table or "",
                config.labels_join_col or "",
                config.labels_order_col or "",
                config.default_performance_metric or "",
                config.performance_binning_mode,
                config.performance_binning_clip_percentile,
                config.drift_cadence_preset,
                config.performance_cadence_preset,
                config.schedule_enabled,
                json.dumps(config.threshold_overrides or {}),
                config.mlflow.experiment_name or "",
                config.mlflow.experiment_id or "",
                config.mlflow.run_id or "",
                config.mlflow.registered_model_name or "",
                config.mlflow.model_version or "",
                config.created_by,
                config.status,
                now,
                now,
            ),
        )
        self._sync_read_model()

    def list_monitor_configs(self, status: str | list[str] | tuple[str, ...] | None = "active") -> list[MonitorConfig]:
        where_sql, params = _monitor_status_filter(status)
        query = f"SELECT * FROM {self._table_names.monitor_configs}{where_sql} ORDER BY updated_at DESC"
        if params:
            frame = self._warehouse.query_params(query, params)
        else:
            frame = self._warehouse.query(query)
        configs = [self._row_to_monitor_config(row) for _, row in frame.iterrows()]
        requested_statuses = _normalized_status_values(status)
        if configs or not requested_statuses:
            return configs

        fallback_query = f"SELECT * FROM {self._table_names.monitor_configs} ORDER BY updated_at DESC"
        fallback_frame = self._warehouse.query(fallback_query)
        fallback_configs = [self._row_to_monitor_config(row) for _, row in fallback_frame.iterrows()]
        if not fallback_configs:
            return []

        recovered = [config for config in fallback_configs if config.status in requested_statuses]
        if recovered:
            logger.warning(
                "status-filtered monitor config query returned no rows; recovered %s configs via unfiltered fallback in namespace %s",
                len(recovered),
                self._table_names.namespace,
            )
        return recovered

    def archive_monitor(self, model_key: str) -> None:
        if not self._monitor_exists(model_key):
            raise KeyError(f"Monitor {model_key!r} does not exist.")
        now = pd.Timestamp.utcnow().isoformat()
        self._warehouse.execute_params(
            f"""
            UPDATE {self._table_names.monitor_configs}
            SET status = 'inactive', updated_at = CAST(%s AS TIMESTAMP)
            WHERE model_key = %s
            """,
            (now, model_key),
        )
        self._sync_read_model()

    def restore_monitor(self, model_key: str) -> None:
        if not self._monitor_exists(model_key):
            raise KeyError(f"Monitor {model_key!r} does not exist.")
        now = pd.Timestamp.utcnow().isoformat()
        self._warehouse.execute_params(
            f"""
            UPDATE {self._table_names.monitor_configs}
            SET status = 'active', updated_at = CAST(%s AS TIMESTAMP)
            WHERE model_key = %s
            """,
            (now, model_key),
        )
        self._sync_read_model()

    def delete_monitor(self, model_key: str) -> None:
        if not self._monitor_exists(model_key):
            raise KeyError(f"Monitor {model_key!r} does not exist.")
        if str(getattr(self._table_names, "catalog", "") or "").strip().lower() == "hive_metastore":
            raise PermanentDeleteUnsupportedError(
                "Permanent delete requires Unity Catalog control-plane tables so the cleanup can run atomically. "
                "Archive this monitor instead, or move the Model Lens control plane to a Unity Catalog managed schema before deleting."
            )
        table_names = (
            self._table_names.monitor_runtime_state,
            self._table_names.refresh_runs,
            self._table_names.comparison_windows,
            self._table_names.drift_metrics,
            self._table_names.quality_metrics,
            self._table_names.quality_history,
            self._table_names.daily_quality_profiles,
            self._table_names.daily_class_quality_profiles,
            self._table_names.daily_feature_profiles,
            self._table_names.daily_class_feature_profiles,
            self._table_names.performance_metrics,
            self._table_names.daily_performance_profiles,
            self._table_names.daily_label_metrics,
            self._table_names.performance_bin_specs,
            self._table_names.incidents,
            self._table_names.incident_history,
            self._table_names.monitor_configs,
        )
        statements = [
            f"DELETE FROM {table_name} WHERE model_key = %s;"
            for table_name in table_names
        ]
        atomic_sql = "\n".join(statements)
        try:
            self._warehouse.execute_atomic_params(atomic_sql, (model_key,) * len(table_names))
        except Exception as error:
            logger.exception("Atomic delete failed for %s", model_key, exc_info=error)
            raise RuntimeError("Atomic monitor delete failed; no partial delete was applied.") from error
        self._sync_read_model()

    def _monitor_exists(self, model_key: str) -> bool:
        frame = self._warehouse.query_params(
            f"""
            SELECT 1 AS found
            FROM {self._table_names.monitor_configs}
            WHERE model_key = %s
            LIMIT 1
            """,
            (model_key,),
        )
        return not frame.empty

    def _row_to_monitor_config(self, row: pd.Series) -> MonitorConfig:
        contract = InferenceContract(
            timestamp_col=_as_text(row.get("timestamp_col")),
            prediction_col=_as_text(row.get("prediction_col")),
            model_id_col=_as_text(row.get("model_id_col")) or None,
            model_version_col=_as_text(row.get("model_version_col")) or None,
            prediction_score_col=_as_text(row.get("prediction_score_col")) or None,
            label_col=_as_text(row.get("label_col")) or None,
            entity_id_col=_as_text(row.get("entity_id_col")) or None,
            feature_columns=parse_string_array(row.get("feature_columns")),
            slice_columns=parse_string_array(row.get("slice_columns")),
            categorical_columns=parse_string_array(row.get("categorical_columns")),
        )
        baseline = BaselinePolicy(
            kind=_as_text(row.get("baseline_kind")) or "rolling",
            n_days=int(row.get("baseline_n_days", 7) or 7),
            baseline_start=_as_text(row.get("baseline_start")) or None,
            baseline_end=_as_text(row.get("baseline_end")) or None,
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
            performance_metric_names=parse_string_array(row.get("performance_metric_names")),
            default_performance_metric=_as_text(row.get("default_performance_metric")) or None,
            performance_binning_mode=_as_text(row.get("performance_binning_mode")) or "quantile",
            performance_binning_clip_percentile=_as_float(row.get("performance_binning_clip_percentile")),
            drift_cadence_preset=_as_text(row.get("drift_cadence_preset")) or "6h",
            performance_cadence_preset=(
                _as_text(row.get("performance_cadence_preset"))
                or ("daily_7d_repair" if contract.label_col else "disabled")
            ),
            schedule_enabled=_as_bool(row.get("schedule_enabled"), default=True),
            threshold_overrides=_safe_json_dict(row.get("threshold_overrides")),
            mlflow=MLflowLineage(
                experiment_name=_as_text(row.get("mlflow_experiment_name")) or None,
                experiment_id=_as_text(row.get("mlflow_experiment_id")) or None,
                run_id=_as_text(row.get("mlflow_run_id")) or None,
                registered_model_name=_as_text(row.get("mlflow_registered_model_name")) or None,
                model_version=_as_text(row.get("mlflow_model_version")) or None,
            ),
            created_by=_as_text(row.get("created_by")) or "app",
            status=_as_text(row.get("status")) or "active",
        )

    def _row_to_runtime_state(self, row: pd.Series) -> MonitorRuntimeState:
        return MonitorRuntimeState(
            model_key=_as_text(row.get("model_key")),
            bootstrap_status=_as_text(row.get("bootstrap_status")) or "pending",
            last_drift_refresh_at=_as_text(row.get("last_drift_refresh_at")) or None,
            last_performance_refresh_at=_as_text(row.get("last_performance_refresh_at")) or None,
            next_drift_due_at=_as_text(row.get("next_drift_due_at")) or None,
            next_performance_due_at=_as_text(row.get("next_performance_due_at")) or None,
            last_label_watermark=_as_text(row.get("last_label_watermark")) or None,
            last_run_status=_as_text(row.get("last_run_status")) or None,
            last_run_error=_as_text(row.get("last_run_error")) or None,
            last_run_started_at=_as_text(row.get("last_run_started_at")) or None,
            last_run_completed_at=_as_text(row.get("last_run_completed_at")) or None,
            backoff_until=_as_text(row.get("backoff_until")) or None,
            consecutive_failures=int(row.get("consecutive_failures", 0) or 0),
        )

    def list_monitor_runtime_states(self, model_keys: list[str] | None = None) -> dict[str, MonitorRuntimeState]:
        if model_keys:
            placeholders = ", ".join(["%s"] * len(model_keys))
            frame = self._warehouse.query_params(
                f"""
                SELECT *
                FROM {self._table_names.monitor_runtime_state}
                WHERE model_key IN ({placeholders})
                """,
                tuple(model_keys),
            )
        else:
            frame = self._warehouse.query(f"SELECT * FROM {self._table_names.monitor_runtime_state}")
        if frame.empty:
            return {}
        return {
            state.model_key: state
            for state in (
                self._row_to_runtime_state(row)
                for _, row in frame.iterrows()
            )
        }

    def get_monitor_runtime_state(self, model_key: str) -> MonitorRuntimeState | None:
        states = self.list_monitor_runtime_states([model_key])
        return states.get(model_key)

    def upsert_monitor_runtime_state(self, state: MonitorRuntimeState) -> None:
        self._warehouse.execute_params(
            f"DELETE FROM {self._table_names.monitor_runtime_state} WHERE model_key = %s",
            (state.model_key,),
        )
        self._warehouse.execute_params(
            f"""
            INSERT INTO {self._table_names.monitor_runtime_state} (
                model_key, bootstrap_status,
                last_drift_refresh_at, last_performance_refresh_at,
                next_drift_due_at, next_performance_due_at,
                last_label_watermark, last_run_status, last_run_error,
                last_run_started_at, last_run_completed_at,
                backoff_until, consecutive_failures
            ) VALUES (
                %s, %s,
                CAST(%s AS TIMESTAMP), CAST(%s AS TIMESTAMP),
                CAST(%s AS TIMESTAMP), CAST(%s AS TIMESTAMP),
                %s, %s, %s,
                CAST(%s AS TIMESTAMP), CAST(%s AS TIMESTAMP),
                CAST(%s AS TIMESTAMP), %s
            )
            """,
            (
                state.model_key,
                state.bootstrap_status,
                state.last_drift_refresh_at,
                state.last_performance_refresh_at,
                state.next_drift_due_at,
                state.next_performance_due_at,
                state.last_label_watermark or "",
                state.last_run_status or "",
                state.last_run_error or "",
                state.last_run_started_at,
                state.last_run_completed_at,
                state.backoff_until,
                state.consecutive_failures,
            ),
        )

    def ensure_monitor_runtime_state(self, config: MonitorConfig) -> MonitorRuntimeState:
        existing = self.get_monitor_runtime_state(config.model_key)
        now = pd.Timestamp.now(tz=timezone.utc).isoformat()
        bootstrap_completed = bool(
            self.get_latest_published_generation_id(config.model_key)
            or not self._warehouse.query_params(
                f"""
                SELECT 1 AS found
                FROM {self._table_names.comparison_windows}
                WHERE model_key = %s
                LIMIT 1
                """,
                (config.model_key,),
            ).empty
        )
        state = MonitorRuntimeState(
            model_key=config.model_key,
            bootstrap_status=(
                "completed"
                if bootstrap_completed
                else (
                    existing.bootstrap_status
                    if existing and existing.bootstrap_status
                    else "pending"
                )
            ),
            last_drift_refresh_at=existing.last_drift_refresh_at if existing else None,
            last_performance_refresh_at=existing.last_performance_refresh_at if existing else None,
            next_drift_due_at=now if config.schedule_enabled else None,
            next_performance_due_at=(
                now
                if config.schedule_enabled and config.has_labels and config.performance_cadence_preset != "disabled"
                else None
            ),
            last_label_watermark=existing.last_label_watermark if existing else None,
            last_run_status=existing.last_run_status if existing else None,
            last_run_error=existing.last_run_error if existing else None,
            last_run_started_at=existing.last_run_started_at if existing else None,
            last_run_completed_at=existing.last_run_completed_at if existing else None,
            backoff_until=existing.backoff_until if existing else None,
            consecutive_failures=existing.consecutive_failures if existing else 0,
        )
        self.upsert_monitor_runtime_state(state)
        return state

    def mark_monitor_bootstrap_pending(self, config: MonitorConfig) -> MonitorRuntimeState:
        existing = self.get_monitor_runtime_state(config.model_key)
        now = pd.Timestamp.now(tz=timezone.utc).isoformat()
        state = MonitorRuntimeState(
            model_key=config.model_key,
            bootstrap_status="pending",
            last_drift_refresh_at=existing.last_drift_refresh_at if existing else None,
            last_performance_refresh_at=existing.last_performance_refresh_at if existing else None,
            next_drift_due_at=now if config.schedule_enabled else None,
            next_performance_due_at=(
                now
                if config.schedule_enabled and config.has_labels and config.performance_cadence_preset != "disabled"
                else None
            ),
            last_label_watermark=existing.last_label_watermark if existing else None,
            last_run_status=None,
            last_run_error=None,
            last_run_started_at=None,
            last_run_completed_at=None,
            backoff_until=None,
            consecutive_failures=0,
        )
        self.upsert_monitor_runtime_state(state)
        return state

    def validate_monitor_source(self, config: MonitorConfig) -> None:
        validate_identifier(config.source_table)
        if config.model_version_value and not config.contract.model_version_col:
            raise ValueError("Model Version Value requires a mapped Model Version Column.")

        if config.model_id_value and not config.contract.model_id_col:
            raise ValueError("Monitored Model ID Value requires a mapped Model ID Column.")

        if config.contract.model_id_col and not config.model_id_value:
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

        source_columns = self._warehouse.get_columns(config.source_table)
        source_join_col = _resolve_source_labels_join_col(
            source_columns,
            config.contract.entity_id_col,
            config.labels_join_col,
        )
        if not (config.contract.label_col and config.labels_join_col and source_join_col):
            raise ValueError(
                "External labels require External Label Column, External Labels Join Column, and either Entity ID Column or the same join column name in the inference table."
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
        self.validate_monitor_source(config)
        ts_col = validate_identifier(config.contract.timestamp_col)

        select_columns = [
            config.contract.timestamp_col,
            config.contract.prediction_col,
        ]
        if config.contract.model_id_col:
            select_columns.append(config.contract.model_id_col)
        optional_source_columns = [
            config.contract.model_version_col,
            config.contract.prediction_score_col,
            config.contract.entity_id_col,
        ]
        for column in optional_source_columns:
            if column:
                select_columns.append(column)
        for column in feature_columns or config.contract.feature_columns:
            select_columns.append(column)
        source_label_column = None
        if config.contract.label_col and not config.labels_table:
            source_label_column = config.contract.label_col
            select_columns.append(config.contract.label_col)
        source_columns = self._warehouse.get_columns(config.source_table) if config.labels_table else []
        source_join_col = _resolve_source_labels_join_col(
            source_columns,
            config.contract.entity_id_col,
            config.labels_join_col,
        )
        if config.labels_table and config.contract.label_col and source_join_col and config.labels_join_col:
            select_columns.append(source_join_col)

        deduped_columns = list(dict.fromkeys(select_columns))
        source_projection = ", ".join(
            f"s.{quote_column(validate_identifier(column))} AS {quote_column(validate_identifier(column))}"
            for column in deduped_columns
        )

        label_projection = ""
        join_sql = ""
        if config.labels_table and config.contract.label_col and source_join_col and config.labels_join_col:
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
                f" ON s.{quote_column(source_join_col)}"
                f" = l.{quote_column(join_col)}"
            )

        filters, params = self._source_filters(config, start_date=start_date, end_date=end_date, alias="s")
        where_sql = f" WHERE {' AND '.join(filters)}" if filters else ""

        source_from_sql = f"{config.source_table} s"
        if (sample_rows_per_day and sample_rows_per_day > 0) or (max_total_rows and max_total_rows > 0):
            sample_limit = int(sample_rows_per_day) if sample_rows_per_day and sample_rows_per_day > 0 else None
            total_limit = int(max_total_rows) if max_total_rows and max_total_rows > 0 else None
            sample_columns = ", ".join(quote_column(validate_identifier(column)) for column in deduped_columns)
            sample_filters, _ = self._source_filters(config, start_date=start_date, end_date=end_date, alias="")
            sample_where_sql = f" WHERE {' AND '.join(sample_filters)}" if sample_filters else ""
            hash_columns = [config.contract.timestamp_col, config.contract.prediction_col]
            if config.contract.model_id_col:
                hash_columns.append(config.contract.model_id_col)
            if config.contract.entity_id_col:
                hash_columns.append(config.contract.entity_id_col)
            hash_order_terms = ", ".join(
                f"COALESCE(CAST({quote_column(validate_identifier(column))} AS STRING), '')"
                for column in dict.fromkeys(hash_columns)
                if column
            )
            if sample_limit is not None:
                sample_where_limit_sql = f"WHERE {quote_column('model_lens_sample_rank')} <= {sample_limit}"
                total_limit_sql = f"ORDER BY xxhash64({hash_order_terms}) LIMIT {total_limit}" if total_limit else ""
                source_from_sql = f"""
                    (
                        SELECT {sample_columns}
                        FROM (
                            SELECT
                                {sample_columns},
                                ROW_NUMBER() OVER (
                                    PARTITION BY CAST({quote_column(ts_col)} AS DATE)
                                    ORDER BY xxhash64({hash_order_terms})
                                ) AS {quote_column("model_lens_sample_rank")}
                            FROM {config.source_table}
                            {sample_where_sql}
                        ) sampled_source
                        {sample_where_limit_sql}
                        {total_limit_sql}
                    ) s
                """
            else:
                total_limit_sql = f"ORDER BY xxhash64({hash_order_terms}) LIMIT {total_limit}" if total_limit else ""
                source_from_sql = f"""
                    (
                        SELECT {sample_columns}
                        FROM {config.source_table}
                        {sample_where_sql}
                        {total_limit_sql}
                    ) s
                """
            params = list(self._source_filters(config, start_date=start_date, end_date=end_date, alias="")[1])
            where_sql = ""

        query = f"""
            SELECT {source_projection}{label_projection}
            FROM {source_from_sql}
            {join_sql}
            {where_sql}
            ORDER BY s.{quote_column(ts_col)}
        """
        frame = self._warehouse.query_params(query, tuple(params)) if params else self._warehouse.query(query)
        if source_label_column and source_label_column not in frame.columns:
            frame[source_label_column] = pd.NA
        return frame

    def get_source_daily_label_metric_rows(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_monitor_source(config)
        if not config.contract.label_col:
            return []

        ts_col = quote_column(validate_identifier(config.contract.timestamp_col))
        prediction_expr = f"s.{quote_column(validate_identifier(config.contract.prediction_col))}"
        score_expr = None
        if config.contract.prediction_score_col:
            score_expr = f"s.{quote_column(validate_identifier(config.contract.prediction_score_col))}"

        label_expr = None
        join_sql = ""
        source_columns = self._warehouse.get_columns(config.source_table) if config.labels_table else []
        source_join_col = _resolve_source_labels_join_col(
            source_columns,
            config.contract.entity_id_col,
            config.labels_join_col,
        )
        if config.labels_table and config.contract.label_col and source_join_col and config.labels_join_col:
            validate_identifier(config.labels_table)
            label_col = validate_identifier(config.contract.label_col)
            join_col = validate_identifier(config.labels_join_col)
            label_expr = f"l.{quote_column(label_col)}"
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
                f" ON s.{quote_column(source_join_col)}"
                f" = l.{quote_column(join_col)}"
            )
        elif config.contract.label_col:
            label_expr = f"s.{quote_column(validate_identifier(config.contract.label_col))}"

        if not label_expr:
            return []

        filters, params = self._source_filters(config, start_date=start_date, end_date=end_date, alias="s")
        where_sql = f" WHERE {' AND '.join(filters)}" if filters else ""
        pred_binary_expr = _sql_binary_prediction_expression(prediction_expr, score_expression=score_expr)
        label_binary_expr = _sql_binary_indicator(label_expr)
        frame = self._warehouse.query_params(
            f"""
            WITH classified AS (
                SELECT
                    CAST(s.{ts_col} AS DATE) AS profile_date,
                    {pred_binary_expr} AS pred_binary,
                    {label_binary_expr} AS label_binary
                FROM {config.source_table} s
                {join_sql}
                {where_sql}
            )
            SELECT
                CAST(profile_date AS STRING) AS profile_date,
                SUM(CASE WHEN label_binary = 1 AND pred_binary = 1 THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN label_binary = 0 AND pred_binary = 1 THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN label_binary = 1 AND pred_binary = 0 THEN 1 ELSE 0 END) AS fn,
                SUM(CASE WHEN label_binary = 0 AND pred_binary = 0 THEN 1 ELSE 0 END) AS tn
            FROM classified
            WHERE pred_binary IS NOT NULL AND label_binary IS NOT NULL
            GROUP BY profile_date
            ORDER BY profile_date
            """,
            tuple(params),
        )
        if frame.empty:
            return []

        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            tp = int(row.get("tp") or 0)
            fp = int(row.get("fp") or 0)
            fn = int(row.get("fn") or 0)
            tn = int(row.get("tn") or 0)
            predicted_positive_count = tp + fp
            actual_positive_count = tp + fn
            total_count = tp + fp + fn + tn
            precision = (tp / predicted_positive_count) if predicted_positive_count > 0 else None
            recall = (tp / actual_positive_count) if predicted_positive_count > 0 and actual_positive_count > 0 else None
            f1 = None
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = (2.0 * precision * recall) / (precision + recall)
            accuracy = ((tp + tn) / total_count) if total_count > 0 else None
            rows.append(
                {
                    "model_key": config.model_key,
                    "profile_date": _as_text(row.get("profile_date")),
                    "actual_positive_count": actual_positive_count,
                    "actual_negative_count": fp + tn,
                    "predicted_positive_count": predicted_positive_count,
                    "predicted_negative_count": fn + tn,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": round(float(precision), 4) if precision is not None else None,
                    "recall": round(float(recall), 4) if recall is not None else None,
                    "f1": round(float(f1), 4) if f1 is not None else None,
                    "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
                }
            )
        return rows

    def get_source_daily_quality_profile_rows(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_monitor_source(config)
        if not config.contract.label_col:
            return []

        ts_col = quote_column(validate_identifier(config.contract.timestamp_col))
        prediction_expr = f"s.{quote_column(validate_identifier(config.contract.prediction_col))}"
        score_expr = None
        if config.contract.prediction_score_col:
            score_expr = f"s.{quote_column(validate_identifier(config.contract.prediction_score_col))}"

        label_expr = None
        join_sql = ""
        source_columns = self._warehouse.get_columns(config.source_table) if config.labels_table else []
        source_join_col = _resolve_source_labels_join_col(
            source_columns,
            config.contract.entity_id_col,
            config.labels_join_col,
        )
        if config.labels_table and config.contract.label_col and source_join_col and config.labels_join_col:
            validate_identifier(config.labels_table)
            label_col = validate_identifier(config.contract.label_col)
            join_col = validate_identifier(config.labels_join_col)
            label_expr = f"l.{quote_column(label_col)}"
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
                f" ON s.{quote_column(source_join_col)}"
                f" = l.{quote_column(join_col)}"
            )
        elif config.contract.label_col:
            label_expr = f"s.{quote_column(validate_identifier(config.contract.label_col))}"

        if not label_expr:
            return []

        normalized_basis = str(class_basis or "").strip().lower()
        normalized_value = str(class_value or "").strip().lower()
        if normalized_basis not in {"actual", "predicted"} or normalized_value not in {"positive", "negative"}:
            return []

        filters, params = self._source_filters(config, start_date=start_date, end_date=end_date, alias="s")
        where_sql = f" WHERE {' AND '.join(filters)}" if filters else ""
        pred_binary_expr = _sql_binary_prediction_expression(prediction_expr, score_expression=score_expr)
        label_binary_expr = _sql_binary_indicator(label_expr)
        class_filter_column = "label_binary" if normalized_basis == "actual" else "pred_binary"
        class_filter_value = 1 if normalized_value == "positive" else 0
        prediction_value_expr = f"CAST({prediction_expr} AS DOUBLE)"
        feature_select = ", ".join(
            f"s.{quote_column(validate_identifier(feature))} AS {quote_column(validate_identifier(feature))}"
            for feature in config.contract.feature_columns
        )
        feature_projection = f", {feature_select}" if feature_select else ""
        null_rate_sql = ", ".join(
            f"ROUND(AVG(CASE WHEN {quote_column(validate_identifier(feature))} IS NULL THEN 100.0 ELSE 0.0 END), 2) AS {quote_column(validate_identifier(feature))}"
            for feature in config.contract.feature_columns
        )
        null_rate_projection = f", {null_rate_sql}" if null_rate_sql else ""
        frame = self._warehouse.query_params(
            f"""
            WITH classified AS (
                SELECT
                    CAST(s.{ts_col} AS DATE) AS profile_date,
                    {prediction_value_expr} AS prediction_value,
                    {pred_binary_expr} AS pred_binary,
                    {label_binary_expr} AS label_binary
                    {feature_projection}
                FROM {config.source_table} s
                {join_sql}
                {where_sql}
            )
            SELECT
                CAST(profile_date AS STRING) AS profile_date,
                COUNT(*) AS row_count,
                AVG(prediction_value) AS prediction_mean,
                STDDEV_SAMP(prediction_value) AS prediction_std,
                SUM(CASE WHEN label_binary IS NOT NULL THEN 1 ELSE 0 END) AS label_row_count
                {null_rate_projection}
            FROM classified
            WHERE {class_filter_column} = {class_filter_value}
            GROUP BY profile_date
            ORDER BY profile_date
            """,
            tuple(params),
        )
        if frame.empty:
            return []

        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            null_rates = {
                feature: round(float(pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0] or 0.0), 2)
                for feature in config.contract.feature_columns
            }
            rows.append(
                {
                    "model_key": config.model_key,
                    "profile_date": _as_text(row.get("profile_date")),
                    "row_count": int(row.get("row_count", 0) or 0),
                    "prediction_mean": _as_float(row.get("prediction_mean")),
                    "prediction_std": _as_float(row.get("prediction_std")),
                    "null_rates": json.dumps(null_rates),
                    "label_row_count": int(row.get("label_row_count", 0) or 0),
                }
            )
        return rows

    def get_source_profile(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        self.validate_monitor_source(config)
        filters, params = self._source_filters(config, start_date=start_date, end_date=end_date, alias="s")
        where_sql = f" WHERE {' AND '.join(filters)}" if filters else ""
        ts_col = quote_column(validate_identifier(config.contract.timestamp_col))
        prediction_col = quote_column(validate_identifier(config.contract.prediction_col))
        label_count_sql = "0 AS label_row_count"
        join_sql = ""
        if config.labels_table and config.contract.label_col:
            source_columns = self._warehouse.get_columns(config.source_table)
            source_join_col = _resolve_source_labels_join_col(
                source_columns,
                config.contract.entity_id_col,
                config.labels_join_col,
            )
            if source_join_col and config.labels_join_col:
                validate_identifier(config.labels_table)
                label_col = validate_identifier(config.contract.label_col)
                join_col = validate_identifier(config.labels_join_col)
                label_expr = f"l.{quote_column(label_col)}"
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
                    f" ON s.{quote_column(source_join_col)}"
                    f" = l.{quote_column(join_col)}"
                )
                label_count_sql = (
                    f"SUM(CASE WHEN {label_expr} IS NOT NULL THEN 1 ELSE 0 END) "
                    "AS label_row_count"
                )
        elif config.contract.label_col:
            label_count_sql = (
                f"SUM(CASE WHEN s.{quote_column(validate_identifier(config.contract.label_col))} IS NOT NULL THEN 1 ELSE 0 END) "
                "AS label_row_count"
            )

        summary = self._warehouse.query_params(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                CAST(MIN(s.{ts_col}) AS STRING) AS min_ts,
                CAST(MAX(s.{ts_col}) AS STRING) AS max_ts,
                AVG(CAST(s.{prediction_col} AS DOUBLE)) AS prediction_mean,
                STDDEV_SAMP(CAST(s.{prediction_col} AS DOUBLE)) AS prediction_std,
                {label_count_sql}
            FROM {config.source_table} s
            {join_sql}
            {where_sql}
            """,
            tuple(params),
        ) if params else self._warehouse.query(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                CAST(MIN(s.{ts_col}) AS STRING) AS min_ts,
                CAST(MAX(s.{ts_col}) AS STRING) AS max_ts,
                AVG(CAST(s.{prediction_col} AS DOUBLE)) AS prediction_mean,
                STDDEV_SAMP(CAST(s.{prediction_col} AS DOUBLE)) AS prediction_std,
                {label_count_sql}
            FROM {config.source_table} s
            {join_sql}
            {where_sql}
            """
        )
        if summary.empty:
            return {
                "total_rows": 0,
                "min_date": None,
                "max_date": None,
                "prediction_mean": None,
                "prediction_std": None,
                "daily_volume": {},
                "null_rates": {},
                "label_row_count": 0,
            }

        summary_row = summary.iloc[0]
        total_rows = int(summary_row.get("total_rows", 0) or 0)
        min_date = _as_text(summary_row.get("min_ts") or "").split("T", 1)[0] or None
        max_date = _as_text(summary_row.get("max_ts") or "").split("T", 1)[0] or None

        daily_volume_query = f"""
            SELECT CAST(s.{ts_col} AS DATE) AS day_key, COUNT(*) AS row_count
            FROM {config.source_table} s
            {where_sql}
            GROUP BY CAST(s.{ts_col} AS DATE)
            ORDER BY day_key
        """
        daily_volume_frame = self._warehouse.query_params(daily_volume_query, tuple(params)) if params else self._warehouse.query(daily_volume_query)
        daily_volume = {
            str(row["day_key"]): int(row["row_count"] or 0)
            for _, row in daily_volume_frame.iterrows()
        } if not daily_volume_frame.empty else {}

        null_rates: dict[str, float] = {}
        if config.contract.feature_columns:
            null_rate_select = ", ".join(
                f"ROUND(AVG(CASE WHEN s.{quote_column(validate_identifier(feature))} IS NULL THEN 100.0 ELSE 0.0 END), 2) AS {quote_column(validate_identifier(feature))}"
                for feature in config.contract.feature_columns
            )
            null_rate_query = f"""
                SELECT {null_rate_select}
                FROM {config.source_table} s
                {where_sql}
            """
            null_rate_frame = self._warehouse.query_params(null_rate_query, tuple(params)) if params else self._warehouse.query(null_rate_query)
            if not null_rate_frame.empty:
                null_row = null_rate_frame.iloc[0].to_dict()
                null_rates = {
                    feature: round(float(pd.to_numeric(pd.Series([null_row.get(feature)]), errors="coerce").iloc[0] or 0.0), 2)
                    for feature in config.contract.feature_columns
                }

        return {
            "total_rows": total_rows,
            "min_date": min_date,
            "max_date": max_date,
            "prediction_mean": summary_row.get("prediction_mean"),
            "prediction_std": summary_row.get("prediction_std"),
            "daily_volume": daily_volume,
            "null_rates": null_rates,
            "label_row_count": int(summary_row.get("label_row_count", 0) or 0),
        }

    def get_source_date_range(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[str | None, str | None]:
        self.validate_monitor_source(config)
        ts_col = validate_identifier(config.contract.timestamp_col)
        filters: list[str] = []
        params: list[object] = []
        if config.model_id_value and config.contract.model_id_col:
            filters.append(f"{quote_column(validate_identifier(config.contract.model_id_col))} = %s")
            params.append(config.model_id_value)
        if config.model_version_value and config.contract.model_version_col:
            filters.append(f"{quote_column(validate_identifier(config.contract.model_version_col))} = %s")
            params.append(config.model_version_value)
        if start_date:
            filters.append(f"CAST({quote_column(ts_col)} AS DATE) >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append(f"CAST({quote_column(ts_col)} AS DATE) <= CAST(%s AS DATE)")
            params.append(end_date)
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"""
            SELECT
                CAST(MIN({quote_column(ts_col)}) AS STRING) AS min_ts,
                CAST(MAX({quote_column(ts_col)}) AS STRING) AS max_ts
            FROM {config.source_table}
            {where_sql}
        """
        frame = self._warehouse.query_params(query, tuple(params)) if params else self._warehouse.query(query)
        if frame.empty:
            return None, None
        min_ts = _as_text(frame.iloc[0].get("min_ts")) or None
        max_ts = _as_text(frame.iloc[0].get("max_ts")) or None
        min_date = min_ts.split("T", 1)[0] if min_ts else None
        max_date = max_ts.split("T", 1)[0] if max_ts else None
        return min_date, max_date

    def get_label_watermark(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str | None:
        if config.labels_table and config.labels_order_col:
            frame = self._warehouse.query(
                f"""
                SELECT CAST(MAX({quote_column(validate_identifier(config.labels_order_col))}) AS STRING) AS watermark
                FROM {config.labels_table}
                """
            )
        elif config.contract.label_col and config.contract.label_col in self._warehouse.get_columns(config.source_table):
            filters, params = self._source_filters(
                config,
                start_date=start_date,
                end_date=end_date,
                alias="",
            )
            filters.append(f"{quote_column(validate_identifier(config.contract.label_col))} IS NOT NULL")
            where_sql = f"WHERE {' AND '.join(filters)}"
            query = f"""
                SELECT
                    CAST(MAX({quote_column(validate_identifier(config.contract.timestamp_col))}) AS STRING) AS watermark,
                    COUNT(*) AS label_count
                FROM {config.source_table}
                {where_sql}
            """
            frame = self._warehouse.query_params(query, tuple(params)) if params else self._warehouse.query(query)
            if frame.empty:
                return None
            label_count = int(frame.iloc[0].get("label_count", 0) or 0)
            if label_count <= 0:
                return None
            watermark = _as_text(frame.iloc[0].get("watermark")) or ""
            return f"{watermark}|{label_count}"
        else:
            return None
        if frame.empty:
            return None
        return _as_text(frame.iloc[0].get("watermark")) or None

    def get_existing_window_keys(self, model_key: str) -> set[tuple[str, str, str, str]]:
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return set()
        comparison_filters = ["model_key = %s"]
        comparison_params: list[object] = [model_key]
        comparison_filters.append("source_run_id = %s")
        comparison_params.append(generation_id)
        try:
            frame = self._warehouse.query_params(
                f"""
                SELECT DISTINCT
                    CAST(baseline_start AS STRING) AS baseline_start,
                    CAST(baseline_end AS STRING) AS baseline_end,
                    CAST(window_start AS STRING) AS window_start,
                    CAST(window_end AS STRING) AS window_end
                FROM {self._table_names.comparison_windows}
                WHERE {' AND '.join(comparison_filters)}
                """,
                tuple(comparison_params),
            )
        except Exception:
            frame = pd.DataFrame()
        if frame.empty:
            drift_filters = ["model_key = %s"]
            drift_params: list[object] = [model_key]
            drift_filters.append("source_run_id = %s")
            drift_params.append(generation_id)
            frame = self._warehouse.query_params(
                f"""
                SELECT DISTINCT
                    CAST(baseline_start AS STRING) AS baseline_start,
                    CAST(baseline_end AS STRING) AS baseline_end,
                    CAST(window_start AS STRING) AS window_start,
                    CAST(window_end AS STRING) AS window_end
                FROM {self._table_names.drift_metrics}
                WHERE {' AND '.join(drift_filters)}
                """,
                tuple(drift_params),
            )
        if frame.empty:
            return set()
        keys: set[tuple[str, str, str, str]] = set()
        for _, row in frame.iterrows():
            keys.add((
                _as_text(row.get("baseline_start")),
                _as_text(row.get("baseline_end")),
                _as_text(row.get("window_start")),
                _as_text(row.get("window_end")),
            ))
        return keys

    def get_latest_published_generation_id(self, model_key: str) -> str | None:
        frame = self._warehouse.query_params(
            f"""
            SELECT generation_id
            FROM {self._table_names.refresh_runs}
            WHERE model_key = %s
              AND generation_id IS NOT NULL
              AND generation_id <> ''
              AND published_at IS NOT NULL
            ORDER BY published_at DESC, completed_at DESC, started_at DESC
            LIMIT 1
            """,
            (model_key,),
        )
        if frame.empty:
            return None
        generation_id = _as_text(frame.iloc[0].get("generation_id")).strip()
        return generation_id or None

    def _generation_scoped_tables(self, *, include_quality_metrics: bool = False) -> tuple[str, ...]:
        tables = [
            self._table_names.comparison_windows,
            self._table_names.drift_metrics,
            self._table_names.quality_history,
            self._table_names.daily_quality_profiles,
            self._table_names.daily_class_quality_profiles,
            self._table_names.daily_feature_profiles,
            self._table_names.daily_class_feature_profiles,
            self._table_names.performance_metrics,
            self._table_names.daily_performance_profiles,
            self._table_names.daily_label_metrics,
            self._table_names.performance_bin_specs,
            self._table_names.incidents,
            self._table_names.incident_history,
        ]
        if include_quality_metrics:
            tables.append(self._table_names.quality_metrics)
        return tuple(tables)

    def _copy_generation_rows(
        self,
        table_name: str,
        *,
        model_key: str,
        from_generation_id: str,
        to_generation_id: str,
    ) -> None:
        if not from_generation_id or not to_generation_id or from_generation_id == to_generation_id:
            return
        columns = self._get_table_columns(table_name, cache=False)
        source_run_column = next(
            (column for column in columns if str(column).strip().lower() == "source_run_id"),
            None,
        )
        if not source_run_column:
            return
        value_columns = [column for column in columns if column != source_run_column]
        insert_columns = ", ".join(quote_column(column) for column in value_columns + [source_run_column])
        select_columns = ", ".join([
            *(quote_column(column) for column in value_columns),
            f"CAST(%s AS STRING) AS {quote_column(source_run_column)}",
        ])
        self._warehouse.execute_params(
            f"""
            INSERT INTO {table_name} ({insert_columns})
            SELECT {select_columns}
            FROM {table_name}
            WHERE model_key = %s
              AND {quote_column(source_run_column)} = %s
            """,
            (to_generation_id, model_key, from_generation_id),
        )

    def _copy_generation_state(
        self,
        model_key: str,
        *,
        from_generation_id: str,
        to_generation_id: str,
    ) -> None:
        if not from_generation_id or not to_generation_id or from_generation_id == to_generation_id:
            return
        for table_name in self._generation_scoped_tables():
            self._copy_generation_rows(
                table_name,
                model_key=model_key,
                from_generation_id=from_generation_id,
                to_generation_id=to_generation_id,
            )

    def prune_published_generations(self, model_key: str, *, keep: int = 2) -> None:
        keep = max(1, int(keep))
        frame = self._warehouse.query_params(
            f"""
            SELECT generation_id
            FROM {self._table_names.refresh_runs}
            WHERE model_key = %s
              AND generation_id IS NOT NULL
              AND generation_id <> ''
              AND published_at IS NOT NULL
            ORDER BY published_at DESC, completed_at DESC, started_at DESC
            LIMIT {keep}
            """,
            (model_key,),
        )
        generation_ids = [
            _as_text(row.get("generation_id")).strip()
            for _, row in frame.iterrows()
            if _as_text(row.get("generation_id")).strip()
        ]
        if not generation_ids:
            return
        placeholders = ", ".join(["%s"] * len(generation_ids))
        params = (model_key, *generation_ids)
        for table_name in self._generation_scoped_tables(include_quality_metrics=True):
            self._warehouse.execute_params(
                f"""
                DELETE FROM {table_name}
                WHERE model_key = %s
                  AND COALESCE(source_run_id, '') NOT IN ({placeholders})
                """,
                params,
            )

    def replace_refresh_result(self, model_key: str, result: RefreshResult) -> None:
        self.append_refresh_result(model_key, result)

    def start_refresh_run(
        self,
        *,
        model_key: str,
        requested_mode: str,
        run_kind: str,
        scope: str = "bootstrap",
        scheduled_at: str | None = None,
        data_min_date: str | None = None,
        data_max_date: str | None = None,
        range_start: str | None = None,
        range_end: str | None = None,
        rows_scanned: int = 0,
        label_rows_scanned: int = 0,
    ) -> str:
        run_id = str(uuid4())
        started_at = pd.Timestamp.now(tz=timezone.utc).isoformat()
        self._warehouse.execute_params(
            f"""
            INSERT INTO {self._table_names.refresh_runs} (
                run_id, model_key, requested_mode, run_kind, scope, status,
                started_at, scheduled_at, completed_at, window_count,
                data_min_date, data_max_date,
                range_start, range_end,
                drift_row_count, quality_row_count, performance_row_count, incident_row_count,
                rows_scanned, label_rows_scanned,
                error_message, generation_id, published_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                CAST(%s AS TIMESTAMP), CAST(%s AS TIMESTAMP), CAST(NULL AS TIMESTAMP), %s,
                CAST(%s AS DATE), CAST(%s AS DATE),
                CAST(%s AS DATE), CAST(%s AS DATE),
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, CAST(NULL AS TIMESTAMP)
            )
            """,
            (
                run_id,
                model_key,
                requested_mode,
                run_kind,
                scope,
                "running",
                started_at,
                scheduled_at,
                0,
                data_min_date or None,
                data_max_date or None,
                range_start or None,
                range_end or None,
                0,
                0,
                0,
                0,
                rows_scanned,
                label_rows_scanned,
                "",
                "",
            ),
        )
        return run_id

    def complete_refresh_run(
        self,
        run_id: str,
        *,
        status: str,
        window_count: int = 0,
        drift_row_count: int = 0,
        quality_row_count: int = 0,
        performance_row_count: int = 0,
        incident_row_count: int = 0,
        error_message: str = "",
        generation_id: str | None = None,
        publish: bool = False,
    ) -> None:
        completed_at = pd.Timestamp.now(tz=timezone.utc).isoformat()
        published_at = completed_at if publish and status == "completed" else None
        self._warehouse.execute_params(
            f"""
            UPDATE {self._table_names.refresh_runs}
            SET
                status = %s,
                completed_at = CAST(%s AS TIMESTAMP),
                window_count = %s,
                drift_row_count = %s,
                quality_row_count = %s,
                performance_row_count = %s,
                incident_row_count = %s,
                error_message = %s,
                generation_id = %s,
                published_at = CAST(%s AS TIMESTAMP)
            WHERE run_id = %s
            """,
            (
                status,
                completed_at,
                window_count,
                drift_row_count,
                quality_row_count,
                performance_row_count,
                incident_row_count,
                error_message,
                generation_id or "",
                published_at,
                run_id,
            ),
        )

    def update_refresh_run_metadata(
        self,
        run_id: str,
        *,
        data_min_date: str | None = None,
        data_max_date: str | None = None,
        range_start: str | None = None,
        range_end: str | None = None,
        rows_scanned: int = 0,
        label_rows_scanned: int = 0,
    ) -> None:
        self._warehouse.execute_params(
            f"""
            UPDATE {self._table_names.refresh_runs}
            SET
                data_min_date = CAST(%s AS DATE),
                data_max_date = CAST(%s AS DATE),
                range_start = CAST(%s AS DATE),
                range_end = CAST(%s AS DATE),
                rows_scanned = %s,
                label_rows_scanned = %s
            WHERE run_id = %s
            """,
            (
                data_min_date or None,
                data_max_date or None,
                range_start or None,
                range_end or None,
                int(rows_scanned or 0),
                int(label_rows_scanned or 0),
                run_id,
            ),
        )

    def get_recent_refresh_runs(self, model_key: str, limit: int = 10) -> list[dict[str, Any]]:
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self._table_names.refresh_runs}
            WHERE model_key = %s
            ORDER BY started_at DESC
            LIMIT {max(1, limit)}
            """,
            (model_key,),
        )
        if frame.empty:
            return []
        return [row.to_dict() for _, row in frame.iterrows()]

    def get_recent_incident_history(self, model_key: str, limit: int = 10) -> list[dict[str, Any]]:
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return []
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self._table_names.incident_history}
            WHERE {' AND '.join(filters)}
            ORDER BY observed_at DESC, window_end DESC
            LIMIT {max(1, limit)}
            """,
            tuple(params),
        )
        if frame.empty:
            return []
        return [row.to_dict() for _, row in frame.iterrows()]

    def get_recent_incident_history_all(self, limit: int = 50) -> list[dict[str, Any]]:
        frame = self._warehouse.query(
            f"""
            WITH latest_published AS (
                SELECT model_key, generation_id
                FROM (
                    SELECT
                        model_key,
                        generation_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY model_key
                            ORDER BY published_at DESC, completed_at DESC, started_at DESC
                        ) AS row_num
                    FROM {self._table_names.refresh_runs}
                    WHERE generation_id IS NOT NULL
                      AND generation_id <> ''
                      AND published_at IS NOT NULL
                ) ranked_generations
                WHERE row_num = 1
            )
            SELECT history.*
            FROM {self._table_names.incident_history} history
            INNER JOIN latest_published published
                ON history.model_key = published.model_key
               AND history.source_run_id = published.generation_id
            ORDER BY observed_at DESC, window_end DESC
            LIMIT {max(1, limit)}
            """
        )
        if frame.empty:
            return []
        return [row.to_dict() for _, row in frame.iterrows()]

    def get_latest_refresh_run(self, model_key: str, scope: str, statuses: tuple[str, ...] | None = None) -> dict[str, Any] | None:
        filters = ["model_key = %s", "scope = %s"]
        params: list[object] = [model_key, scope]
        if statuses:
            placeholders = ", ".join(["%s"] * len(statuses))
            filters.append(f"status IN ({placeholders})")
            params.extend(statuses)
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self._table_names.refresh_runs}
            WHERE {' AND '.join(filters)}
            ORDER BY started_at DESC
            LIMIT 1
            """,
            tuple(params),
        )
        if frame.empty:
            return None
        return frame.iloc[0].to_dict()

    def get_stale_running_refresh_runs(self, started_before: str) -> list[dict[str, Any]]:
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self._table_names.refresh_runs}
            WHERE status = 'running'
              AND completed_at IS NULL
              AND started_at < CAST(%s AS TIMESTAMP)
            ORDER BY started_at ASC
            """,
            (started_before,),
        )
        if frame.empty:
            return []
        return [row.to_dict() for _, row in frame.iterrows()]

    def get_daily_quality_profile_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return []
        filters.append("source_run_id = %s")
        params.append(generation_id)
        if start_date:
            filters.append("profile_date >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append("profile_date <= CAST(%s AS DATE)")
            params.append(end_date)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_rows AS (
                SELECT *
                FROM {self._table_names.daily_quality_profiles}
                WHERE {' AND '.join(filters)}
            ),
            recent_dates AS (
                SELECT profile_date
                FROM filtered_rows
                GROUP BY profile_date
                ORDER BY profile_date DESC
                LIMIT {_MAX_DAILY_HISTORY_DAYS}
            )
            SELECT *
            FROM filtered_rows
            WHERE profile_date IN (SELECT profile_date FROM recent_dates)
            ORDER BY profile_date
            """,
            tuple(params),
        )
        return [row.to_dict() for _, row in frame.iterrows()] if not frame.empty else []

    def get_daily_class_quality_profile_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return []
        filters.append("source_run_id = %s")
        params.append(generation_id)
        if start_date:
            filters.append("profile_date >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append("profile_date <= CAST(%s AS DATE)")
            params.append(end_date)
        if class_basis:
            filters.append("class_basis = %s")
            params.append(class_basis)
        if class_value:
            filters.append("class_value = %s")
            params.append(class_value)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_rows AS (
                SELECT *
                FROM {self._table_names.daily_class_quality_profiles}
                WHERE {' AND '.join(filters)}
            ),
            recent_dates AS (
                SELECT profile_date
                FROM filtered_rows
                GROUP BY profile_date
                ORDER BY profile_date DESC
                LIMIT {_MAX_DAILY_HISTORY_DAYS}
            )
            SELECT *
            FROM filtered_rows
            WHERE profile_date IN (SELECT profile_date FROM recent_dates)
            ORDER BY profile_date, class_basis, class_value
            """,
            tuple(params),
        )
        return [row.to_dict() for _, row in frame.iterrows()] if not frame.empty else []

    def get_daily_feature_profile_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return []
        filters.append("source_run_id = %s")
        params.append(generation_id)
        if start_date:
            filters.append("profile_date >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append("profile_date <= CAST(%s AS DATE)")
            params.append(end_date)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_rows AS (
                SELECT *
                FROM {self._table_names.daily_feature_profiles}
                WHERE {' AND '.join(filters)}
            ),
            recent_dates AS (
                SELECT profile_date
                FROM filtered_rows
                GROUP BY profile_date
                ORDER BY profile_date DESC
                LIMIT {_MAX_DAILY_HISTORY_DAYS}
            )
            SELECT *
            FROM filtered_rows
            WHERE profile_date IN (SELECT profile_date FROM recent_dates)
            ORDER BY profile_date, feature_name
            """,
            tuple(params),
        )
        return [row.to_dict() for _, row in frame.iterrows()] if not frame.empty else []

    def get_daily_class_feature_profile_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return []
        filters.append("source_run_id = %s")
        params.append(generation_id)
        if start_date:
            filters.append("profile_date >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append("profile_date <= CAST(%s AS DATE)")
            params.append(end_date)
        if class_basis:
            filters.append("class_basis = %s")
            params.append(class_basis)
        if class_value:
            filters.append("class_value = %s")
            params.append(class_value)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_rows AS (
                SELECT *
                FROM {self._table_names.daily_class_feature_profiles}
                WHERE {' AND '.join(filters)}
            ),
            recent_dates AS (
                SELECT profile_date
                FROM filtered_rows
                GROUP BY profile_date
                ORDER BY profile_date DESC
                LIMIT {_MAX_DAILY_HISTORY_DAYS}
            )
            SELECT *
            FROM filtered_rows
            WHERE profile_date IN (SELECT profile_date FROM recent_dates)
            ORDER BY profile_date, class_basis, class_value, feature_name
            """,
            tuple(params),
        )
        return [row.to_dict() for _, row in frame.iterrows()] if not frame.empty else []

    def get_daily_performance_profile_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return []
        filters.append("source_run_id = %s")
        params.append(generation_id)
        if start_date:
            filters.append("profile_date >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append("profile_date <= CAST(%s AS DATE)")
            params.append(end_date)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_rows AS (
                SELECT *
                FROM {self._table_names.daily_performance_profiles}
                WHERE {' AND '.join(filters)}
            ),
            recent_dates AS (
                SELECT profile_date
                FROM filtered_rows
                GROUP BY profile_date
                ORDER BY profile_date DESC
                LIMIT {_MAX_DAILY_HISTORY_DAYS}
            )
            SELECT *
            FROM filtered_rows
            WHERE profile_date IN (SELECT profile_date FROM recent_dates)
            ORDER BY profile_date, feature_name, bin_label, metric_name
            """,
            tuple(params),
        )
        return [row.to_dict() for _, row in frame.iterrows()] if not frame.empty else []

    def get_daily_label_metric_rows(
        self,
        model_key: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return []
        filters.append("source_run_id = %s")
        params.append(generation_id)
        if start_date:
            filters.append("profile_date >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append("profile_date <= CAST(%s AS DATE)")
            params.append(end_date)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_rows AS (
                SELECT *
                FROM {self._table_names.daily_label_metrics}
                WHERE {' AND '.join(filters)}
            ),
            recent_dates AS (
                SELECT profile_date
                FROM filtered_rows
                GROUP BY profile_date
                ORDER BY profile_date DESC
                LIMIT {_MAX_DAILY_HISTORY_DAYS}
            )
            SELECT *
            FROM filtered_rows
            WHERE profile_date IN (SELECT profile_date FROM recent_dates)
            ORDER BY profile_date
            """,
            tuple(params),
        )
        return [row.to_dict() for _, row in frame.iterrows()] if not frame.empty else []

    def has_daily_label_metric_rows(self, model_key: str) -> bool:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return False
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT 1 AS has_rows
            FROM {self._table_names.daily_label_metrics}
            WHERE {' AND '.join(filters)}
            LIMIT 1
            """,
            tuple(params),
        )
        return not frame.empty

    def get_performance_bin_specs(self, model_key: str) -> dict[str, tuple[float, ...]]:
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return {}
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT feature_name, edges_json
            FROM {self._table_names.performance_bin_specs}
            WHERE {' AND '.join(filters)}
            ORDER BY feature_name
            """,
            tuple(params),
        )
        if frame.empty:
            return {}
        specs: dict[str, tuple[float, ...]] = {}
        for _, row in frame.iterrows():
            feature_name = _as_text(row.get("feature_name")).strip()
            if not feature_name:
                continue
            try:
                loaded = json.loads(_as_text(row.get("edges_json")))
            except Exception:
                continue
            if not isinstance(loaded, list):
                continue
            edges = tuple(float(value) for value in loaded)
            if len(edges) < 2:
                continue
            specs[feature_name] = edges
        return specs

    def replace_performance_bin_specs(
        self,
        model_key: str,
        specs: dict[str, tuple[float, ...]],
        *,
        source_run_id: str | None = None,
    ) -> None:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        if source_run_id:
            filters.append("source_run_id = %s")
            params.append(source_run_id)
        self._warehouse.execute_params(
            f"DELETE FROM {self._table_names.performance_bin_specs} WHERE {' AND '.join(filters)}",
            tuple(params),
        )
        payload = [
            (
                model_key,
                feature_name,
                json.dumps([float(value) for value in edges]),
                pd.Timestamp.now(tz=timezone.utc).isoformat(),
                source_run_id or "",
            )
            for feature_name, edges in sorted(specs.items())
            if feature_name and len(edges) >= 2
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.performance_bin_specs} (
                model_key, feature_name, edges_json, computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _get_daily_quality_profile_rows_for_generation(
        self,
        model_key: str,
        *,
        source_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        if source_run_id:
            filters.append("source_run_id = %s")
            params.append(source_run_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self._table_names.daily_quality_profiles}
            WHERE {' AND '.join(filters)}
            ORDER BY profile_date
            """,
            tuple(params),
        )
        return [row.to_dict() for _, row in frame.iterrows()] if not frame.empty else []

    def _rewrite_quality_summary(self, model_key: str, *, source_run_id: str | None = None) -> None:
        filters = ["model_key = %s"]
        params: list[object] = [model_key]
        if source_run_id:
            filters.append("source_run_id = %s")
            params.append(source_run_id)
        self._warehouse.execute_params(
            f"DELETE FROM {self._table_names.quality_metrics} WHERE {' AND '.join(filters)}",
            tuple(params),
        )
        quality_rows = _aggregate_quality_summary_rows(
            model_key,
            self._get_daily_quality_profile_rows_for_generation(model_key, source_run_id=source_run_id),
            pd.Timestamp.now(tz=timezone.utc).isoformat(),
        )
        self._insert_quality_rows(quality_rows, source_run_id=source_run_id)

    def replace_all_refresh_results(self, model_key: str, result: RefreshResult, source_run_id: str | None = None) -> None:
        generation_id = source_run_id or ""
        self._insert_window_rows(result.window_rows, source_run_id=source_run_id)
        self._insert_drift_rows(result.drift_rows, source_run_id=generation_id)
        self._insert_quality_history_rows(result.quality_history_rows, source_run_id=generation_id)
        self._insert_daily_quality_profile_rows(result.daily_quality_profile_rows, source_run_id=source_run_id)
        self._insert_daily_class_quality_profile_rows(result.daily_class_quality_profile_rows, source_run_id=source_run_id)
        self._insert_daily_feature_profile_rows(result.daily_feature_profile_rows, source_run_id=source_run_id)
        self._insert_daily_class_feature_profile_rows(result.daily_class_feature_profile_rows, source_run_id=source_run_id)
        self._insert_performance_rows(result.performance_rows, source_run_id=generation_id)
        self._insert_daily_performance_profile_rows(result.daily_performance_profile_rows, source_run_id=source_run_id)
        self._insert_daily_label_metric_rows(result.daily_label_metric_rows, source_run_id=source_run_id)
        self.replace_performance_bin_specs(model_key, result.performance_bin_specs, source_run_id=generation_id)
        self._insert_incident_rows(result.incident_rows, source_run_id=generation_id)
        self._insert_incident_history_rows(result.incident_history_rows, source_run_id=generation_id)
        self._rewrite_quality_summary(model_key, source_run_id=generation_id)
        self._sync_read_model()

    def append_refresh_result(self, model_key: str, result: RefreshResult, source_run_id: str | None = None) -> None:
        previous_generation_id = self.get_latest_published_generation_id(model_key)
        generation_id = source_run_id or previous_generation_id or ""
        if previous_generation_id and generation_id and previous_generation_id != generation_id:
            self._copy_generation_state(
                model_key,
                from_generation_id=previous_generation_id,
                to_generation_id=generation_id,
            )
        drift_windows = {
            (
                _as_text(row.get("baseline_start")),
                _as_text(row.get("baseline_end")),
                _as_text(row.get("window_start")),
                _as_text(row.get("window_end")),
            )
            for row in result.drift_rows
        }
        performance_windows = {
            (
                _as_text(row.get("window_start")),
                _as_text(row.get("window_end")),
            )
            for row in result.performance_rows
        }
        quality_windows = {_as_text(row.get("window_id")) for row in result.quality_history_rows}
        incident_history_windows = {_as_text(row.get("window_id")) for row in result.incident_history_rows}
        daily_quality_dates = {_as_text(row.get("profile_date")) for row in result.daily_quality_profile_rows}
        daily_class_quality_keys = {
            (
                _as_text(row.get("profile_date")),
                _as_text(row.get("class_basis")),
                _as_text(row.get("class_value")),
            )
            for row in result.daily_class_quality_profile_rows
        }
        daily_feature_dates = {_as_text(row.get("profile_date")) for row in result.daily_feature_profile_rows}
        daily_class_feature_keys = {
            (
                _as_text(row.get("profile_date")),
                _as_text(row.get("class_basis")),
                _as_text(row.get("class_value")),
            )
            for row in result.daily_class_feature_profile_rows
        }
        daily_performance_dates = {_as_text(row.get("profile_date")) for row in result.daily_performance_profile_rows}
        daily_label_metric_dates = {_as_text(row.get("profile_date")) for row in result.daily_label_metric_rows}

        for baseline_start, baseline_end, window_start, window_end in drift_windows:
            self._warehouse.execute_params(
                f"""
                DELETE FROM {self._table_names.drift_metrics}
                WHERE model_key = %s
                  AND source_run_id = %s
                  AND baseline_start = CAST(%s AS DATE)
                  AND baseline_end = CAST(%s AS DATE)
                  AND window_start = CAST(%s AS DATE)
                  AND window_end = CAST(%s AS DATE)
                """,
                (model_key, generation_id, baseline_start, baseline_end, window_start, window_end),
            )

        for window_start, window_end in performance_windows:
            self._warehouse.execute_params(
                f"""
                DELETE FROM {self._table_names.performance_metrics}
                WHERE model_key = %s
                  AND source_run_id = %s
                  AND window_start = CAST(%s AS DATE)
                  AND window_end = CAST(%s AS DATE)
                """,
                (model_key, generation_id, window_start, window_end),
            )

        for window_id in {_as_text(row.get("window_id")) for row in result.window_rows}:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.comparison_windows} WHERE model_key = %s AND source_run_id = %s AND window_id = %s",
                (model_key, generation_id, window_id),
            )

        for window_id in quality_windows:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.quality_history} WHERE model_key = %s AND source_run_id = %s AND window_id = %s",
                (model_key, generation_id, window_id),
            )

        for window_id in incident_history_windows:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.incident_history} WHERE model_key = %s AND source_run_id = %s AND window_id = %s",
                (model_key, generation_id, window_id),
            )

        for profile_date in daily_quality_dates:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.daily_quality_profiles} WHERE model_key = %s AND source_run_id = %s AND profile_date = CAST(%s AS DATE)",
                (model_key, generation_id, profile_date),
            )

        for profile_date, class_basis, class_value in daily_class_quality_keys:
            self._warehouse.execute_params(
                f"""
                DELETE FROM {self._table_names.daily_class_quality_profiles}
                WHERE model_key = %s
                  AND source_run_id = %s
                  AND profile_date = CAST(%s AS DATE)
                  AND class_basis = %s
                  AND class_value = %s
                """,
                (model_key, generation_id, profile_date, class_basis, class_value),
            )

        for profile_date in daily_feature_dates:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.daily_feature_profiles} WHERE model_key = %s AND source_run_id = %s AND profile_date = CAST(%s AS DATE)",
                (model_key, generation_id, profile_date),
            )

        for profile_date, class_basis, class_value in daily_class_feature_keys:
            self._warehouse.execute_params(
                f"""
                DELETE FROM {self._table_names.daily_class_feature_profiles}
                WHERE model_key = %s
                  AND source_run_id = %s
                  AND profile_date = CAST(%s AS DATE)
                  AND class_basis = %s
                  AND class_value = %s
                """,
                (model_key, generation_id, profile_date, class_basis, class_value),
            )

        for profile_date in daily_performance_dates:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.daily_performance_profiles} WHERE model_key = %s AND source_run_id = %s AND profile_date = CAST(%s AS DATE)",
                (model_key, generation_id, profile_date),
            )

        for profile_date in daily_label_metric_dates:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.daily_label_metrics} WHERE model_key = %s AND source_run_id = %s AND profile_date = CAST(%s AS DATE)",
                (model_key, generation_id, profile_date),
            )

        if result.incident_rows or result.incident_history_rows:
            self._warehouse.execute_params(
                f"DELETE FROM {self._table_names.incidents} WHERE model_key = %s AND source_run_id = %s",
                (model_key, generation_id),
            )

        self._insert_window_rows(result.window_rows, source_run_id=generation_id)
        self._insert_drift_rows(result.drift_rows, source_run_id=generation_id)
        self._insert_quality_history_rows(result.quality_history_rows, source_run_id=generation_id)
        self._insert_daily_quality_profile_rows(result.daily_quality_profile_rows, source_run_id=generation_id)
        self._insert_daily_class_quality_profile_rows(result.daily_class_quality_profile_rows, source_run_id=generation_id)
        self._insert_daily_feature_profile_rows(result.daily_feature_profile_rows, source_run_id=generation_id)
        self._insert_daily_class_feature_profile_rows(result.daily_class_feature_profile_rows, source_run_id=generation_id)
        self._insert_performance_rows(result.performance_rows, source_run_id=generation_id)
        self._insert_daily_performance_profile_rows(result.daily_performance_profile_rows, source_run_id=generation_id)
        self._insert_daily_label_metric_rows(result.daily_label_metric_rows, source_run_id=generation_id)
        if result.performance_bin_specs:
            persisted_specs = self.get_performance_bin_specs(model_key)
            persisted_specs.update(result.performance_bin_specs)
            self.replace_performance_bin_specs(model_key, persisted_specs, source_run_id=generation_id)
        self._insert_incident_rows(result.incident_rows, source_run_id=generation_id)
        self._insert_incident_history_rows(result.incident_history_rows, source_run_id=generation_id)
        self._rewrite_quality_summary(model_key, source_run_id=generation_id)
        self._sync_read_model()

    def _insert_drift_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["window_id"],
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
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.drift_metrics} (
                model_key, window_id, feature_name, metric_name, metric_value,
                window_start, window_end, baseline_start, baseline_end,
                ref_mean, cur_mean, ref_std, cur_std,
                ref_null_pct, cur_null_pct, ref_count, cur_count, computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_quality_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
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
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.quality_metrics} (
                model_key, total_rows, min_date, max_date,
                prediction_mean, prediction_std, daily_volume, null_rates, computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_window_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["window_id"],
                row["model_key"],
                row["window_grain"],
                row["window_start"],
                row["window_end"],
                row["baseline_start"],
                row["baseline_end"],
                row["baseline_kind"],
                row["created_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.comparison_windows} (
                window_id, model_key, window_grain,
                window_start, window_end,
                baseline_start, baseline_end, baseline_kind,
                created_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_quality_history_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["window_id"],
                row["window_start"],
                row["window_end"],
                row["baseline_start"],
                row["baseline_end"],
                row["row_count"],
                row["prediction_mean"],
                row["prediction_std"],
                row["null_rates"],
                row["computed_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.quality_history} (
                model_key, window_id, window_start, window_end,
                baseline_start, baseline_end, row_count,
                prediction_mean, prediction_std, null_rates, computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_daily_quality_profile_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["profile_date"],
                row["row_count"],
                row["prediction_mean"],
                row["prediction_std"],
                row["null_rates"],
                row["label_row_count"],
                row["computed_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.daily_quality_profiles} (
                model_key, profile_date, row_count,
                prediction_mean, prediction_std, null_rates, label_row_count,
                computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_daily_class_quality_profile_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["profile_date"],
                row["class_basis"],
                row["class_value"],
                row["row_count"],
                row["prediction_mean"],
                row["prediction_std"],
                row["null_rates"],
                row["label_row_count"],
                row["computed_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.daily_class_quality_profiles} (
                model_key, profile_date, class_basis, class_value, row_count,
                prediction_mean, prediction_std, null_rates, label_row_count,
                computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_daily_feature_profile_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["profile_date"],
                row["feature_name"],
                row["feature_kind"],
                row["row_count"],
                row["non_null_count"],
                row["null_pct"],
                row["mean"],
                row["std"],
                row["min_value"],
                row["max_value"],
                row["distribution_json"],
                row["computed_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.daily_feature_profiles} (
                model_key, profile_date, feature_name, feature_kind,
                row_count, non_null_count, null_pct,
                mean, std, min_value, max_value,
                distribution_json, computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_daily_class_feature_profile_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["profile_date"],
                row["class_basis"],
                row["class_value"],
                row["feature_name"],
                row["feature_kind"],
                row["row_count"],
                row["non_null_count"],
                row["null_pct"],
                row["mean"],
                row["std"],
                row["min_value"],
                row["max_value"],
                row["distribution_json"],
                row["computed_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.daily_class_feature_profiles} (
                model_key, profile_date, class_basis, class_value, feature_name, feature_kind,
                row_count, non_null_count, null_pct,
                mean, std, min_value, max_value,
                distribution_json, computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_performance_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["window_id"],
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
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.performance_metrics} (
                model_key, window_id, feature_name, bin_label,
                baseline_metric, current_metric, delta,
                volume_pct, contribution, metric_name,
                window_start, window_end, computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_daily_performance_profile_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["profile_date"],
                row["feature_name"],
                row["bin_label"],
                row["metric_name"],
                row["metric_value"],
                row["row_count"],
                row["volume_pct"],
                row["computed_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.daily_performance_profiles} (
                model_key, profile_date, feature_name, bin_label,
                metric_name, metric_value, row_count, volume_pct, computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_daily_label_metric_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["profile_date"],
                row["actual_positive_count"],
                row["actual_negative_count"],
                row["predicted_positive_count"],
                row["predicted_negative_count"],
                row["tp"],
                row["fp"],
                row["fn"],
                row["tn"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["accuracy"],
                row["computed_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.daily_label_metrics} (
                model_key, profile_date,
                actual_positive_count, actual_negative_count,
                predicted_positive_count, predicted_negative_count,
                tp, fp, fn, tn,
                precision, recall, f1, accuracy,
                computed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_incident_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
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
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.incidents} (
                model_key, feature_name, metric_name,
                severity, status, metric_value, window_end, observed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def _insert_incident_history_rows(self, rows: list[dict], *, source_run_id: str | None = None) -> None:
        payload = [
            (
                row["model_key"],
                row["feature_name"],
                row["metric_name"],
                row["event_type"],
                row["severity"],
                row["status"],
                row["metric_value"],
                row["window_id"],
                row["window_start"],
                row["window_end"],
                row["baseline_start"],
                row["baseline_end"],
                row["observed_at"],
                source_run_id or "",
            )
            for row in rows
        ]
        self._warehouse.execute_batch(
            f"""
            INSERT INTO {self._table_names.incident_history} (
                model_key, feature_name, metric_name,
                event_type, severity, status, metric_value,
                window_id, window_start, window_end, baseline_start, baseline_end, observed_at, source_run_id
            ) VALUES
            """.strip(),
            payload,
        )

    def get_current_incident_state(self, model_key: str) -> dict[tuple[str, str, str], dict[str, Any]]:
        generation_id = self.get_latest_published_generation_id(model_key)
        if not generation_id:
            return {}
        filters = ["model_key = %s", "status = 'open'"]
        params: list[object] = [model_key]
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT model_key, feature_name, metric_name, severity, status, metric_value, window_end, observed_at
            FROM {self._table_names.incidents}
            WHERE {' AND '.join(filters)}
            """,
            tuple(params),
        )
        if frame.empty:
            return {}
        state: dict[tuple[str, str, str], dict[str, Any]] = {}
        for _, row in frame.iterrows():
            key = (
                _as_text(row.get("model_key")),
                _as_text(row.get("feature_name")),
                _as_text(row.get("metric_name")),
            )
            state[key] = {
                "model_key": key[0],
                "feature_name": key[1],
                "metric_name": key[2],
                "severity": _as_text(row.get("severity")),
                "status": _as_text(row.get("status")) or "open",
                "metric_value": float(row.get("metric_value") or 0.0),
                "window_end": _as_text(row.get("window_end")),
                "observed_at": _as_text(row.get("observed_at")),
            }
        return state

    def _get_monitor_summary_from_warehouse(self) -> pd.DataFrame:
        return self._warehouse.query(
            f"""
            WITH latest_published AS (
                SELECT model_key, generation_id
                FROM (
                    SELECT
                        model_key,
                        generation_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY model_key
                            ORDER BY published_at DESC, completed_at DESC, started_at DESC
                        ) AS row_num
                    FROM {self._table_names.refresh_runs}
                    WHERE generation_id IS NOT NULL
                      AND generation_id <> ''
                      AND published_at IS NOT NULL
                ) ranked_generations
                WHERE row_num = 1
            ),
            historical_drift AS (
                SELECT
                    d.model_key,
                    COALESCE(MAX(CASE WHEN metric_name = 'psi' THEN metric_value END), 0) AS max_psi,
                    COUNT(DISTINCT CASE WHEN metric_name = 'psi' THEN feature_name END) AS feature_count,
                    MAX(window_end) AS latest_window_end
                FROM {self._table_names.drift_metrics} d
                INNER JOIN latest_published published
                    ON d.model_key = published.model_key
                   AND d.source_run_id = published.generation_id
                GROUP BY d.model_key
            ),
            latest_quality AS (
                SELECT q.*
                FROM {self._table_names.quality_metrics} q
                INNER JOIN latest_published published
                    ON q.model_key = published.model_key
                   AND q.source_run_id = published.generation_id
                INNER JOIN (
                    SELECT quality.model_key, MAX(quality.computed_at) AS latest_computed_at
                    FROM {self._table_names.quality_metrics} quality
                    INNER JOIN latest_published published_quality
                        ON quality.model_key = published_quality.model_key
                       AND quality.source_run_id = published_quality.generation_id
                    GROUP BY quality.model_key
                ) latest
                    ON q.model_key = latest.model_key
                   AND q.computed_at = latest.latest_computed_at
            ),
            open_incidents AS (
                SELECT incidents.model_key, COUNT(*) AS open_incident_count
                FROM {self._table_names.incidents} incidents
                INNER JOIN latest_published published
                    ON incidents.model_key = published.model_key
                   AND incidents.source_run_id = published.generation_id
                WHERE incidents.status = 'open'
                GROUP BY incidents.model_key
            )
            SELECT
                c.model_key,
                c.display_name,
                COALESCE(d.max_psi, 0) AS max_psi,
                COALESCE(d.feature_count, 0) AS feature_count,
                d.latest_window_end,
                q.total_rows,
                q.max_date AS latest_data_date,
                q.computed_at AS last_refresh_at,
                COALESCE(i.open_incident_count, 0) AS open_incident_count
            FROM {self._table_names.monitor_configs} c
            LEFT JOIN historical_drift d
                ON c.model_key = d.model_key
            LEFT JOIN latest_quality q
                ON c.model_key = q.model_key
            LEFT JOIN open_incidents i
                ON c.model_key = i.model_key
            WHERE c.status = 'active'
            ORDER BY max_psi DESC, c.display_name
            """
        )

    def _get_open_incidents_from_warehouse(self) -> pd.DataFrame:
        return self._warehouse.query(
            f"""
            WITH latest_published AS (
                SELECT model_key, generation_id
                FROM (
                    SELECT
                        model_key,
                        generation_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY model_key
                            ORDER BY published_at DESC, completed_at DESC, started_at DESC
                        ) AS row_num
                    FROM {self._table_names.refresh_runs}
                    WHERE generation_id IS NOT NULL
                      AND generation_id <> ''
                      AND published_at IS NOT NULL
                ) ranked_generations
                WHERE row_num = 1
            )
            SELECT incidents.model_key, feature_name, metric_name, severity, metric_value, window_end, observed_at
            FROM {self._table_names.incidents} incidents
            INNER JOIN latest_published published
                ON incidents.model_key = published.model_key
               AND incidents.source_run_id = published.generation_id
            WHERE incidents.status = 'open'
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
