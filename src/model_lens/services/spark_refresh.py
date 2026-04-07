from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from model_lens.config import settings
from model_lens.domain.models import MonitorConfig, RefreshResult
from model_lens.domain.performance_metrics import default_performance_metric_names
from model_lens.services.incidents import DEFAULT_THRESHOLDS
from model_lens.services.control_plane import (
    ControlPlaneRepository,
    _resolve_source_labels_join_col,
)
from model_lens.services.lakebase import LakebaseReadModel
from model_lens.services.spark_session import get_spark_session
from model_lens.services.table_names import TableNames
from model_lens.services.warehouse import WarehouseConnection
from model_lens.analytics.drift import (
    EPSILON,
)


CATEGORICAL_TOP_N = 100
PERFORMANCE_BIN_COUNT = 10
NUMERIC_DRIFT_BIN_COUNT = 20


@dataclass(frozen=True)
class SparkDailyProfiles:
    daily_quality_profile_rows: list[dict[str, Any]]
    daily_feature_profile_rows: list[dict[str, Any]]
    daily_performance_profile_rows: list[dict[str, Any]]
    performance_bin_specs: dict[str, tuple[float, ...]]


QUALITY_PROFILE_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("profile_date", StringType(), False),
    StructField("row_count", LongType(), False),
    StructField("prediction_mean", DoubleType(), True),
    StructField("prediction_std", DoubleType(), True),
    StructField("null_rates", StringType(), False),
    StructField("label_row_count", LongType(), False),
    StructField("computed_at", StringType(), False),
])

FEATURE_PROFILE_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("profile_date", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("feature_kind", StringType(), False),
    StructField("row_count", LongType(), False),
    StructField("non_null_count", LongType(), False),
    StructField("null_pct", DoubleType(), False),
    StructField("mean", DoubleType(), True),
    StructField("std", DoubleType(), True),
    StructField("min_value", DoubleType(), True),
    StructField("max_value", DoubleType(), True),
    StructField("distribution_json", StringType(), False),
    StructField("computed_at", StringType(), False),
])

PERFORMANCE_PROFILE_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("profile_date", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("bin_label", StringType(), False),
    StructField("metric_name", StringType(), False),
    StructField("metric_value", DoubleType(), False),
    StructField("row_count", LongType(), False),
    StructField("volume_pct", DoubleType(), False),
    StructField("computed_at", StringType(), False),
])

METADATA_SCHEMA = StructType([
    StructField("window_id", StringType(), False),
    StructField("model_key", StringType(), False),
    StructField("window_grain", StringType(), False),
    StructField("window_start", StringType(), False),
    StructField("window_end", StringType(), False),
    StructField("baseline_start", StringType(), False),
    StructField("baseline_end", StringType(), False),
    StructField("baseline_kind", StringType(), False),
])

QUALITY_METRIC_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("total_rows", LongType(), False),
    StructField("min_date", StringType(), True),
    StructField("max_date", StringType(), True),
    StructField("prediction_mean", DoubleType(), True),
    StructField("prediction_std", DoubleType(), True),
    StructField("daily_volume", StringType(), False),
    StructField("null_rates", StringType(), False),
    StructField("computed_at", StringType(), False),
])

DRIFT_METRIC_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("metric_name", StringType(), False),
    StructField("metric_value", DoubleType(), False),
    StructField("window_start", StringType(), False),
    StructField("window_end", StringType(), False),
    StructField("baseline_start", StringType(), False),
    StructField("baseline_end", StringType(), False),
    StructField("ref_mean", DoubleType(), True),
    StructField("cur_mean", DoubleType(), True),
    StructField("ref_std", DoubleType(), True),
    StructField("cur_std", DoubleType(), True),
    StructField("ref_null_pct", DoubleType(), False),
    StructField("cur_null_pct", DoubleType(), False),
    StructField("ref_count", LongType(), False),
    StructField("cur_count", LongType(), False),
    StructField("computed_at", StringType(), False),
])

WINDOW_WRITE_SCHEMA = StructType([
    StructField("window_id", StringType(), False),
    StructField("model_key", StringType(), False),
    StructField("window_grain", StringType(), False),
    StructField("window_start", StringType(), False),
    StructField("window_end", StringType(), False),
    StructField("baseline_start", StringType(), False),
    StructField("baseline_end", StringType(), False),
    StructField("baseline_kind", StringType(), False),
    StructField("created_at", StringType(), False),
    StructField("source_run_id", StringType(), False),
])

QUALITY_HISTORY_WRITE_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("window_id", StringType(), False),
    StructField("window_start", StringType(), False),
    StructField("window_end", StringType(), False),
    StructField("baseline_start", StringType(), False),
    StructField("baseline_end", StringType(), False),
    StructField("row_count", LongType(), False),
    StructField("prediction_mean", DoubleType(), True),
    StructField("prediction_std", DoubleType(), True),
    StructField("null_rates", StringType(), False),
    StructField("computed_at", StringType(), False),
])

DAILY_QUALITY_WRITE_SCHEMA = StructType([
    *QUALITY_PROFILE_SCHEMA.fields,
    StructField("source_run_id", StringType(), False),
])

DAILY_FEATURE_WRITE_SCHEMA = StructType([
    *FEATURE_PROFILE_SCHEMA.fields,
    StructField("source_run_id", StringType(), False),
])

PERFORMANCE_METRIC_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("bin_label", StringType(), False),
    StructField("baseline_metric", DoubleType(), False),
    StructField("current_metric", DoubleType(), False),
    StructField("delta", DoubleType(), False),
    StructField("volume_pct", DoubleType(), False),
    StructField("contribution", DoubleType(), False),
    StructField("metric_name", StringType(), False),
    StructField("window_start", StringType(), False),
    StructField("window_end", StringType(), False),
    StructField("computed_at", StringType(), False),
])

DAILY_PERFORMANCE_WRITE_SCHEMA = StructType([
    *PERFORMANCE_PROFILE_SCHEMA.fields,
    StructField("source_run_id", StringType(), False),
])

PERFORMANCE_BIN_SPEC_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("edges_json", StringType(), False),
    StructField("computed_at", StringType(), False),
])

INCIDENT_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("metric_name", StringType(), False),
    StructField("severity", StringType(), False),
    StructField("status", StringType(), False),
    StructField("metric_value", DoubleType(), False),
    StructField("window_end", StringType(), False),
    StructField("observed_at", StringType(), False),
])

INCIDENT_HISTORY_SCHEMA = StructType([
    StructField("model_key", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("metric_name", StringType(), False),
    StructField("event_type", StringType(), False),
    StructField("severity", StringType(), False),
    StructField("status", StringType(), False),
    StructField("metric_value", DoubleType(), False),
    StructField("window_id", StringType(), False),
    StructField("window_start", StringType(), False),
    StructField("window_end", StringType(), False),
    StructField("baseline_start", StringType(), False),
    StructField("baseline_end", StringType(), False),
    StructField("observed_at", StringType(), False),
])

_NULL_RATE_SCHEMA = MapType(StringType(), DoubleType(), True)
_CATEGORICAL_COUNTS_SCHEMA = MapType(StringType(), DoubleType(), True)
_NUMERIC_DISTRIBUTION_SCHEMA = StructType([
    StructField("edges", ArrayType(DoubleType()), True),
    StructField("counts", ArrayType(DoubleType()), True),
    StructField("sample_values", ArrayType(DoubleType()), True),
])


def _spark_col(name: str):
    return F.col(f"`{name}`")


def _safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_date(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _iter_local_rows(frame: DataFrame):
    return frame.toLocalIterator()


def _union_all(frames: list[DataFrame]) -> DataFrame | None:
    if not frames:
        return None
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)
    return combined


def _sql_string_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_date_literal(value: str) -> str:
    return f"CAST({_sql_string_literal(value)} AS DATE)"


def _normalize_rows_with_model_key(
    rows: list[dict[str, Any]],
    *,
    default_model_key: str | None = None,
    row_kind: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    normalized_default = str(default_model_key or "").strip() or None
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = dict(row)
        current_model_key = str(payload.get("model_key") or "").strip() or None
        if current_model_key is None:
            if normalized_default is None:
                raise ValueError(
                    f"{row_kind} row at index {index} is missing model_key and no default model key was supplied"
                )
            payload["model_key"] = normalized_default
        elif normalized_default is not None and current_model_key != normalized_default:
            raise ValueError(
                f"{row_kind} row at index {index} has model_key={current_model_key!r}, expected {normalized_default!r}"
            )
        else:
            payload["model_key"] = current_model_key
        normalized_rows.append(payload)
    return normalized_rows


class SparkRefreshRepository(ControlPlaneRepository):
    def __init__(
        self,
        warehouse: WarehouseConnection,
        table_names: TableNames,
        read_model: LakebaseReadModel | None = None,
        spark=None,
    ):
        super().__init__(warehouse=warehouse, table_names=table_names, read_model=read_model)
        self._spark = spark or get_spark_session()

    def fork_for_worker(self) -> SparkRefreshRepository:
        warehouse = WarehouseConnection(
            warehouse_id=getattr(self._warehouse, "_warehouse_id", ""),
            host=getattr(self._warehouse, "_host", ""),
        )
        return SparkRefreshRepository(
            warehouse=warehouse,
            table_names=self._table_names,
            read_model=None,
            spark=self._spark,
        )

    def recommended_max_parallel_refresh_workers(self) -> int:
        return 1

    def _read_table(self, table_name: str) -> DataFrame:
        return self._spark.table(table_name)

    def _delete_where(self, table_name: str, predicate_sql: str) -> None:
        self._spark.sql(f"DELETE FROM {table_name} WHERE {predicate_sql}")

    def _append_df_to_table(self, table_name: str, frame: DataFrame) -> None:
        if not frame.take(1):
            return
        frame.write.mode("append").saveAsTable(table_name)

    def _typed_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        schema: StructType,
        date_columns: tuple[str, ...] = (),
        timestamp_columns: tuple[str, ...] = (),
        default_model_key: str | None = None,
        row_kind: str = "rows",
    ) -> DataFrame:
        normalized_rows = rows or []
        requires_model_key = any(field.name == "model_key" and not field.nullable for field in schema.fields)
        if requires_model_key:
            normalized_rows = _normalize_rows_with_model_key(
                normalized_rows,
                default_model_key=default_model_key,
                row_kind=row_kind,
            )
        frame = self._spark.createDataFrame(normalized_rows, schema=schema)
        for column_name in date_columns:
            frame = frame.withColumn(column_name, F.to_date(F.col(column_name)))
        for column_name in timestamp_columns:
            frame = frame.withColumn(column_name, F.to_timestamp(F.col(column_name)))
        return frame

    def _quality_metric_df_from_rows(self, rows: list[dict[str, Any]], *, default_model_key: str | None = None) -> DataFrame:
        return self._typed_df_from_rows(
            rows,
            schema=QUALITY_METRIC_SCHEMA,
            date_columns=("min_date", "max_date"),
            timestamp_columns=("computed_at",),
            default_model_key=default_model_key,
            row_kind="quality_metric",
        )

    def _drift_df_from_rows(self, rows: list[dict[str, Any]], *, default_model_key: str | None = None) -> DataFrame:
        return self._typed_df_from_rows(
            rows,
            schema=DRIFT_METRIC_SCHEMA,
            date_columns=("window_start", "window_end", "baseline_start", "baseline_end"),
            timestamp_columns=("computed_at",),
            default_model_key=default_model_key,
            row_kind="drift_metric",
        )

    def _window_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        source_run_id: str | None = None,
        default_model_key: str | None = None,
    ) -> DataFrame:
        payload = [{**row, "source_run_id": source_run_id or ""} for row in rows]
        return self._typed_df_from_rows(
            payload,
            schema=WINDOW_WRITE_SCHEMA,
            date_columns=("window_start", "window_end", "baseline_start", "baseline_end"),
            timestamp_columns=("created_at",),
            default_model_key=default_model_key,
            row_kind="comparison_window",
        )

    def _quality_history_df_from_rows(self, rows: list[dict[str, Any]], *, default_model_key: str | None = None) -> DataFrame:
        return self._typed_df_from_rows(
            rows,
            schema=QUALITY_HISTORY_WRITE_SCHEMA,
            date_columns=("window_start", "window_end", "baseline_start", "baseline_end"),
            timestamp_columns=("computed_at",),
            default_model_key=default_model_key,
            row_kind="quality_history",
        )

    def _daily_quality_profile_persist_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        source_run_id: str | None = None,
        default_model_key: str | None = None,
    ) -> DataFrame:
        payload = [{**row, "source_run_id": source_run_id or ""} for row in rows]
        return self._typed_df_from_rows(
            payload,
            schema=DAILY_QUALITY_WRITE_SCHEMA,
            date_columns=("profile_date",),
            timestamp_columns=("computed_at",),
            default_model_key=default_model_key,
            row_kind="daily_quality_profile",
        )

    def _daily_feature_profile_persist_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        source_run_id: str | None = None,
        default_model_key: str | None = None,
    ) -> DataFrame:
        payload = [{**row, "source_run_id": source_run_id or ""} for row in rows]
        return self._typed_df_from_rows(
            payload,
            schema=DAILY_FEATURE_WRITE_SCHEMA,
            date_columns=("profile_date",),
            timestamp_columns=("computed_at",),
            default_model_key=default_model_key,
            row_kind="daily_feature_profile",
        )

    def _performance_df_from_rows(self, rows: list[dict[str, Any]], *, default_model_key: str | None = None) -> DataFrame:
        return self._typed_df_from_rows(
            rows,
            schema=PERFORMANCE_METRIC_SCHEMA,
            date_columns=("window_start", "window_end"),
            timestamp_columns=("computed_at",),
            default_model_key=default_model_key,
            row_kind="performance_metric",
        )

    def _daily_performance_profile_persist_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        source_run_id: str | None = None,
        default_model_key: str | None = None,
    ) -> DataFrame:
        payload = [{**row, "source_run_id": source_run_id or ""} for row in rows]
        return self._typed_df_from_rows(
            payload,
            schema=DAILY_PERFORMANCE_WRITE_SCHEMA,
            date_columns=("profile_date",),
            timestamp_columns=("computed_at",),
            default_model_key=default_model_key,
            row_kind="daily_performance_profile",
        )

    def _incident_df_from_rows(self, rows: list[dict[str, Any]], *, default_model_key: str | None = None) -> DataFrame:
        return self._typed_df_from_rows(
            rows,
            schema=INCIDENT_SCHEMA,
            date_columns=("window_end",),
            timestamp_columns=("observed_at",),
            default_model_key=default_model_key,
            row_kind="incident",
        )

    def _incident_history_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        default_model_key: str | None = None,
    ) -> DataFrame:
        return self._typed_df_from_rows(
            rows,
            schema=INCIDENT_HISTORY_SCHEMA,
            date_columns=("window_start", "window_end", "baseline_start", "baseline_end"),
            timestamp_columns=("observed_at",),
            default_model_key=default_model_key,
            row_kind="incident_history",
        )

    def _performance_bin_spec_df_from_specs(self, model_key: str, specs: dict[str, tuple[float, ...]]) -> DataFrame:
        frame = self._spark.createDataFrame(
            [
                {
                    "model_key": model_key,
                    "feature_name": feature_name,
                    "edges_json": json.dumps([float(value) for value in edges]),
                    "computed_at": None,
                }
                for feature_name, edges in sorted(specs.items())
                if feature_name and len(edges) >= 2
            ],
            schema=PERFORMANCE_BIN_SPEC_SCHEMA,
        )
        return frame.withColumn(
            "computed_at",
            F.coalesce(F.to_timestamp(F.col("computed_at")), F.current_timestamp()),
        )

    def _base_source_df(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        selected_columns: tuple[str, ...] | None = None,
    ) -> tuple[DataFrame, list[str]]:
        self.validate_monitor_source(config)
        source_columns = list(self._spark.table(config.source_table).columns)
        select_columns = [
            config.contract.timestamp_col,
            config.contract.prediction_col,
        ]
        if config.contract.model_id_col:
            select_columns.append(config.contract.model_id_col)
        optional_columns = [
            config.contract.model_version_col,
            config.contract.prediction_score_col,
            config.contract.entity_id_col,
        ]
        for column in optional_columns:
            if column:
                select_columns.append(column)
        for column in selected_columns or config.contract.feature_columns:
            if column:
                select_columns.append(column)
        if config.contract.label_col and not config.labels_table:
            select_columns.append(config.contract.label_col)
        projection = [
            _spark_col(column).alias(column)
            for column in dict.fromkeys(select_columns)
            if column and column in source_columns
        ]
        df = self._spark.table(config.source_table).select(*projection)
        if config.model_id_value and config.contract.model_id_col:
            df = df.filter(_spark_col(config.contract.model_id_col) == F.lit(config.model_id_value))
        if config.model_version_value and config.contract.model_version_col:
            df = df.filter(_spark_col(config.contract.model_version_col) == F.lit(config.model_version_value))
        event_date = F.to_date(_spark_col(config.contract.timestamp_col))
        if start_date:
            df = df.filter(event_date >= F.to_date(F.lit(start_date)))
        if end_date:
            df = df.filter(event_date <= F.to_date(F.lit(end_date)))
        return df, source_columns

    def load_source_df(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        selected_columns: tuple[str, ...] | None = None,
    ) -> DataFrame:
        df, source_columns = self._base_source_df(
            config,
            start_date=start_date,
            end_date=end_date,
            selected_columns=selected_columns,
        )
        if not config.labels_table or not config.contract.label_col:
            return df

        source_join_col = _resolve_source_labels_join_col(
            source_columns,
            config.contract.entity_id_col,
            config.labels_join_col,
        )
        if not source_join_col or not config.labels_join_col:
            return df

        labels_df = self._read_table(config.labels_table)
        if config.labels_order_col:
            rank_window = Window.partitionBy(_spark_col(config.labels_join_col)).orderBy(_spark_col(config.labels_order_col).desc())
            labels_df = (
                labels_df
                .withColumn("_model_lens_label_rank", F.row_number().over(rank_window))
                .filter(F.col("_model_lens_label_rank") == 1)
                .drop("_model_lens_label_rank")
            )
        labels_df = labels_df.select(
            _spark_col(config.labels_join_col).alias("_model_lens_join_key"),
            _spark_col(config.contract.label_col).alias(config.contract.label_col),
        )
        return (
            df.join(
                labels_df,
                df[source_join_col] == labels_df["_model_lens_join_key"],
                "left",
            )
            .drop("_model_lens_join_key")
        )

    def get_source_date_range(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[str | None, str | None]:
        df, _ = self._base_source_df(config, start_date=start_date, end_date=end_date, selected_columns=())
        row = (
            df.agg(
                F.date_format(F.min(_spark_col(config.contract.timestamp_col)), "yyyy-MM-dd").alias("min_date"),
                F.date_format(F.max(_spark_col(config.contract.timestamp_col)), "yyyy-MM-dd").alias("max_date"),
            )
            .first()
        )
        return _iso_date(row["min_date"]), _iso_date(row["max_date"])

    def get_source_profile(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        df = self.load_source_df(config, start_date=start_date, end_date=end_date, selected_columns=config.contract.feature_columns)
        prediction = _spark_col(config.contract.prediction_col).cast("double")
        summary = (
            df.agg(
                F.count("*").alias("total_rows"),
                F.date_format(F.min(_spark_col(config.contract.timestamp_col)), "yyyy-MM-dd").alias("min_date"),
                F.date_format(F.max(_spark_col(config.contract.timestamp_col)), "yyyy-MM-dd").alias("max_date"),
                F.avg(prediction).alias("prediction_mean"),
                F.stddev_samp(prediction).alias("prediction_std"),
                (
                    F.sum(F.when(_spark_col(config.contract.label_col).isNotNull(), F.lit(1)).otherwise(F.lit(0)))
                    if config.contract.label_col and config.contract.label_col in df.columns
                    else F.lit(0)
                ).alias("label_row_count"),
            )
            .first()
        )
        total_rows = int(summary["total_rows"] or 0)
        if total_rows <= 0:
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

        event_date = F.to_date(_spark_col(config.contract.timestamp_col)).alias("profile_date")
        daily_volume_rows = (
            df.select(event_date)
            .groupBy("profile_date")
            .agg(F.count("*").alias("row_count"))
            .orderBy("profile_date")
        )
        daily_volume = {
            str(row["profile_date"]): int(row["row_count"] or 0)
            for row in _iter_local_rows(daily_volume_rows)
            if row["profile_date"] is not None
        }
        null_rates: dict[str, float] = {}
        if config.contract.feature_columns:
            null_exprs = [
                F.round(
                    F.avg(F.when(_spark_col(feature).isNull(), F.lit(100.0)).otherwise(F.lit(0.0))),
                    2,
                ).alias(feature)
                for feature in config.contract.feature_columns
                if feature in df.columns
            ]
            if null_exprs:
                null_row = df.agg(*null_exprs).first().asDict()
                null_rates = {
                    str(feature): round(float(value), 2)
                    for feature, value in null_row.items()
                    if value is not None
                }
        return {
            "total_rows": total_rows,
            "min_date": _iso_date(summary["min_date"]),
            "max_date": _iso_date(summary["max_date"]),
            "prediction_mean": _safe_float(summary["prediction_mean"]),
            "prediction_std": _safe_float(summary["prediction_std"]),
            "daily_volume": daily_volume,
            "null_rates": null_rates,
            "label_row_count": int(summary["label_row_count"] or 0),
        }

    def get_label_watermark(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str | None:
        if config.labels_table and config.labels_order_col:
            row = self._read_table(config.labels_table).agg(F.max(_spark_col(config.labels_order_col)).cast("string").alias("watermark")).first()
            return str(row["watermark"]) if row["watermark"] is not None else None
        if not config.contract.label_col:
            return None
        df, _ = self._base_source_df(config, start_date=start_date, end_date=end_date, selected_columns=())
        if config.contract.label_col not in df.columns:
            return None
        row = (
            df.filter(_spark_col(config.contract.label_col).isNotNull())
            .agg(
                F.max(_spark_col(config.contract.timestamp_col)).cast("string").alias("watermark"),
                F.count("*").alias("label_count"),
            )
            .first()
        )
        if row["watermark"] is None:
            return None
        return f"{row['watermark']}|{int(row['label_count'] or 0)}"

    def get_existing_window_keys(self, model_key: str) -> set[tuple[str, str, str, str]]:
        try:
            frame = (
                self._read_table(self._table_names.comparison_windows)
                .filter(F.col("model_key") == F.lit(model_key))
                .select(
                    F.date_format(F.col("baseline_start"), "yyyy-MM-dd").alias("baseline_start"),
                    F.date_format(F.col("baseline_end"), "yyyy-MM-dd").alias("baseline_end"),
                    F.date_format(F.col("window_start"), "yyyy-MM-dd").alias("window_start"),
                    F.date_format(F.col("window_end"), "yyyy-MM-dd").alias("window_end"),
                )
                .distinct()
            )
            rows = list(_iter_local_rows(frame))
        except Exception:
            rows = []
        if not rows:
            try:
                rows = list(_iter_local_rows(
                    self._read_table(self._table_names.drift_metrics)
                    .filter(F.col("model_key") == F.lit(model_key))
                    .select(
                        F.date_format(F.col("baseline_start"), "yyyy-MM-dd").alias("baseline_start"),
                        F.date_format(F.col("baseline_end"), "yyyy-MM-dd").alias("baseline_end"),
                        F.date_format(F.col("window_start"), "yyyy-MM-dd").alias("window_start"),
                        F.date_format(F.col("window_end"), "yyyy-MM-dd").alias("window_end"),
                    )
                    .distinct()
                ))
            except Exception:
                rows = []
        return {
            (
                _iso_date(row["baseline_start"]) or "",
                _iso_date(row["baseline_end"]) or "",
                _iso_date(row["window_start"]) or "",
                _iso_date(row["window_end"]) or "",
            )
            for row in rows
        }

    def get_performance_bin_specs(self, model_key: str) -> dict[str, tuple[float, ...]]:
        try:
            rows = list(_iter_local_rows(
                self._read_table(self._table_names.performance_bin_specs)
                .filter(F.col("model_key") == F.lit(model_key))
                .select("feature_name", "edges_json")
                .orderBy("feature_name")
            ))
        except Exception:
            return {}
        specs: dict[str, tuple[float, ...]] = {}
        for row in rows:
            feature_name = str(row["feature_name"] or "").strip()
            if not feature_name:
                continue
            try:
                loaded = json.loads(str(row["edges_json"] or ""))
            except Exception:
                continue
            if not isinstance(loaded, list):
                continue
            edges = tuple(float(value) for value in loaded)
            if len(edges) < 2:
                continue
            specs[feature_name] = edges
        return specs

    def replace_performance_bin_specs(self, model_key: str, specs: dict[str, tuple[float, ...]]) -> None:
        self._delete_where(
            self._table_names.performance_bin_specs,
            f"model_key = {_sql_string_literal(model_key)}",
        )
        self._append_df_to_table(
            self._table_names.performance_bin_specs,
            self._performance_bin_spec_df_from_specs(model_key, specs),
        )

    def get_current_incident_state(self, model_key: str) -> dict[tuple[str, str, str], dict[str, Any]]:
        try:
            rows = _iter_local_rows(
                self._read_table(self._table_names.incidents)
                .filter((F.col("model_key") == F.lit(model_key)) & (F.col("status") == F.lit("open")))
                .select(
                    "model_key",
                    "feature_name",
                    "metric_name",
                    "severity",
                    "status",
                    "metric_value",
                    F.date_format(F.col("window_end"), "yyyy-MM-dd").alias("window_end"),
                    F.col("observed_at").cast("string").alias("observed_at"),
                )
            )
        except Exception:
            return {}
        state: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (
                str(row["model_key"] or ""),
                str(row["feature_name"] or ""),
                str(row["metric_name"] or ""),
            )
            state[key] = {
                "model_key": key[0],
                "feature_name": key[1],
                "metric_name": key[2],
                "severity": str(row["severity"] or ""),
                "status": str(row["status"] or "open") or "open",
                "metric_value": float(row["metric_value"] or 0.0),
                "window_end": str(row["window_end"] or ""),
                "observed_at": str(row["observed_at"] or ""),
            }
        return state

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
        df = (
            self.load_source_df(
                config,
                start_date=start_date,
                end_date=end_date,
                selected_columns=config.contract.feature_columns,
            )
            .withColumn("_model_lens_profile_date", F.to_date(_spark_col(config.contract.timestamp_col)))
            .filter(F.col("_model_lens_profile_date").isNotNull())
        )
        df = df.persist()
        try:
            numeric_bin_specs = (
                self._build_performance_bin_specs(
                    config=config,
                    source_df=df,
                    existing_specs=existing_bin_specs,
                )
                if include_drift_quality or include_performance
                else {}
            )
            daily_quality_rows = (
                self._build_daily_quality_profile_rows(config=config, source_df=df, computed_at=computed_at)
                if include_drift_quality
                else []
            )
            daily_feature_rows = (
                self._build_daily_feature_profile_rows(
                    config=config,
                    source_df=df,
                    computed_at=computed_at,
                    bin_specs=numeric_bin_specs,
                )
                if include_drift_quality
                else []
            )
            daily_performance_rows = (
                self._build_daily_performance_profile_rows(
                    config=config,
                    source_df=df,
                    computed_at=computed_at,
                    bin_specs=numeric_bin_specs,
                )
                if include_performance
                else []
            )
            return SparkDailyProfiles(
                daily_quality_profile_rows=daily_quality_rows,
                daily_feature_profile_rows=daily_feature_rows,
                daily_performance_profile_rows=daily_performance_rows,
                performance_bin_specs=numeric_bin_specs,
            )
        finally:
            df.unpersist()

    def _daily_quality_profile_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        default_model_key: str | None = None,
    ) -> DataFrame:
        frame = self._typed_df_from_rows(
            rows,
            schema=QUALITY_PROFILE_SCHEMA,
            default_model_key=default_model_key,
            row_kind="daily_quality_profile_current",
        )
        return frame.withColumn("profile_date", F.to_date(F.col("profile_date")))

    def _daily_feature_profile_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        default_model_key: str | None = None,
    ) -> DataFrame:
        frame = self._typed_df_from_rows(
            rows,
            schema=FEATURE_PROFILE_SCHEMA,
            default_model_key=default_model_key,
            row_kind="daily_feature_profile_current",
        )
        return frame.withColumn("profile_date", F.to_date(F.col("profile_date")))

    def _daily_performance_profile_df_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        default_model_key: str | None = None,
    ) -> DataFrame:
        frame = self._typed_df_from_rows(
            rows,
            schema=PERFORMANCE_PROFILE_SCHEMA,
            default_model_key=default_model_key,
            row_kind="daily_performance_profile_current",
        )
        return frame.withColumn("profile_date", F.to_date(F.col("profile_date")))

    def _metadata_df(self, metadata_list: list[dict[str, str]], *, default_model_key: str | None = None) -> DataFrame:
        frame = self._typed_df_from_rows(
            metadata_list,
            schema=METADATA_SCHEMA,
            default_model_key=default_model_key,
            row_kind="comparison_window_metadata",
        )
        return (
            frame
            .withColumn("window_start", F.to_date(F.col("window_start")))
            .withColumn("window_end", F.to_date(F.col("window_end")))
            .withColumn("baseline_start", F.to_date(F.col("baseline_start")))
            .withColumn("baseline_end", F.to_date(F.col("baseline_end")))
        )

    def _load_persisted_daily_quality_profile_df(self, model_key: str, start_date: str, end_date: str) -> DataFrame:
        return (
            self._read_table(self._table_names.daily_quality_profiles)
            .filter(F.col("model_key") == F.lit(model_key))
            .filter(F.col("profile_date") >= F.to_date(F.lit(start_date)))
            .filter(F.col("profile_date") <= F.to_date(F.lit(end_date)))
            .select(*QUALITY_PROFILE_SCHEMA.fieldNames())
        )

    def _load_persisted_daily_feature_profile_df(self, model_key: str, start_date: str, end_date: str) -> DataFrame:
        return (
            self._read_table(self._table_names.daily_feature_profiles)
            .filter(F.col("model_key") == F.lit(model_key))
            .filter(F.col("profile_date") >= F.to_date(F.lit(start_date)))
            .filter(F.col("profile_date") <= F.to_date(F.lit(end_date)))
            .select(*FEATURE_PROFILE_SCHEMA.fieldNames())
        )

    def _load_persisted_daily_performance_profile_df(self, model_key: str, start_date: str, end_date: str) -> DataFrame:
        return (
            self._read_table(self._table_names.daily_performance_profiles)
            .filter(F.col("model_key") == F.lit(model_key))
            .filter(F.col("profile_date") >= F.to_date(F.lit(start_date)))
            .filter(F.col("profile_date") <= F.to_date(F.lit(end_date)))
            .select(*PERFORMANCE_PROFILE_SCHEMA.fieldNames())
        )

    def _merge_profile_df(
        self,
        *,
        persisted_df: DataFrame,
        current_df: DataFrame,
        key_columns: tuple[str, ...],
    ) -> DataFrame:
        if not current_df.take(1):
            return persisted_df
        current_keys = current_df.select(*key_columns).distinct()
        persisted_remaining = persisted_df.join(current_keys, on=list(key_columns), how="left_anti")
        return persisted_remaining.unionByName(current_df)

    def _incident_threshold_columns(self) -> tuple[F.Column, F.Column]:
        warning_threshold = (
            F.when(F.col("metric_name") == F.lit("psi"), F.lit(float(DEFAULT_THRESHOLDS["psi"]["warning"])))
            .when(F.col("metric_name") == F.lit("js_divergence"), F.lit(float(DEFAULT_THRESHOLDS["js_divergence"]["warning"])))
            .when(F.col("metric_name") == F.lit("kl_divergence"), F.lit(float(DEFAULT_THRESHOLDS["kl_divergence"]["warning"])))
        )
        critical_threshold = (
            F.when(F.col("metric_name") == F.lit("psi"), F.lit(float(DEFAULT_THRESHOLDS["psi"]["critical"])))
            .when(F.col("metric_name") == F.lit("js_divergence"), F.lit(float(DEFAULT_THRESHOLDS["js_divergence"]["critical"])))
            .when(F.col("metric_name") == F.lit("kl_divergence"), F.lit(float(DEFAULT_THRESHOLDS["kl_divergence"]["critical"])))
        )
        return warning_threshold, critical_threshold

    def _incident_candidates_df_from_drift_df(self, drift_df: DataFrame) -> DataFrame:
        warning_threshold, critical_threshold = self._incident_threshold_columns()
        return (
            drift_df
            .withColumn("_warning_threshold", warning_threshold)
            .withColumn("_critical_threshold", critical_threshold)
            .withColumn(
                "_severity_rank",
                F.when(
                    F.col("_critical_threshold").isNotNull() & (F.col("metric_value") >= F.col("_critical_threshold")),
                    F.lit(2),
                ).when(
                    F.col("_warning_threshold").isNotNull() & (F.col("metric_value") >= F.col("_warning_threshold")),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
            .filter(F.col("_severity_rank") > 0)
            .withColumn(
                "severity",
                F.when(F.col("_severity_rank") >= F.lit(2), F.lit("critical")).otherwise(F.lit("warning")),
            )
            .withColumn("observed_at", F.col("computed_at").cast("string"))
        )

    def _derive_incident_rows_from_drift_rows(
        self,
        *,
        drift_rows: list[dict[str, Any]],
        latest_window_end: str,
    ) -> list[dict[str, Any]]:
        if not drift_rows or not latest_window_end:
            return []
        candidates = self._incident_candidates_df_from_drift_df(self._drift_df_from_rows(drift_rows)).filter(
            F.col("window_end") == F.to_date(F.lit(latest_window_end))
        )
        if not candidates.take(1):
            return []
        selection_window = Window.partitionBy("model_key", "feature_name", "metric_name").orderBy(
            F.col("_severity_rank").desc(),
            F.col("metric_value").desc(),
        )
        rows = (
            candidates
            .withColumn("_incident_rank", F.row_number().over(selection_window))
            .filter(F.col("_incident_rank") == F.lit(1))
            .select(
                "model_key",
                "feature_name",
                "metric_name",
                "severity",
                F.lit("open").alias("status"),
                "metric_value",
                F.date_format(F.col("window_end"), "yyyy-MM-dd").alias("window_end"),
                "observed_at",
            )
        )
        return [row.asDict() for row in _iter_local_rows(rows)]

    def _derive_incident_history_rows_from_drift_rows(
        self,
        *,
        drift_rows: list[dict[str, Any]],
        window_rows: list[dict[str, Any]],
        prior_open_incidents: dict[tuple[str, str, str], dict[str, Any]] | None = None,
        computed_at: str,
    ) -> list[dict[str, Any]]:
        if not window_rows:
            return []
        candidates = self._incident_candidates_df_from_drift_df(self._drift_df_from_rows(drift_rows))
        state_window = Window.partitionBy(
            "window_id",
            "model_key",
            "feature_name",
            "metric_name",
        ).orderBy(
            F.col("_severity_rank").desc(),
            F.col("metric_value").desc(),
        )
        current_state_df = (
            candidates
            .withColumn("_state_rank", F.row_number().over(state_window))
            .filter(F.col("_state_rank") == F.lit(1))
            .select(
                "window_id",
                "model_key",
                "feature_name",
                "metric_name",
                "severity",
                F.col("_severity_rank").alias("severity_rank"),
                "metric_value",
            )
        )
        window_df = (
            self._window_df_from_rows(window_rows)
            .select("window_id", "window_start", "window_end", "baseline_start", "baseline_end", "created_at")
        )
        ordered_windows = (
            window_df
            .withColumn(
                "window_order",
                F.row_number().over(Window.orderBy("window_end", "window_start", "window_id")),
            )
            .withColumn("observed_at", F.coalesce(F.col("created_at").cast("string"), F.lit(computed_at)))
        )
        prior_payload = [
            {
                "model_key": key[0],
                "feature_name": key[1],
                "metric_name": key[2],
                "prior_severity": str(value.get("severity") or ""),
                "prior_metric_value": float(value.get("metric_value") or 0.0),
                "prior_present": True,
            }
            for key, value in (prior_open_incidents or {}).items()
        ]
        prior_df = self._spark.createDataFrame(
            prior_payload or [],
            schema=StructType([
                StructField("model_key", StringType(), False),
                StructField("feature_name", StringType(), False),
                StructField("metric_name", StringType(), False),
                StructField("prior_severity", StringType(), False),
                StructField("prior_metric_value", DoubleType(), False),
                StructField("prior_present", BooleanType(), True),
            ]),
        )
        if prior_payload:
            prior_df = (
                prior_df
                .withColumn("prior_present", F.lit(True))
                .withColumn(
                    "prior_severity_rank",
                    F.when(F.col("prior_severity") == F.lit("critical"), F.lit(2))
                    .when(F.col("prior_severity") == F.lit("warning"), F.lit(1))
                    .otherwise(F.lit(0)),
                )
            )
        else:
            prior_df = (
                prior_df
                .withColumn("prior_present", F.lit(False))
                .withColumn("prior_severity_rank", F.lit(0))
            )
        key_df = current_state_df.select("model_key", "feature_name", "metric_name").distinct()
        if prior_payload:
            key_df = key_df.unionByName(
                prior_df.select("model_key", "feature_name", "metric_name").distinct()
            ).distinct()
        timeline = (
            key_df.crossJoin(ordered_windows)
            .join(current_state_df, on=["window_id", "model_key", "feature_name", "metric_name"], how="left")
            .join(prior_df, on=["model_key", "feature_name", "metric_name"], how="left")
            .withColumn("current_present", F.col("severity_rank").isNotNull())
        )
        event_partition = Window.partitionBy("model_key", "feature_name", "metric_name").orderBy("window_order")
        timeline = (
            timeline
            .withColumn(
                "previous_present",
                F.coalesce(
                    F.lag("current_present").over(event_partition),
                    F.col("prior_present"),
                    F.lit(False),
                ),
            )
            .withColumn(
                "previous_severity",
                F.coalesce(
                    F.lag("severity").over(event_partition),
                    F.col("prior_severity"),
                ),
            )
            .withColumn(
                "previous_severity_rank",
                F.coalesce(
                    F.lag("severity_rank").over(event_partition),
                    F.col("prior_severity_rank"),
                    F.lit(0),
                ),
            )
            .withColumn(
                "event_type",
                F.when(
                    F.col("current_present") & ~F.col("previous_present"),
                    F.lit("opened"),
                ).when(
                    F.col("current_present") & F.col("previous_present") & (F.col("severity_rank") > F.col("previous_severity_rank")),
                    F.lit("escalated"),
                ).when(
                    F.col("current_present") & F.col("previous_present") & (F.col("severity_rank") < F.col("previous_severity_rank")),
                    F.lit("downgraded"),
                ).when(
                    F.col("current_present") & F.col("previous_present"),
                    F.lit("ongoing"),
                ).when(
                    ~F.col("current_present") & F.col("previous_present"),
                    F.lit("recovered"),
                ),
            )
            .filter(F.col("event_type").isNotNull())
            .select(
                "model_key",
                "feature_name",
                "metric_name",
                "event_type",
                F.when(F.col("current_present"), F.col("severity")).otherwise(F.col("previous_severity")).alias("severity"),
                F.when(F.col("event_type") == F.lit("recovered"), F.lit("closed")).otherwise(F.lit("open")).alias("status"),
                F.when(F.col("current_present"), F.col("metric_value")).otherwise(F.lit(0.0)).alias("metric_value"),
                "window_id",
                F.date_format(F.col("window_start"), "yyyy-MM-dd").alias("window_start"),
                F.date_format(F.col("window_end"), "yyyy-MM-dd").alias("window_end"),
                F.date_format(F.col("baseline_start"), "yyyy-MM-dd").alias("baseline_start"),
                F.date_format(F.col("baseline_end"), "yyyy-MM-dd").alias("baseline_end"),
                "observed_at",
            )
            .orderBy("window_end", "window_start", "window_id", "feature_name", "metric_name")
        )
        return [row.asDict() for row in _iter_local_rows(timeline)]

    def _rewrite_quality_summary(self, model_key: str) -> None:
        self._delete_where(
            self._table_names.quality_metrics,
            f"model_key = {_sql_string_literal(model_key)}",
        )
        quality_df = (
            self._read_table(self._table_names.daily_quality_profiles)
            .filter(F.col("model_key") == F.lit(model_key))
            .withColumn("_null_rates_map", F.from_json(F.col("null_rates"), _NULL_RATE_SCHEMA))
        )
        if not quality_df.take(1):
            return
        aggregate = (
            quality_df.agg(
                F.sum("row_count").alias("total_rows"),
                F.date_format(F.min("profile_date"), "yyyy-MM-dd").alias("min_date"),
                F.date_format(F.max("profile_date"), "yyyy-MM-dd").alias("max_date"),
                F.sum(
                    F.when(
                        F.col("prediction_mean").isNotNull(),
                        F.col("row_count") * F.col("prediction_mean"),
                    ).otherwise(F.lit(0.0))
                ).alias("_sum_x"),
                F.sum(
                    F.when(
                        F.col("prediction_mean").isNotNull(),
                        (
                            F.when(
                                F.col("prediction_std").isNotNull() & (F.col("row_count") > 1),
                                (F.col("row_count") - 1) * F.pow(F.col("prediction_std"), 2),
                            ).otherwise(F.lit(0.0))
                            + (F.col("row_count") * F.pow(F.col("prediction_mean"), 2))
                        ),
                    ).otherwise(F.lit(0.0))
                ).alias("_variance_terms"),
            )
            .first()
        )
        total_rows = int(aggregate["total_rows"] or 0)
        if total_rows <= 0:
            return
        prediction_mean = float(aggregate["_sum_x"] or 0.0) / total_rows
        variance_numerator = max(float(aggregate["_variance_terms"] or 0.0) - (total_rows * (prediction_mean ** 2)), 0.0)
        prediction_std = (variance_numerator / (total_rows - 1)) ** 0.5 if total_rows > 1 else None
        daily_volume = {
            str(row["profile_date"]): int(row["row_count"] or 0)
            for row in _iter_local_rows(
                quality_df.select(F.date_format(F.col("profile_date"), "yyyy-MM-dd").alias("profile_date"), "row_count")
                .orderBy("profile_date")
            )
            if row["profile_date"] is not None
        }
        null_rate_rows = (
            quality_df.select("row_count", F.explode_outer(F.map_entries(F.col("_null_rates_map"))).alias("_entry"))
            .select(
                "row_count",
                F.col("_entry.key").alias("feature_name"),
                F.col("_entry.value").cast("double").alias("null_pct"),
            )
            .groupBy("feature_name")
            .agg(F.sum(F.col("null_pct") * F.col("row_count")).alias("weighted_null_sum"))
            .orderBy("feature_name")
        )
        null_rates = {
            str(row["feature_name"]): round(float(row["weighted_null_sum"]) / total_rows, 2)
            for row in _iter_local_rows(null_rate_rows)
            if row["feature_name"] is not None and row["weighted_null_sum"] is not None
        }
        summary_rows = [{
            "model_key": model_key,
            "total_rows": total_rows,
            "min_date": str(aggregate["min_date"]) if aggregate["min_date"] is not None else None,
            "max_date": str(aggregate["max_date"]) if aggregate["max_date"] is not None else None,
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "daily_volume": json.dumps(daily_volume),
            "null_rates": json.dumps(null_rates),
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }]
        self._append_df_to_table(
            self._table_names.quality_metrics,
            self._quality_metric_df_from_rows(summary_rows),
        )

    def replace_all_refresh_results(self, model_key: str, result: RefreshResult, source_run_id: str | None = None) -> None:
        for table_name in (
            self._table_names.drift_metrics,
            self._table_names.performance_metrics,
            self._table_names.quality_metrics,
            self._table_names.quality_history,
            self._table_names.daily_quality_profiles,
            self._table_names.daily_feature_profiles,
            self._table_names.daily_performance_profiles,
            self._table_names.performance_bin_specs,
            self._table_names.incidents,
            self._table_names.incident_history,
            self._table_names.comparison_windows,
        ):
            self._delete_where(table_name, f"model_key = {_sql_string_literal(model_key)}")
        self._append_df_to_table(
            self._table_names.comparison_windows,
            self._window_df_from_rows(result.window_rows, source_run_id=source_run_id, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.drift_metrics,
            self._drift_df_from_rows(result.drift_rows, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.quality_history,
            self._quality_history_df_from_rows(result.quality_history_rows, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.daily_quality_profiles,
            self._daily_quality_profile_persist_df_from_rows(
                result.daily_quality_profile_rows,
                source_run_id=source_run_id,
                default_model_key=model_key,
            ),
        )
        self._append_df_to_table(
            self._table_names.daily_feature_profiles,
            self._daily_feature_profile_persist_df_from_rows(
                result.daily_feature_profile_rows,
                source_run_id=source_run_id,
                default_model_key=model_key,
            ),
        )
        self._append_df_to_table(
            self._table_names.performance_metrics,
            self._performance_df_from_rows(result.performance_rows, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.daily_performance_profiles,
            self._daily_performance_profile_persist_df_from_rows(
                result.daily_performance_profile_rows,
                source_run_id=source_run_id,
                default_model_key=model_key,
            ),
        )
        self.replace_performance_bin_specs(model_key, result.performance_bin_specs)
        self._append_df_to_table(
            self._table_names.incidents,
            self._incident_df_from_rows(result.incident_rows, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.incident_history,
            self._incident_history_df_from_rows(result.incident_history_rows, default_model_key=model_key),
        )
        self._rewrite_quality_summary(model_key)
        self._sync_read_model()

    def append_refresh_result(self, model_key: str, result: RefreshResult, source_run_id: str | None = None) -> None:
        model_key_predicate = f"model_key = {_sql_string_literal(model_key)}"

        def _delete_string_values(table_name: str, column_name: str, values: set[str]) -> None:
            cleaned = sorted(value for value in values if value)
            if not cleaned:
                return
            literals = ", ".join(_sql_string_literal(value) for value in cleaned)
            self._delete_where(
                table_name,
                f"{model_key_predicate} AND {column_name} IN ({literals})",
            )

        def _delete_date_values(table_name: str, column_name: str, values: set[str]) -> None:
            cleaned = sorted(value for value in values if value)
            if not cleaned:
                return
            literals = ", ".join(_sql_date_literal(value) for value in cleaned)
            self._delete_where(
                table_name,
                f"{model_key_predicate} AND {column_name} IN ({literals})",
            )

        def _delete_composite_windows(
            table_name: str,
            windows: set[tuple[str, ...]],
            columns: tuple[str, ...],
        ) -> None:
            predicates = []
            for window in sorted(windows):
                if any(not value for value in window):
                    continue
                clauses = [
                    f"{column_name} = {_sql_date_literal(column_value)}"
                    for column_name, column_value in zip(columns, window, strict=True)
                ]
                predicates.append("(" + " AND ".join(clauses) + ")")
            if not predicates:
                return
            self._delete_where(
                table_name,
                f"{model_key_predicate} AND (" + " OR ".join(predicates) + ")",
            )

        drift_windows = {
            (
                str(row.get("baseline_start") or ""),
                str(row.get("baseline_end") or ""),
                str(row.get("window_start") or ""),
                str(row.get("window_end") or ""),
            )
            for row in result.drift_rows
        }
        performance_windows = {
            (
                str(row.get("window_start") or ""),
                str(row.get("window_end") or ""),
            )
            for row in result.performance_rows
        }
        quality_windows = {str(row.get("window_id") or "") for row in result.quality_history_rows}
        incident_history_windows = {str(row.get("window_id") or "") for row in result.incident_history_rows}
        daily_quality_dates = {str(row.get("profile_date") or "") for row in result.daily_quality_profile_rows}
        daily_feature_dates = {str(row.get("profile_date") or "") for row in result.daily_feature_profile_rows}
        daily_performance_dates = {str(row.get("profile_date") or "") for row in result.daily_performance_profile_rows}
        _delete_composite_windows(
            self._table_names.drift_metrics,
            drift_windows,
            ("baseline_start", "baseline_end", "window_start", "window_end"),
        )
        _delete_composite_windows(
            self._table_names.performance_metrics,
            performance_windows,
            ("window_start", "window_end"),
        )
        _delete_string_values(
            self._table_names.comparison_windows,
            "window_id",
            {str(row.get("window_id") or "") for row in result.window_rows},
        )
        _delete_string_values(self._table_names.quality_history, "window_id", quality_windows)
        _delete_string_values(self._table_names.incident_history, "window_id", incident_history_windows)
        _delete_date_values(self._table_names.daily_quality_profiles, "profile_date", daily_quality_dates)
        _delete_date_values(self._table_names.daily_feature_profiles, "profile_date", daily_feature_dates)
        _delete_date_values(self._table_names.daily_performance_profiles, "profile_date", daily_performance_dates)

        if result.incident_rows or result.incident_history_rows:
            self._delete_where(
                self._table_names.incidents,
                model_key_predicate,
            )

        self._append_df_to_table(
            self._table_names.comparison_windows,
            self._window_df_from_rows(result.window_rows, source_run_id=source_run_id, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.drift_metrics,
            self._drift_df_from_rows(result.drift_rows, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.quality_history,
            self._quality_history_df_from_rows(result.quality_history_rows, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.daily_quality_profiles,
            self._daily_quality_profile_persist_df_from_rows(
                result.daily_quality_profile_rows,
                source_run_id=source_run_id,
                default_model_key=model_key,
            ),
        )
        self._append_df_to_table(
            self._table_names.daily_feature_profiles,
            self._daily_feature_profile_persist_df_from_rows(
                result.daily_feature_profile_rows,
                source_run_id=source_run_id,
                default_model_key=model_key,
            ),
        )
        self._append_df_to_table(
            self._table_names.performance_metrics,
            self._performance_df_from_rows(result.performance_rows, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.daily_performance_profiles,
            self._daily_performance_profile_persist_df_from_rows(
                result.daily_performance_profile_rows,
                source_run_id=source_run_id,
                default_model_key=model_key,
            ),
        )
        if result.performance_bin_specs:
            persisted_specs = self.get_performance_bin_specs(model_key)
            persisted_specs.update(result.performance_bin_specs)
            self.replace_performance_bin_specs(model_key, persisted_specs)
        self._append_df_to_table(
            self._table_names.incidents,
            self._incident_df_from_rows(result.incident_rows, default_model_key=model_key),
        )
        self._append_df_to_table(
            self._table_names.incident_history,
            self._incident_history_df_from_rows(result.incident_history_rows, default_model_key=model_key),
        )
        self._rewrite_quality_summary(model_key)
        self._sync_read_model()

    def _derive_quality_history_rows_from_df(
        self,
        *,
        config: MonitorConfig,
        metadata_df: DataFrame,
        quality_df: DataFrame,
        computed_at: str,
    ) -> list[dict[str, Any]]:
        if not metadata_df.take(1) or not quality_df.take(1):
            return []
        quality_with_maps = quality_df.withColumn(
            "_null_rates_map",
            F.from_json(F.col("null_rates"), _NULL_RATE_SCHEMA),
        )
        joined = quality_with_maps.join(
            metadata_df,
            (
                (quality_with_maps.profile_date >= metadata_df.window_start)
                & (quality_with_maps.profile_date <= metadata_df.window_end)
            ),
            "inner",
        )
        if not joined.take(1):
            return []
        agg_exprs = [
            F.sum("row_count").alias("row_count"),
            F.sum(
                F.when(
                    F.col("prediction_mean").isNotNull(),
                    F.col("row_count") * F.col("prediction_mean"),
                ).otherwise(F.lit(0.0))
            ).alias("_sum_x"),
            F.sum(
                F.when(
                    F.col("prediction_mean").isNotNull(),
                    (
                        F.when(
                            F.col("prediction_std").isNotNull() & (F.col("row_count") > 1),
                            (F.col("row_count") - 1) * F.pow(F.col("prediction_std"), 2),
                        ).otherwise(F.lit(0.0))
                        + (F.col("row_count") * F.pow(F.col("prediction_mean"), 2))
                    ),
                ).otherwise(F.lit(0.0))
            ).alias("_variance_terms"),
        ]
        null_rate_aliases: list[tuple[str, str]] = []
        for index, feature in enumerate(config.contract.feature_columns):
            alias = f"_null_rate_sum_{index}"
            null_rate_aliases.append((feature, alias))
            agg_exprs.append(
                F.sum(
                    F.coalesce(F.element_at(F.col("_null_rates_map"), F.lit(feature)), F.lit(0.0))
                    * F.col("row_count")
                ).alias(alias)
            )
        aggregated = (
            joined.groupBy(
                "window_id",
                "model_key",
                "window_start",
                "window_end",
                "baseline_start",
                "baseline_end",
            )
            .agg(*agg_exprs)
            .withColumn(
                "prediction_mean",
                F.when(F.col("row_count") > 0, F.col("_sum_x") / F.col("row_count")),
            )
            .withColumn(
                "_variance_numerator",
                F.greatest(
                    F.col("_variance_terms") - (F.col("row_count") * F.pow(F.col("prediction_mean"), 2)),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                "prediction_std",
                F.when(
                    F.col("row_count") > 1,
                    F.sqrt(F.col("_variance_numerator") / (F.col("row_count") - 1)),
                ),
            )
            .orderBy("window_end", "window_start", "window_id")
        )
        rows: list[dict[str, Any]] = []
        for row in _iter_local_rows(aggregated):
            row_count = int(row["row_count"] or 0)
            null_rates = {
                feature: round(float(row[alias]) / row_count, 2)
                for feature, alias in null_rate_aliases
                if row_count > 0 and row[alias] is not None
            }
            rows.append({
                "model_key": config.model_key,
                "window_id": str(row["window_id"]),
                "window_start": _iso_date(row["window_start"]),
                "window_end": _iso_date(row["window_end"]),
                "baseline_start": _iso_date(row["baseline_start"]),
                "baseline_end": _iso_date(row["baseline_end"]),
                "row_count": row_count,
                "prediction_mean": _safe_float(row["prediction_mean"]),
                "prediction_std": _safe_float(row["prediction_std"]),
                "null_rates": json.dumps(null_rates),
                "computed_at": computed_at,
            })
        return rows

    def _derive_performance_rows_from_df(
        self,
        *,
        config: MonitorConfig,
        metadata_df: DataFrame,
        quality_df: DataFrame,
        performance_df: DataFrame,
        computed_at: str,
    ) -> list[dict[str, Any]]:
        if not metadata_df.take(1) or not performance_df.take(1):
            return []
        window_cols = [
            "window_id",
            "model_key",
            "window_start",
            "window_end",
            "baseline_start",
            "baseline_end",
        ]
        baseline_rows = (
            performance_df.join(
                metadata_df,
                (
                    (performance_df.profile_date >= metadata_df.baseline_start)
                    & (performance_df.profile_date <= metadata_df.baseline_end)
                ),
                "inner",
            )
            .groupBy(*window_cols, "feature_name", "bin_label", "metric_name")
            .agg(
                F.sum(F.col("metric_value") * F.col("row_count")).alias("baseline_weighted_sum"),
                F.sum("row_count").alias("baseline_row_count"),
            )
        )
        current_rows = (
            performance_df.join(
                metadata_df,
                (
                    (performance_df.profile_date >= metadata_df.window_start)
                    & (performance_df.profile_date <= metadata_df.window_end)
                ),
                "inner",
            )
            .groupBy(*window_cols, "feature_name", "bin_label", "metric_name")
            .agg(
                F.sum(F.col("metric_value") * F.col("row_count")).alias("current_weighted_sum"),
                F.sum("row_count").alias("current_row_count"),
            )
        )
        if not baseline_rows.take(1) or not current_rows.take(1):
            return []
        current_quality_totals = (
            quality_df.join(
                metadata_df,
                (
                    (quality_df.profile_date >= metadata_df.window_start)
                    & (quality_df.profile_date <= metadata_df.window_end)
                ),
                "inner",
            )
            .groupBy(*window_cols)
            .agg(F.sum("row_count").alias("total_current_rows"))
        )
        regression_mode = (config.problem_type or "classification").strip().lower() == "regression"
        joined = (
            baseline_rows
            .join(
                current_rows,
                on=window_cols + ["feature_name", "bin_label", "metric_name"],
                how="inner",
            )
            .join(current_quality_totals, on=window_cols, how="left")
            .filter((F.col("baseline_row_count") > 0) & (F.col("current_row_count") > 0))
            .withColumn("baseline_metric", F.col("baseline_weighted_sum") / F.col("baseline_row_count"))
            .withColumn("current_metric", F.col("current_weighted_sum") / F.col("current_row_count"))
            .withColumn(
                "delta",
                (
                    F.col("baseline_metric") - F.col("current_metric")
                    if regression_mode
                    else F.col("current_metric") - F.col("baseline_metric")
                ),
            )
            .withColumn(
                "_total_current_rows",
                F.greatest(F.coalesce(F.col("total_current_rows"), F.col("current_row_count")), F.lit(1)),
            )
            .withColumn("volume_pct", (F.col("current_row_count") / F.col("_total_current_rows")) * F.lit(100.0))
            .withColumn("contribution", F.col("delta") * F.col("volume_pct") / F.lit(100.0))
            .select(
                *window_cols,
                "feature_name",
                "bin_label",
                "metric_name",
                "baseline_metric",
                "current_metric",
                "delta",
                "volume_pct",
                "contribution",
            )
            .orderBy("window_end", "window_start", "feature_name", "bin_label", "metric_name")
        )
        rows: list[dict[str, Any]] = []
        for row in _iter_local_rows(joined):
            rows.append({
                "model_key": config.model_key,
                "feature_name": str(row["feature_name"]),
                "bin_label": str(row["bin_label"]),
                "baseline_metric": round(float(row["baseline_metric"]), 4),
                "current_metric": round(float(row["current_metric"]), 4),
                "delta": round(float(row["delta"]), 4),
                "volume_pct": round(float(row["volume_pct"]), 2),
                "contribution": round(float(row["contribution"]), 4),
                "metric_name": str(row["metric_name"]),
                "window_start": _iso_date(row["window_start"]),
                "window_end": _iso_date(row["window_end"]),
                "computed_at": computed_at,
            })
        return rows

    def _derive_categorical_drift_rows_from_df(
        self,
        *,
        config: MonitorConfig,
        metadata_df: DataFrame,
        feature_df: DataFrame,
        computed_at: str,
    ) -> list[dict[str, Any]]:
        categorical_features = tuple(
            feature
            for feature in config.contract.categorical_columns
            if feature in config.contract.feature_columns
        )
        if not categorical_features:
            return []
        categorical_df = (
            feature_df
            .filter(F.col("feature_kind") == F.lit("categorical"))
            .filter(F.col("feature_name").isin(list(categorical_features)))
            .withColumn(
                "_counts_map",
                F.from_json(F.col("distribution_json"), _CATEGORICAL_COUNTS_SCHEMA),
            )
        )
        if not categorical_df.take(1):
            return []
        window_cols = [
            "window_id",
            "model_key",
            "window_start",
            "window_end",
            "baseline_start",
            "baseline_end",
        ]
        baseline_stats = (
            categorical_df.join(
                metadata_df,
                (
                    (categorical_df.profile_date >= metadata_df.baseline_start)
                    & (categorical_df.profile_date <= metadata_df.baseline_end)
                ),
                "inner",
            )
            .groupBy(*window_cols, "feature_name")
            .agg(
                F.sum("row_count").alias("ref_total_rows"),
                F.sum("non_null_count").alias("ref_non_null_count"),
            )
        )
        current_stats = (
            categorical_df.join(
                metadata_df,
                (
                    (categorical_df.profile_date >= metadata_df.window_start)
                    & (categorical_df.profile_date <= metadata_df.window_end)
                ),
                "inner",
            )
            .groupBy(*window_cols, "feature_name")
            .agg(
                F.sum("row_count").alias("cur_total_rows"),
                F.sum("non_null_count").alias("cur_non_null_count"),
            )
        )
        baseline_counts = (
            categorical_df.join(
                metadata_df,
                (
                    (categorical_df.profile_date >= metadata_df.baseline_start)
                    & (categorical_df.profile_date <= metadata_df.baseline_end)
                ),
                "inner",
            )
            .select(
                *window_cols,
                "feature_name",
                F.explode_outer(F.col("_counts_map")).alias("category", "category_count"),
            )
            .groupBy(*window_cols, "feature_name", "category")
            .agg(F.sum("category_count").alias("category_count"))
            .withColumn("role", F.lit("baseline"))
        )
        current_counts = (
            categorical_df.join(
                metadata_df,
                (
                    (categorical_df.profile_date >= metadata_df.window_start)
                    & (categorical_df.profile_date <= metadata_df.window_end)
                ),
                "inner",
            )
            .select(
                *window_cols,
                "feature_name",
                F.explode_outer(F.col("_counts_map")).alias("category", "category_count"),
            )
            .groupBy(*window_cols, "feature_name", "category")
            .agg(F.sum("category_count").alias("category_count"))
            .withColumn("role", F.lit("current"))
        )
        counts = (
            baseline_counts.unionByName(current_counts)
            .groupBy(*window_cols, "feature_name", "category")
            .pivot("role", ["baseline", "current"])
            .sum("category_count")
            .na.fill(0.0, ["baseline", "current"])
            .withColumnRenamed("baseline", "ref_count")
            .withColumnRenamed("current", "cur_count")
        )
        if not counts.take(1):
            return []
        partition = Window.partitionBy(*window_cols, "feature_name")
        counts = (
            counts
            .withColumn("_ref_total", F.sum("ref_count").over(partition))
            .withColumn("_cur_total", F.sum("cur_count").over(partition))
            .filter((F.col("_ref_total") > 0) & (F.col("_cur_total") > 0))
            .withColumn("_ref_prop_raw", (F.col("ref_count") / F.col("_ref_total")) + F.lit(EPSILON))
            .withColumn("_cur_prop_raw", (F.col("cur_count") / F.col("_cur_total")) + F.lit(EPSILON))
            .withColumn("_ref_prop_denom", F.sum("_ref_prop_raw").over(partition))
            .withColumn("_cur_prop_denom", F.sum("_cur_prop_raw").over(partition))
            .withColumn("ref_prop", F.col("_ref_prop_raw") / F.col("_ref_prop_denom"))
            .withColumn("cur_prop", F.col("_cur_prop_raw") / F.col("_cur_prop_denom"))
            .withColumn("midpoint", (F.col("ref_prop") + F.col("cur_prop")) / F.lit(2.0))
            .withColumn("psi_component", (F.col("cur_prop") - F.col("ref_prop")) * F.log(F.col("cur_prop") / F.col("ref_prop")))
            .withColumn("kl_component", F.col("cur_prop") * F.log(F.col("cur_prop") / F.col("ref_prop")))
            .withColumn(
                "js_component",
                F.lit(0.5) * (
                    (F.col("ref_prop") * F.log2(F.col("ref_prop") / F.col("midpoint")))
                    + (F.col("cur_prop") * F.log2(F.col("cur_prop") / F.col("midpoint")))
                ),
            )
        )
        metrics = (
            counts.groupBy(*window_cols, "feature_name")
            .agg(
                F.sum("psi_component").alias("psi"),
                F.sum("kl_component").alias("kl_divergence"),
                F.sum("js_component").alias("js_divergence"),
            )
            .join(baseline_stats, on=window_cols + ["feature_name"], how="inner")
            .join(current_stats, on=window_cols + ["feature_name"], how="inner")
            .orderBy("window_end", "window_start", "feature_name")
        )
        rows: list[dict[str, Any]] = []
        for row in _iter_local_rows(metrics):
            ref_total_rows = int(row["ref_total_rows"] or 0)
            cur_total_rows = int(row["cur_total_rows"] or 0)
            common_payload = {
                "model_key": config.model_key,
                "feature_name": str(row["feature_name"]),
                "window_id": str(row["window_id"]),
                "window_start": _iso_date(row["window_start"]),
                "window_end": _iso_date(row["window_end"]),
                "baseline_start": _iso_date(row["baseline_start"]),
                "baseline_end": _iso_date(row["baseline_end"]),
                "ref_mean": float("nan"),
                "cur_mean": float("nan"),
                "ref_std": float("nan"),
                "cur_std": float("nan"),
                "ref_null_pct": round(float((ref_total_rows - int(row["ref_non_null_count"] or 0)) / ref_total_rows * 100), 2) if ref_total_rows else 0.0,
                "cur_null_pct": round(float((cur_total_rows - int(row["cur_non_null_count"] or 0)) / cur_total_rows * 100), 2) if cur_total_rows else 0.0,
                "ref_count": int(row["ref_non_null_count"] or 0),
                "cur_count": int(row["cur_non_null_count"] or 0),
                "computed_at": computed_at,
            }
            for metric_name in ("psi", "kl_divergence", "js_divergence"):
                rows.append({
                    **common_payload,
                    "metric_name": metric_name,
                    "metric_value": round(float(row[metric_name]), 6),
                })
        return rows

    def _derive_numeric_drift_rows_from_df(
        self,
        *,
        config: MonitorConfig,
        metadata_df: DataFrame,
        feature_df: DataFrame,
        computed_at: str,
    ) -> list[dict[str, Any]]:
        categorical_set = set(config.contract.categorical_columns)
        numeric_features = tuple(
            feature
            for feature in config.contract.feature_columns
            if feature not in categorical_set
        )
        if not numeric_features:
            return []
        empty_array = F.array().cast(ArrayType(DoubleType()))
        numeric_df = (
            feature_df
            .filter(F.col("feature_kind") == F.lit("numeric"))
            .filter(F.col("feature_name").isin(list(numeric_features)))
            .withColumn(
                "_distribution_payload",
                F.from_json(F.col("distribution_json"), _NUMERIC_DISTRIBUTION_SCHEMA),
            )
            .withColumn(
                "_hist_edges",
                F.when(
                    F.col("_distribution_payload.edges").isNotNull(),
                    F.col("_distribution_payload.edges"),
                ).otherwise(empty_array),
            )
            .withColumn(
                "_hist_counts",
                F.when(
                    F.col("_distribution_payload.counts").isNotNull(),
                    F.col("_distribution_payload.counts"),
                ).otherwise(empty_array),
            )
            .filter(F.size(F.col("_hist_counts")) > 0)
            .filter(F.size(F.col("_hist_edges")) == (F.size(F.col("_hist_counts")) + F.lit(1)))
        )
        if not numeric_df.take(1):
            return []
        window_cols = [
            "window_id",
            "model_key",
            "window_start",
            "window_end",
            "baseline_start",
            "baseline_end",
        ]
        def _aggregate_numeric_range(date_start_col: str, date_end_col: str, prefix: str) -> tuple[DataFrame, DataFrame]:
            joined = numeric_df.join(
                metadata_df,
                (
                    (numeric_df.profile_date >= F.col(date_start_col))
                    & (numeric_df.profile_date <= F.col(date_end_col))
                ),
                "inner",
            )
            stats = (
                joined.groupBy(*window_cols, "feature_name")
                .agg(
                    F.sum("row_count").alias(f"{prefix}_total_rows"),
                    F.sum("non_null_count").alias(f"{prefix}_non_null_count"),
                    F.min("min_value").alias(f"{prefix}_min_value"),
                    F.max("max_value").alias(f"{prefix}_max_value"),
                    F.sum(
                        F.when(
                            F.col("mean").isNotNull() & (F.col("non_null_count") > 0),
                            F.col("non_null_count") * F.col("mean"),
                        ).otherwise(F.lit(0.0))
                    ).alias(f"{prefix}_sum_x"),
                    F.sum(
                        F.when(
                            F.col("mean").isNotNull() & (F.col("non_null_count") > 0),
                            (
                                F.when(
                                    F.col("std").isNotNull() & (F.col("non_null_count") > 1),
                                    (F.col("non_null_count") - 1) * F.pow(F.col("std"), 2),
                                ).otherwise(F.lit(0.0))
                                + (F.col("non_null_count") * F.pow(F.col("mean"), 2))
                            ),
                        ).otherwise(F.lit(0.0))
                    ).alias(f"{prefix}_variance_terms"),
                )
            )
            counts = (
                joined.select(
                    *window_cols,
                    "feature_name",
                    F.posexplode_outer(F.col("_hist_counts")).alias("bin_index", "bin_count"),
                    F.element_at(F.col("_hist_edges"), F.col("bin_index") + F.lit(1)).alias("bin_left"),
                    F.element_at(F.col("_hist_edges"), F.col("bin_index") + F.lit(2)).alias("bin_right"),
                )
                .groupBy(*window_cols, "feature_name", "bin_index", "bin_left", "bin_right")
                .agg(F.sum("bin_count").alias(f"{prefix}_bin_count"))
            )
            return stats, counts

        baseline_stats, baseline_bins = _aggregate_numeric_range("baseline_start", "baseline_end", "ref")
        current_stats, current_bins = _aggregate_numeric_range("window_start", "window_end", "cur")
        if not baseline_stats.take(1) or not current_stats.take(1):
            return []
        count_join_columns = window_cols + ["feature_name", "bin_index"]
        counts = (
            baseline_bins
            .join(current_bins, on=count_join_columns, how="full_outer")
            .select(
                *[
                    F.coalesce(
                        baseline_bins[column_name],
                        current_bins[column_name],
                    ).alias(column_name)
                    for column_name in count_join_columns
                ],
                F.coalesce(baseline_bins["bin_left"], current_bins["bin_left"]).alias("bin_left"),
                F.coalesce(baseline_bins["bin_right"], current_bins["bin_right"]).alias("bin_right"),
                F.coalesce(baseline_bins["ref_bin_count"], F.lit(0.0)).alias("ref_bin_count"),
                F.coalesce(current_bins["cur_bin_count"], F.lit(0.0)).alias("cur_bin_count"),
            )
        )
        if not counts.take(1):
            return []
        partition = Window.partitionBy(*window_cols, "feature_name")
        joined = (
            counts
            .withColumn("_ref_total", F.sum("ref_bin_count").over(partition))
            .withColumn("_cur_total", F.sum("cur_bin_count").over(partition))
            .filter((F.col("_ref_total") > 0) & (F.col("_cur_total") > 0))
            .withColumn("_ref_prop_raw", (F.col("ref_bin_count") / F.col("_ref_total")) + F.lit(EPSILON))
            .withColumn("_cur_prop_raw", (F.col("cur_bin_count") / F.col("_cur_total")) + F.lit(EPSILON))
            .withColumn("_ref_prop_denom", F.sum("_ref_prop_raw").over(partition))
            .withColumn("_cur_prop_denom", F.sum("_cur_prop_raw").over(partition))
            .withColumn("ref_prop", F.col("_ref_prop_raw") / F.col("_ref_prop_denom"))
            .withColumn("cur_prop", F.col("_cur_prop_raw") / F.col("_cur_prop_denom"))
            .withColumn("midpoint", (F.col("ref_prop") + F.col("cur_prop")) / F.lit(2.0))
            .withColumn("psi_component", (F.col("cur_prop") - F.col("ref_prop")) * F.log(F.col("cur_prop") / F.col("ref_prop")))
            .withColumn("kl_component", F.col("cur_prop") * F.log(F.col("cur_prop") / F.col("ref_prop")))
            .withColumn(
                "js_component",
                F.lit(0.5) * (
                    (F.col("ref_prop") * F.log2(F.col("ref_prop") / F.col("midpoint")))
                    + (F.col("cur_prop") * F.log2(F.col("cur_prop") / F.col("midpoint")))
                ),
            )
        )
        metrics = (
            joined.groupBy(*window_cols, "feature_name")
            .agg(
                F.sum("psi_component").alias("psi"),
                F.sum("kl_component").alias("kl_divergence"),
                F.sum("js_component").alias("js_divergence"),
            )
            .join(baseline_stats, on=window_cols + ["feature_name"], how="inner")
            .join(current_stats, on=window_cols + ["feature_name"], how="inner")
            .filter((F.col("ref_non_null_count") > 0) & (F.col("cur_non_null_count") > 0))
            .withColumn("ref_mean", F.col("ref_sum_x") / F.col("ref_non_null_count"))
            .withColumn("cur_mean", F.col("cur_sum_x") / F.col("cur_non_null_count"))
            .withColumn(
                "_ref_variance_numerator",
                F.greatest(
                    F.col("ref_variance_terms") - (F.col("ref_non_null_count") * F.pow(F.col("ref_mean"), 2)),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                "_cur_variance_numerator",
                F.greatest(
                    F.col("cur_variance_terms") - (F.col("cur_non_null_count") * F.pow(F.col("cur_mean"), 2)),
                    F.lit(0.0),
                ),
            )
            .withColumn("ref_std", F.sqrt(F.col("_ref_variance_numerator") / F.col("ref_non_null_count")))
            .withColumn("cur_std", F.sqrt(F.col("_cur_variance_numerator") / F.col("cur_non_null_count")))
            .withColumn(
                "ref_null_pct",
                F.when(
                    F.col("ref_total_rows") > 0,
                    ((F.col("ref_total_rows") - F.col("ref_non_null_count")) / F.col("ref_total_rows")) * F.lit(100.0),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "cur_null_pct",
                F.when(
                    F.col("cur_total_rows") > 0,
                    ((F.col("cur_total_rows") - F.col("cur_non_null_count")) / F.col("cur_total_rows")) * F.lit(100.0),
                ).otherwise(F.lit(0.0)),
            )
            .select(
                *window_cols,
                "feature_name",
                "psi",
                "kl_divergence",
                "js_divergence",
                "ref_mean",
                "cur_mean",
                "ref_std",
                "cur_std",
                "ref_null_pct",
                "cur_null_pct",
                "ref_non_null_count",
                "cur_non_null_count",
            )
            .orderBy("window_end", "window_start", "feature_name")
        )
        rows: list[dict[str, Any]] = []
        for row in _iter_local_rows(metrics):
            ref_count = int(row["ref_non_null_count"] or 0)
            cur_count = int(row["cur_non_null_count"] or 0)
            ref_mean = _safe_float(row["ref_mean"])
            cur_mean = _safe_float(row["cur_mean"])
            ref_std = _safe_float(row["ref_std"])
            cur_std = _safe_float(row["cur_std"])
            common_payload = {
                "model_key": config.model_key,
                "feature_name": str(row["feature_name"]),
                "window_id": str(row["window_id"]),
                "window_start": _iso_date(row["window_start"]),
                "window_end": _iso_date(row["window_end"]),
                "baseline_start": _iso_date(row["baseline_start"]),
                "baseline_end": _iso_date(row["baseline_end"]),
                "ref_mean": float(ref_mean) if ref_mean is not None else float("nan"),
                "cur_mean": float(cur_mean) if cur_mean is not None else float("nan"),
                "ref_std": float(ref_std) if ref_std is not None else float("nan"),
                "cur_std": float(cur_std) if cur_std is not None else float("nan"),
                "ref_null_pct": round(float(row["ref_null_pct"] or 0.0), 2),
                "cur_null_pct": round(float(row["cur_null_pct"] or 0.0), 2),
                "ref_count": ref_count,
                "cur_count": cur_count,
                "computed_at": computed_at,
            }
            for metric_name in ("psi", "kl_divergence", "js_divergence"):
                rows.append({
                    **common_payload,
                    "metric_name": metric_name,
                    "metric_value": round(float(row[metric_name]), 6),
                })
        return rows

    def _derive_drift_rows_from_df(
        self,
        *,
        config: MonitorConfig,
        metadata_df: DataFrame,
        feature_df: DataFrame,
        computed_at: str,
    ) -> list[dict[str, Any]]:
        if not metadata_df.take(1) or not feature_df.take(1):
            return []
        rows = self._derive_categorical_drift_rows_from_df(
            config=config,
            metadata_df=metadata_df,
            feature_df=feature_df,
            computed_at=computed_at,
        )
        rows.extend(
            self._derive_numeric_drift_rows_from_df(
                config=config,
                metadata_df=metadata_df,
                feature_df=feature_df,
                computed_at=computed_at,
            )
        )
        return rows

    def derive_refresh_result_from_daily_profile_rows(
        self,
        *,
        config: MonitorConfig,
        metadata_list: list[dict[str, str]],
        current_daily_quality_profile_rows: list[dict[str, Any]],
        current_daily_feature_profile_rows: list[dict[str, Any]],
        current_daily_performance_profile_rows: list[dict[str, Any]],
        derivation_start: str,
        derivation_end: str,
        computed_at: str,
        prior_open_incidents: dict[tuple[str, str, str], dict[str, Any]] | None = None,
        include_drift_quality: bool = True,
        include_performance: bool = True,
    ) -> RefreshResult:
        metadata_df = self._metadata_df(metadata_list, default_model_key=config.model_key)
        current_quality_df = self._daily_quality_profile_df_from_rows(
            current_daily_quality_profile_rows,
            default_model_key=config.model_key,
        )
        current_feature_df = self._daily_feature_profile_df_from_rows(
            current_daily_feature_profile_rows,
            default_model_key=config.model_key,
        )
        current_performance_df = self._daily_performance_profile_df_from_rows(
            current_daily_performance_profile_rows,
            default_model_key=config.model_key,
        )

        quality_df = current_quality_df
        feature_df = current_feature_df
        performance_df = current_performance_df
        if derivation_start and derivation_end:
            if include_drift_quality:
                quality_df = self._merge_profile_df(
                    persisted_df=self._load_persisted_daily_quality_profile_df(config.model_key, derivation_start, derivation_end),
                    current_df=current_quality_df,
                    key_columns=("profile_date",),
                )
                feature_df = self._merge_profile_df(
                    persisted_df=self._load_persisted_daily_feature_profile_df(config.model_key, derivation_start, derivation_end),
                    current_df=current_feature_df,
                    key_columns=("profile_date", "feature_name"),
                )
            if include_performance:
                performance_df = self._merge_profile_df(
                    persisted_df=self._load_persisted_daily_performance_profile_df(config.model_key, derivation_start, derivation_end),
                    current_df=current_performance_df,
                    key_columns=("profile_date", "feature_name", "bin_label", "metric_name"),
                )

        window_rows = [
            {
                "window_id": metadata["window_id"],
                "model_key": config.model_key,
                "window_grain": metadata["window_grain"],
                "window_start": metadata["window_start"],
                "window_end": metadata["window_end"],
                "baseline_start": metadata["baseline_start"],
                "baseline_end": metadata["baseline_end"],
                "baseline_kind": metadata["baseline_kind"],
                "created_at": computed_at,
            }
            for metadata in metadata_list
        ] if include_drift_quality else []
        quality_history_rows = (
            self._derive_quality_history_rows_from_df(
                config=config,
                metadata_df=metadata_df,
                quality_df=quality_df,
                computed_at=computed_at,
            )
            if include_drift_quality
            else []
        )
        drift_rows = (
            self._derive_drift_rows_from_df(
                config=config,
                metadata_df=metadata_df,
                feature_df=feature_df,
                computed_at=computed_at,
            )
            if include_drift_quality
            else []
        )
        performance_rows = (
            self._derive_performance_rows_from_df(
                config=config,
                metadata_df=metadata_df,
                quality_df=quality_df,
                performance_df=performance_df,
                computed_at=computed_at,
            )
            if include_performance and config.contract.label_col
            else []
        )
        latest_window_end = max((metadata["window_end"] for metadata in metadata_list), default="")
        incident_rows = (
            self._derive_incident_rows_from_drift_rows(
                drift_rows=drift_rows,
                latest_window_end=latest_window_end,
            )
            if include_drift_quality
            else []
        )
        incident_history_rows = (
            self._derive_incident_history_rows_from_drift_rows(
                drift_rows=drift_rows,
                window_rows=window_rows,
                prior_open_incidents=prior_open_incidents,
                computed_at=computed_at,
            )
            if include_drift_quality
            else []
        )
        return RefreshResult(
            drift_rows=drift_rows,
            quality_rows=[],
            performance_rows=performance_rows,
            incident_rows=incident_rows,
            incident_history_rows=incident_history_rows,
            quality_history_rows=quality_history_rows,
            window_rows=window_rows,
            daily_quality_profile_rows=current_daily_quality_profile_rows,
            daily_feature_profile_rows=current_daily_feature_profile_rows,
            daily_performance_profile_rows=current_daily_performance_profile_rows,
        )

    def _build_daily_quality_profile_rows(
        self,
        *,
        config: MonitorConfig,
        source_df: DataFrame,
        computed_at: str,
    ) -> list[dict[str, Any]]:
        prediction = _spark_col(config.contract.prediction_col).cast("double")
        agg_exprs = [
            F.count("*").alias("row_count"),
            F.avg(prediction).alias("prediction_mean"),
            F.stddev_samp(prediction).alias("prediction_std"),
            (
                F.sum(F.when(_spark_col(config.contract.label_col).isNotNull(), F.lit(1)).otherwise(F.lit(0)))
                if config.contract.label_col and config.contract.label_col in source_df.columns
                else F.lit(0)
            ).alias("label_row_count"),
        ]
        null_rate_features = [feature for feature in config.contract.feature_columns if feature in source_df.columns]
        agg_exprs.extend(
            F.round(
                F.avg(F.when(_spark_col(feature).isNull(), F.lit(100.0)).otherwise(F.lit(0.0))),
                2,
            ).alias(feature)
            for feature in null_rate_features
        )
        rows = (
            source_df.groupBy("_model_lens_profile_date")
            .agg(*agg_exprs)
            .orderBy("_model_lens_profile_date")
        )
        payload: list[dict[str, Any]] = []
        for row in _iter_local_rows(rows):
            null_rates = {
                feature: round(float(row[feature]), 2)
                for feature in null_rate_features
                if row[feature] is not None
            }
            payload.append({
                "model_key": config.model_key,
                "profile_date": str(row["_model_lens_profile_date"]),
                "row_count": int(row["row_count"] or 0),
                "prediction_mean": _safe_float(row["prediction_mean"]),
                "prediction_std": _safe_float(row["prediction_std"]),
                "null_rates": json.dumps(null_rates),
                "label_row_count": int(row["label_row_count"] or 0),
                "computed_at": computed_at,
            })
        return payload

    def _build_daily_feature_profile_rows(
        self,
        *,
        config: MonitorConfig,
        source_df: DataFrame,
        computed_at: str,
        bin_specs: dict[str, tuple[float, ...]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        categorical_set = set(config.contract.categorical_columns)
        numeric_features = [
            feature
            for feature in config.contract.feature_columns
            if feature in source_df.columns and feature not in categorical_set
        ]
        categorical_features = [
            feature
            for feature in config.contract.feature_columns
            if feature in source_df.columns and feature in categorical_set
        ]
        numeric_stats_map = self._build_daily_numeric_feature_stats_map(
            source_df=source_df,
            features=numeric_features,
        )
        categorical_stats_map = self._build_daily_categorical_feature_stats_map(
            source_df=source_df,
            features=categorical_features,
        )
        for feature in config.contract.feature_columns:
            if feature not in source_df.columns:
                continue
            if feature in categorical_set:
                rows.extend(
                    self._build_daily_categorical_feature_rows(
                        config=config,
                        source_df=source_df,
                        feature=feature,
                        computed_at=computed_at,
                        stats_by_date=categorical_stats_map.get(feature),
                    )
                )
            else:
                rows.extend(
                    self._build_daily_numeric_feature_rows(
                        config=config,
                        source_df=source_df,
                        feature=feature,
                        computed_at=computed_at,
                        bin_specs=bin_specs,
                        stats_by_date=numeric_stats_map.get(feature),
                    )
                )
        return rows

    def _build_daily_numeric_feature_stats_map(
        self,
        *,
        source_df: DataFrame,
        features: list[str],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if not features:
            return {}
        agg_exprs: list[Any] = [
            F.count("*").alias("_model_lens_row_count"),
        ]
        alias_map: dict[str, dict[str, str]] = {}
        for index, feature in enumerate(features):
            value_col = _spark_col(feature).cast("double")
            aliases = {
                "non_null_count": f"_model_lens_num_{index}_non_null_count",
                "null_pct": f"_model_lens_num_{index}_null_pct",
                "mean": f"_model_lens_num_{index}_mean",
                "std": f"_model_lens_num_{index}_std",
                "min_value": f"_model_lens_num_{index}_min_value",
                "max_value": f"_model_lens_num_{index}_max_value",
            }
            alias_map[feature] = aliases
            agg_exprs.extend([
                F.sum(F.when(value_col.isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias(aliases["non_null_count"]),
                F.round(
                    F.avg(F.when(value_col.isNull(), F.lit(100.0)).otherwise(F.lit(0.0))),
                    2,
                ).alias(aliases["null_pct"]),
                F.avg(value_col).alias(aliases["mean"]),
                F.stddev_samp(value_col).alias(aliases["std"]),
                F.min(value_col).alias(aliases["min_value"]),
                F.max(value_col).alias(aliases["max_value"]),
            ])
        rows = (
            source_df.groupBy("_model_lens_profile_date")
            .agg(*agg_exprs)
            .orderBy("_model_lens_profile_date")
        )
        stats_map = {feature: {} for feature in features}
        for row in _iter_local_rows(rows):
            profile_date = str(row["_model_lens_profile_date"])
            row_count = int(row["_model_lens_row_count"] or 0)
            for feature in features:
                aliases = alias_map[feature]
                stats_map[feature][profile_date] = {
                    "row_count": row_count,
                    "non_null_count": int(row[aliases["non_null_count"]] or 0),
                    "null_pct": round(float(row[aliases["null_pct"]] or 0.0), 2),
                    "mean": _safe_float(row[aliases["mean"]]),
                    "std": _safe_float(row[aliases["std"]]),
                    "min_value": _safe_float(row[aliases["min_value"]]),
                    "max_value": _safe_float(row[aliases["max_value"]]),
                }
        return stats_map

    def _build_daily_categorical_feature_stats_map(
        self,
        *,
        source_df: DataFrame,
        features: list[str],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if not features:
            return {}
        agg_exprs: list[Any] = [
            F.count("*").alias("_model_lens_row_count"),
        ]
        alias_map: dict[str, dict[str, str]] = {}
        for index, feature in enumerate(features):
            aliases = {
                "non_null_count": f"_model_lens_cat_{index}_non_null_count",
                "null_pct": f"_model_lens_cat_{index}_null_pct",
            }
            alias_map[feature] = aliases
            agg_exprs.extend([
                F.sum(F.when(_spark_col(feature).isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias(aliases["non_null_count"]),
                F.round(
                    F.avg(F.when(_spark_col(feature).isNull(), F.lit(100.0)).otherwise(F.lit(0.0))),
                    2,
                ).alias(aliases["null_pct"]),
            ])
        rows = (
            source_df.groupBy("_model_lens_profile_date")
            .agg(*agg_exprs)
            .orderBy("_model_lens_profile_date")
        )
        stats_map = {feature: {} for feature in features}
        for row in _iter_local_rows(rows):
            profile_date = str(row["_model_lens_profile_date"])
            row_count = int(row["_model_lens_row_count"] or 0)
            for feature in features:
                aliases = alias_map[feature]
                stats_map[feature][profile_date] = {
                    "row_count": row_count,
                    "non_null_count": int(row[aliases["non_null_count"]] or 0),
                    "null_pct": round(float(row[aliases["null_pct"]] or 0.0), 2),
                }
        return stats_map

    def _build_daily_numeric_feature_rows(
        self,
        *,
        config: MonitorConfig,
        source_df: DataFrame,
        feature: str,
        computed_at: str,
        bin_specs: dict[str, tuple[float, ...]],
        stats_by_date: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        numeric = source_df.withColumn("_model_lens_numeric_value", _spark_col(feature).cast("double"))
        edges = tuple(float(value) for value in bin_specs.get(feature, ()))
        count_map: dict[str, list[float]] = {}
        if len(edges) >= 2:
            bucketizer = Bucketizer(
                splits=list(edges),
                inputCol="_model_lens_numeric_value",
                outputCol="_model_lens_bin_index",
                handleInvalid="skip",
            )
            count_rows = (
                bucketizer.transform(numeric.filter(F.col("_model_lens_numeric_value").isNotNull()))
                .groupBy("_model_lens_profile_date", "_model_lens_bin_index")
                .agg(F.count("*").alias("bin_count"))
                .orderBy("_model_lens_profile_date", "_model_lens_bin_index")
            )
            for row in _iter_local_rows(count_rows):
                key = str(row["_model_lens_profile_date"])
                counts = count_map.setdefault(key, [0.0] * (len(edges) - 1))
                bin_index = int(row["_model_lens_bin_index"] or 0)
                if 0 <= bin_index < len(counts):
                    counts[bin_index] = float(row["bin_count"] or 0.0)
        payload: list[dict[str, Any]] = []
        feature_stats = stats_by_date or {}
        if not feature_stats:
            stats_rows = (
                numeric.groupBy("_model_lens_profile_date")
                .agg(
                    F.count("*").alias("row_count"),
                    F.sum(F.when(F.col("_model_lens_numeric_value").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("non_null_count"),
                    F.round(
                        F.avg(F.when(F.col("_model_lens_numeric_value").isNull(), F.lit(100.0)).otherwise(F.lit(0.0))),
                        2,
                    ).alias("null_pct"),
                    F.avg("_model_lens_numeric_value").alias("mean"),
                    F.stddev_samp("_model_lens_numeric_value").alias("std"),
                    F.min("_model_lens_numeric_value").alias("min_value"),
                    F.max("_model_lens_numeric_value").alias("max_value"),
                )
                .orderBy("_model_lens_profile_date")
            )
            feature_stats = {
                str(row["_model_lens_profile_date"]): {
                    "row_count": int(row["row_count"] or 0),
                    "non_null_count": int(row["non_null_count"] or 0),
                    "null_pct": round(float(row["null_pct"] or 0.0), 2),
                    "mean": _safe_float(row["mean"]),
                    "std": _safe_float(row["std"]),
                    "min_value": _safe_float(row["min_value"]),
                    "max_value": _safe_float(row["max_value"]),
                }
                for row in _iter_local_rows(stats_rows)
            }
        for profile_date in sorted(feature_stats):
            row = feature_stats[profile_date]
            distribution_payload = {
                "edges": [round(float(value), 6) for value in edges],
                "counts": [round(float(value), 6) for value in count_map.get(profile_date, [0.0] * max(len(edges) - 1, 0))],
                "sample_values": [],
            }
            payload.append({
                "model_key": config.model_key,
                "profile_date": profile_date,
                "feature_name": feature,
                "feature_kind": "numeric",
                "row_count": int(row["row_count"] or 0),
                "non_null_count": int(row["non_null_count"] or 0),
                "null_pct": round(float(row["null_pct"] or 0.0), 2),
                "mean": _safe_float(row["mean"]),
                "std": _safe_float(row["std"]),
                "min_value": _safe_float(row["min_value"]),
                "max_value": _safe_float(row["max_value"]),
                "distribution_json": json.dumps(distribution_payload),
                "computed_at": computed_at,
            })
        return payload

    def _build_daily_categorical_feature_rows(
        self,
        *,
        config: MonitorConfig,
        source_df: DataFrame,
        feature: str,
        computed_at: str,
        stats_by_date: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        feature_value = F.coalesce(_spark_col(feature).cast("string"), F.lit("__NULL__")).alias("_model_lens_feature_value")
        counts_df = (
            source_df
            .select("_model_lens_profile_date", feature_value)
            .groupBy("_model_lens_profile_date", "_model_lens_feature_value")
            .agg(F.count("*").alias("category_count"))
        )
        rank_window = Window.partitionBy("_model_lens_profile_date").orderBy(
            F.col("category_count").desc(),
            F.col("_model_lens_feature_value").asc(),
        )
        top_counts_rows = (
            counts_df
            .withColumn("_model_lens_category_rank", F.row_number().over(rank_window))
            .withColumn(
                "_model_lens_bucket",
                F.when(F.col("_model_lens_category_rank") <= F.lit(CATEGORICAL_TOP_N), F.col("_model_lens_feature_value"))
                .otherwise(F.lit("__OTHER__")),
            )
            .groupBy("_model_lens_profile_date", "_model_lens_bucket")
            .agg(F.sum("category_count").alias("bucket_count"))
            .orderBy("_model_lens_profile_date", "_model_lens_bucket")
        )
        distribution_map: dict[str, dict[str, int]] = {}
        for row in _iter_local_rows(top_counts_rows):
            profile_date = str(row["_model_lens_profile_date"])
            distribution_map.setdefault(profile_date, {})[str(row["_model_lens_bucket"])] = int(row["bucket_count"] or 0)
        payload: list[dict[str, Any]] = []
        feature_stats = stats_by_date or {}
        if not feature_stats:
            stats_rows = (
                source_df.groupBy("_model_lens_profile_date")
                .agg(
                    F.count("*").alias("row_count"),
                    F.sum(F.when(_spark_col(feature).isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("non_null_count"),
                    F.round(
                        F.avg(F.when(_spark_col(feature).isNull(), F.lit(100.0)).otherwise(F.lit(0.0))),
                        2,
                    ).alias("null_pct"),
                )
                .orderBy("_model_lens_profile_date")
            )
            feature_stats = {
                str(row["_model_lens_profile_date"]): {
                    "row_count": int(row["row_count"] or 0),
                    "non_null_count": int(row["non_null_count"] or 0),
                    "null_pct": round(float(row["null_pct"] or 0.0), 2),
                }
                for row in _iter_local_rows(stats_rows)
            }
        for profile_date in sorted(feature_stats):
            row = feature_stats[profile_date]
            payload.append({
                "model_key": config.model_key,
                "profile_date": profile_date,
                "feature_name": feature,
                "feature_kind": "categorical",
                "row_count": int(row["row_count"] or 0),
                "non_null_count": int(row["non_null_count"] or 0),
                "null_pct": round(float(row["null_pct"] or 0.0), 2),
                "mean": None,
                "std": None,
                "min_value": None,
                "max_value": None,
                "distribution_json": json.dumps(distribution_map.get(profile_date, {})),
                "computed_at": computed_at,
            })
        return payload

    def _build_performance_bin_specs(
        self,
        *,
        config: MonitorConfig,
        source_df: DataFrame,
        existing_specs: dict[str, tuple[float, ...]] | None,
    ) -> dict[str, tuple[float, ...]]:
        categorical_set = set(config.contract.categorical_columns)
        specs = {
            str(feature_name): tuple(float(value) for value in edges)
            for feature_name, edges in (existing_specs or {}).items()
            if feature_name and len(edges) >= 2
        }
        candidate_features = [
            feature
            for feature in config.contract.feature_columns
            if feature in source_df.columns and feature not in specs and feature not in categorical_set
        ]
        if not candidate_features:
            return specs
        agg_exprs: list[Any] = []
        alias_map: dict[str, tuple[str, str, str]] = {}
        for index, feature in enumerate(candidate_features):
            value_col = _spark_col(feature).cast("double")
            count_alias = f"_model_lens_{index}_non_null_count"
            min_alias = f"_model_lens_{index}_min_value"
            max_alias = f"_model_lens_{index}_max_value"
            alias_map[feature] = (count_alias, min_alias, max_alias)
            agg_exprs.extend([
                F.count(value_col).alias(count_alias),
                F.min(value_col).alias(min_alias),
                F.max(value_col).alias(max_alias),
            ])
        row = source_df.agg(*agg_exprs).first()
        for feature in candidate_features:
            count_alias, min_alias, max_alias = alias_map[feature]
            non_null_count = int(row[count_alias] or 0)
            min_value = _safe_float(row[min_alias])
            max_value = _safe_float(row[max_alias])
            if non_null_count < max(2, PERFORMANCE_BIN_COUNT) or min_value is None or max_value is None:
                continue
            if min_value == max_value:
                padding = max(abs(min_value) * 0.01, 0.5)
                min_value -= padding
                max_value += padding
            edges = np.linspace(min_value, max_value, num=PERFORMANCE_BIN_COUNT + 1, dtype=float)
            specs[feature] = tuple(round(float(value), 6) for value in edges.tolist())
        return specs

    def _build_daily_performance_profile_rows(
        self,
        *,
        config: MonitorConfig,
        source_df: DataFrame,
        computed_at: str,
        bin_specs: dict[str, tuple[float, ...]],
    ) -> list[dict[str, Any]]:
        if not config.contract.label_col or config.contract.label_col not in source_df.columns:
            return []
        selected_metric_names = tuple(config.performance_metric_names or default_performance_metric_names(config.problem_type))
        total_rows_df = source_df.groupBy("_model_lens_profile_date").agg(F.count("*").alias("_total_rows"))
        regression_mode = (config.problem_type or "classification").strip().lower() == "regression"
        metric_frames: list[DataFrame] = []
        for feature, edges in sorted(bin_specs.items()):
            if feature not in source_df.columns:
                continue
            metric_frame = (
                self._daily_regression_metrics_by_bin_df(config, source_df, feature, edges, selected_metric_names)
                if regression_mode
                else self._daily_classification_metrics_by_bin_df(config, source_df, feature, edges, selected_metric_names)
            )
            if metric_frame is not None:
                metric_frames.append(metric_frame)
        combined = _union_all(metric_frames)
        if combined is None:
            return []
        final_frame = (
            combined.join(total_rows_df, on="_model_lens_profile_date", how="left")
            .withColumn("_total_rows", F.greatest(F.coalesce(F.col("_total_rows"), F.col("row_count")), F.lit(1)))
            .withColumn("volume_pct", F.round((F.col("row_count") / F.col("_total_rows")) * F.lit(100.0), 2))
            .select(
                "model_key",
                F.date_format(F.col("_model_lens_profile_date"), "yyyy-MM-dd").alias("profile_date"),
                "feature_name",
                "bin_label",
                "metric_name",
                "metric_value",
                "row_count",
                "volume_pct",
                F.lit(computed_at).alias("computed_at"),
            )
            .orderBy("profile_date", "feature_name", "bin_label", "metric_name")
        )
        payload: list[dict[str, Any]] = []
        for row in _iter_local_rows(final_frame):
            payload.append({
                "model_key": str(row["model_key"]),
                "profile_date": str(row["profile_date"]),
                "feature_name": str(row["feature_name"]),
                "bin_label": str(row["bin_label"]),
                "metric_name": str(row["metric_name"]),
                "metric_value": float(row["metric_value"]),
                "row_count": int(row["row_count"] or 0),
                "volume_pct": round(float(row["volume_pct"] or 0.0), 2),
                "computed_at": computed_at,
            })
        return payload

    def _bin_label_expr(self, edges: tuple[float, ...], *, bin_col: str = "_model_lens_bin_index"):
        mapping: list[Any] = []
        for bin_index in range(max(len(edges) - 1, 0)):
            label = f"[{edges[bin_index]:.4g}, {edges[bin_index + 1]:.4g})"
            mapping.extend([F.lit(bin_index), F.lit(label)])
        if not mapping:
            return F.lit("")
        return F.coalesce(F.create_map(*mapping)[F.col(bin_col)], F.lit(""))

    def _daily_classification_metrics_by_bin_df(
        self,
        config: MonitorConfig,
        source_df: DataFrame,
        feature: str,
        edges: tuple[float, ...],
        metric_names: tuple[str, ...],
    ) -> DataFrame | None:
        feature_df = (
            source_df
            .select(
                "_model_lens_profile_date",
                _spark_col(feature).cast("double").alias("_model_lens_feature_value"),
                _spark_col(config.contract.prediction_col).cast("double").alias("_model_lens_prediction"),
                _spark_col(config.contract.label_col).cast("double").alias("_model_lens_label"),
            )
            .filter(
                F.col("_model_lens_feature_value").isNotNull()
                & F.col("_model_lens_prediction").isNotNull()
                & F.col("_model_lens_label").isNotNull()
            )
        )
        if not feature_df.take(1):
            return None
        bucketizer = Bucketizer(
            splits=list(edges),
            inputCol="_model_lens_feature_value",
            outputCol="_model_lens_bin_index",
            handleInvalid="skip",
        )
        bucketed = (
            bucketizer.transform(feature_df)
            .withColumn("_model_lens_pred_binary", F.when(F.col("_model_lens_prediction") >= F.lit(0.5), F.lit(1)).otherwise(F.lit(0)))
            .withColumn("_model_lens_truth_binary", F.col("_model_lens_label").cast("int"))
        )
        metric_df = (
            bucketed.groupBy("_model_lens_profile_date", "_model_lens_bin_index")
            .agg(
                F.count("*").alias("row_count"),
                F.sum(F.when((F.col("_model_lens_pred_binary") == 1) & (F.col("_model_lens_truth_binary") == 1), F.lit(1)).otherwise(F.lit(0))).alias("tp"),
                F.sum(F.when((F.col("_model_lens_pred_binary") == 1) & (F.col("_model_lens_truth_binary") == 0), F.lit(1)).otherwise(F.lit(0))).alias("fp"),
                F.sum(F.when((F.col("_model_lens_pred_binary") == 0) & (F.col("_model_lens_truth_binary") == 1), F.lit(1)).otherwise(F.lit(0))).alias("fn"),
                F.sum(F.when((F.col("_model_lens_pred_binary") == 0) & (F.col("_model_lens_truth_binary") == 0), F.lit(1)).otherwise(F.lit(0))).alias("tn"),
            )
            .withColumn("precision", F.when(F.col("tp") + F.col("fp") > 0, F.col("tp") / (F.col("tp") + F.col("fp"))).otherwise(F.lit(0.0)))
            .withColumn("recall", F.when(F.col("tp") + F.col("fn") > 0, F.col("tp") / (F.col("tp") + F.col("fn"))).otherwise(F.lit(0.0)))
            .withColumn(
                "f1",
                F.when(
                    F.col("precision") + F.col("recall") > 0,
                    (F.lit(2.0) * F.col("precision") * F.col("recall")) / (F.col("precision") + F.col("recall")),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn("accuracy", (F.col("tp") + F.col("tn")) / F.col("row_count"))
            .withColumn("_model_lens_bin_index", F.col("_model_lens_bin_index").cast("int"))
        )
        selected = [metric_name for metric_name in metric_names if metric_name in {"f1", "precision", "recall", "accuracy"}]
        if not selected:
            return None
        stack_expr = ", ".join([f"'{metric_name}', `{metric_name}`" for metric_name in selected])
        return (
            metric_df
            .select(
                F.lit(config.model_key).alias("model_key"),
                "_model_lens_profile_date",
                F.lit(feature).alias("feature_name"),
                self._bin_label_expr(edges).alias("bin_label"),
                "row_count",
                F.expr(f"stack({len(selected)}, {stack_expr}) as (metric_name, metric_value)"),
            )
            .filter(F.col("metric_value").isNotNull())
        )

    def _daily_regression_metrics_by_bin_df(
        self,
        config: MonitorConfig,
        source_df: DataFrame,
        feature: str,
        edges: tuple[float, ...],
        metric_names: tuple[str, ...],
    ) -> DataFrame | None:
        feature_df = (
            source_df
            .select(
                "_model_lens_profile_date",
                _spark_col(feature).cast("double").alias("_model_lens_feature_value"),
                _spark_col(config.contract.prediction_col).cast("double").alias("_model_lens_prediction"),
                _spark_col(config.contract.label_col).cast("double").alias("_model_lens_label"),
            )
            .filter(
                F.col("_model_lens_feature_value").isNotNull()
                & F.col("_model_lens_prediction").isNotNull()
                & F.col("_model_lens_label").isNotNull()
            )
        )
        if not feature_df.take(1):
            return None
        bucketizer = Bucketizer(
            splits=list(edges),
            inputCol="_model_lens_feature_value",
            outputCol="_model_lens_bin_index",
            handleInvalid="skip",
        )
        metric_df = (
            bucketizer.transform(feature_df)
            .withColumn("_model_lens_abs_error", F.abs(F.col("_model_lens_prediction") - F.col("_model_lens_label")))
            .withColumn("_model_lens_squared_error", F.pow(F.col("_model_lens_prediction") - F.col("_model_lens_label"), 2))
            .groupBy("_model_lens_profile_date", "_model_lens_bin_index")
            .agg(
                F.count("*").alias("row_count"),
                F.avg("_model_lens_abs_error").alias("mae"),
                F.sqrt(F.avg("_model_lens_squared_error")).alias("rmse"),
            )
            .withColumn("_model_lens_bin_index", F.col("_model_lens_bin_index").cast("int"))
        )
        selected = [metric_name for metric_name in metric_names if metric_name in {"mae", "rmse"}]
        if not selected:
            return None
        stack_expr = ", ".join([f"'{metric_name}', `{metric_name}`" for metric_name in selected])
        return (
            metric_df
            .select(
                F.lit(config.model_key).alias("model_key"),
                "_model_lens_profile_date",
                F.lit(feature).alias("feature_name"),
                self._bin_label_expr(edges).alias("bin_label"),
                "row_count",
                F.expr(f"stack({len(selected)}, {stack_expr}) as (metric_name, metric_value)"),
            )
            .filter(F.col("metric_value").isNotNull())
        )


def build_refresh_repository(
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
) -> SparkRefreshRepository:
    from model_lens.services.control_plane import _build_read_model

    return SparkRefreshRepository(
        warehouse=WarehouseConnection(warehouse_id=warehouse_id or settings.sql_warehouse_id),
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
