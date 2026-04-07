from __future__ import annotations

import json
import math
from uuid import uuid4

import pytest

from pyspark.sql import functions as F

from model_lens.domain.models import InferenceContract, MonitorConfig, RefreshResult
from model_lens.services.table_names import TableNames


SparkSession = pytest.importorskip("pyspark.sql").SparkSession
SparkRefreshRepository = pytest.importorskip("model_lens.services.spark_refresh").SparkRefreshRepository


class DummyWarehouse:
    def __init__(self, spark: SparkSession) -> None:
        self._spark = spark

    def get_columns(self, table_name: str) -> list[str]:
        return list(self._spark.table(table_name).columns)

    def query(self, sql: str):
        raise AssertionError(f"Unexpected warehouse query: {sql}")


class RecordingSparkPersistenceRepository(SparkRefreshRepository):
    def __init__(self, *, spark: SparkSession, table_names: TableNames, table_frames: dict[str, object] | None = None) -> None:
        super().__init__(
            warehouse=DummyWarehouse(spark),  # type: ignore[arg-type]
            table_names=table_names,
            spark=spark,
        )
        self._table_frames = table_frames or {}
        self.deleted: list[tuple[str, str]] = []
        self.appended: list[tuple[str, list[dict[str, object]]]] = []
        self.rewritten: list[str] = []
        self.synced = False
        self.replaced_bin_specs: list[tuple[str, dict[str, tuple[float, ...]]]] = []

    def _read_table(self, table_name: str):
        frame = self._table_frames.get(table_name)
        if frame is None:
            raise AssertionError(f"Unexpected table read: {table_name}")
        return frame

    def _delete_where(self, table_name: str, predicate_sql: str) -> None:
        self.deleted.append((table_name, predicate_sql))

    def _append_df_to_table(self, table_name: str, frame) -> None:
        self.appended.append((table_name, [row.asDict(recursive=True) for row in frame.collect()]))

    def _sync_read_model(self) -> None:
        self.synced = True


class RecordingSparkAppendRepository(RecordingSparkPersistenceRepository):
    def _rewrite_quality_summary(self, model_key: str) -> None:
        self.rewritten.append(model_key)

    def get_performance_bin_specs(self, model_key: str) -> dict[str, tuple[float, ...]]:
        del model_key
        return {"amount": (0.0, 1.0, 2.0)}

    def replace_performance_bin_specs(self, model_key: str, specs: dict[str, tuple[float, ...]]) -> None:
        self.replaced_bin_specs.append((model_key, specs))


def _spark() -> SparkSession:
    try:
        return SparkSession.builder.master("local[1]").appName("model-lens-spark-tests").getOrCreate()
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"Local Spark is unavailable in this environment: {error}")


def _source_table_name() -> str:
    return f"model_lens_source_{uuid4().hex}"


def _label_table_name() -> str:
    return f"model_lens_labels_{uuid4().hex}"


class PersistedSparkRefreshRepository(SparkRefreshRepository):
    def __init__(
        self,
        *,
        spark: SparkSession,
        persisted_quality_rows: list[dict[str, object]],
        persisted_feature_rows: list[dict[str, object]],
        persisted_performance_rows: list[dict[str, object]],
    ) -> None:
        super().__init__(
            warehouse=DummyWarehouse(spark),  # type: ignore[arg-type]
            table_names=TableNames(catalog="main", schema="default"),
            spark=spark,
        )
        self._persisted_quality_rows = persisted_quality_rows
        self._persisted_feature_rows = persisted_feature_rows
        self._persisted_performance_rows = persisted_performance_rows

    def _load_persisted_daily_quality_profile_df(self, model_key: str, start_date: str, end_date: str):
        del model_key, start_date, end_date
        return self._daily_quality_profile_df_from_rows(self._persisted_quality_rows)

    def _load_persisted_daily_feature_profile_df(self, model_key: str, start_date: str, end_date: str):
        del model_key, start_date, end_date
        return self._daily_feature_profile_df_from_rows(self._persisted_feature_rows)

    def _load_persisted_daily_performance_profile_df(self, model_key: str, start_date: str, end_date: str):
        del model_key, start_date, end_date
        return self._daily_performance_profile_df_from_rows(self._persisted_performance_rows)


def test_build_daily_profiles_uses_spark_for_quality_feature_and_performance_rows() -> None:
    spark = _spark()
    source_table = _source_table_name()
    spark.createDataFrame(
        [
            {"event_ts": "2026-01-01T00:00:00", "prediction": 0.9, "label": 1, "amount": 100.0, "country": "US"},
            {"event_ts": "2026-01-01T01:00:00", "prediction": 0.1, "label": 0, "amount": 110.0, "country": "CA"},
            {"event_ts": "2026-01-01T02:00:00", "prediction": 0.8, "label": 1, "amount": 115.0, "country": "US"},
            {"event_ts": "2026-01-01T03:00:00", "prediction": 0.2, "label": 0, "amount": 118.0, "country": "CA"},
            {"event_ts": "2026-01-01T04:00:00", "prediction": 0.7, "label": 1, "amount": 122.0, "country": "US"},
            {"event_ts": "2026-01-01T05:00:00", "prediction": 0.3, "label": 0, "amount": 128.0, "country": "CA"},
            {"event_ts": "2026-01-02T00:00:00", "prediction": 0.8, "label": 1, "amount": 120.0, "country": "US"},
            {"event_ts": "2026-01-02T01:00:00", "prediction": 0.2, "label": 0, "amount": 130.0, "country": "US"},
            {"event_ts": "2026-01-02T02:00:00", "prediction": 0.75, "label": 1, "amount": 132.0, "country": "US"},
            {"event_ts": "2026-01-02T03:00:00", "prediction": 0.25, "label": 0, "amount": 135.0, "country": "CA"},
            {"event_ts": "2026-01-02T04:00:00", "prediction": 0.85, "label": 1, "amount": 140.0, "country": "US"},
            {"event_ts": "2026-01-02T05:00:00", "prediction": 0.15, "label": 0, "amount": 145.0, "country": "CA"},
        ]
    ).createOrReplaceTempView(source_table)
    repository = SparkRefreshRepository(
        warehouse=DummyWarehouse(spark),  # type: ignore[arg-type]
        table_names=TableNames(catalog="main", schema="default"),
        spark=spark,
    )
    config = MonitorConfig(
        model_key="fraud_v1",
        display_name="Fraud V1",
        source_table=source_table,
        contract=InferenceContract(
            timestamp_col="event_ts",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount", "country"),
            categorical_columns=("country",),
        ),
        problem_type="classification",
    )

    result = repository.build_daily_profiles(
        config,
        start_date="2026-01-01",
        end_date="2026-01-02",
        computed_at="2026-01-03T00:00:00+00:00",
        include_drift_quality=True,
        include_performance=True,
        existing_bin_specs=None,
    )

    assert len(result.daily_quality_profile_rows) == 2
    assert len(result.daily_feature_profile_rows) == 4
    assert "amount" in result.performance_bin_specs
    assert result.daily_performance_profile_rows
    amount_profile = next(
        row
        for row in result.daily_feature_profile_rows
        if row["feature_name"] == "amount" and row["profile_date"] == "2026-01-01"
    )
    distribution_payload = json.loads(str(amount_profile["distribution_json"]))
    assert distribution_payload["edges"]
    assert distribution_payload["counts"]
    assert distribution_payload["sample_values"] == []
    metric_names = {row["metric_name"] for row in result.daily_performance_profile_rows}
    assert {"f1", "precision", "recall"} <= metric_names


def test_append_refresh_result_backfills_missing_model_key_before_persisting() -> None:
    spark = _spark()
    repository = RecordingSparkAppendRepository(
        spark=spark,
        table_names=TableNames(catalog="main", schema="default"),
        table_frames={},
    )

    repository.append_refresh_result(
        "fraud_v1",
        RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            daily_quality_profile_rows=[
                {
                    "profile_date": "2026-01-01",
                    "row_count": 5,
                    "prediction_mean": 0.5,
                    "prediction_std": 0.1,
                    "null_rates": "{}",
                    "label_row_count": 5,
                    "computed_at": "2026-01-02T00:00:00+00:00",
                }
            ],
        ),
        source_run_id="run-1",
    )

    daily_quality_rows = next(
        rows
        for table_name, rows in repository.appended
        if table_name == repository._table_names.daily_quality_profiles
    )
    assert daily_quality_rows[0]["model_key"] == "fraud_v1"


def test_spark_refresh_repository_uses_shared_external_join_key_without_entity_id() -> None:
    spark = _spark()
    source_table = _source_table_name()
    labels_table = _label_table_name()
    spark.createDataFrame(
        [
            {"event_ts": "2026-01-01T00:00:00", "prediction": 0.9, "gc_transaction": "t1", "amount": 100.0},
            {"event_ts": "2026-01-01T01:00:00", "prediction": 0.2, "gc_transaction": "t2", "amount": 120.0},
        ]
    ).createOrReplaceTempView(source_table)
    spark.createDataFrame(
        [
            {"gc_transaction": "t1", "label": 1, "label_ts": "2026-01-02T00:00:00"},
            {"gc_transaction": "t2", "label": 0, "label_ts": "2026-01-02T00:01:00"},
        ]
    ).createOrReplaceTempView(labels_table)
    repository = SparkRefreshRepository(
        warehouse=DummyWarehouse(spark),  # type: ignore[arg-type]
        table_names=TableNames(catalog="main", schema="default"),
        spark=spark,
    )
    config = MonitorConfig(
        model_key="spoof_ios",
        display_name="Spoof iOS",
        source_table=source_table,
        labels_table=labels_table,
        labels_join_col="gc_transaction",
        labels_order_col="label_ts",
        contract=InferenceContract(
            timestamp_col="event_ts",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount", "gc_transaction"),
        ),
        problem_type="classification",
    )

    profile = repository.get_source_profile(config, start_date="2026-01-01", end_date="2026-01-01")
    label_signature = repository.get_label_watermark(config, start_date="2026-01-01", end_date="2026-01-01")

    assert profile["label_row_count"] == 2
    assert label_signature == "2026-01-02T00:01:00"


def test_spark_refresh_repository_derives_window_rows_from_spark_daily_profiles() -> None:
    spark = _spark()
    source_table = _source_table_name()
    spark.createDataFrame(
        [
            {"event_ts": "2026-01-01T00:00:00", "prediction": 0.2, "label": 0, "amount": 100.0, "country": "US"},
            {"event_ts": "2026-01-01T01:00:00", "prediction": 0.3, "label": 0, "amount": 105.0, "country": "CA"},
            {"event_ts": "2026-01-02T00:00:00", "prediction": 0.25, "label": 0, "amount": 110.0, "country": "US"},
            {"event_ts": "2026-01-02T01:00:00", "prediction": 0.35, "label": 0, "amount": 115.0, "country": "CA"},
            {"event_ts": "2026-01-03T00:00:00", "prediction": 0.8, "label": 1, "amount": 180.0, "country": "US"},
            {"event_ts": "2026-01-03T01:00:00", "prediction": 0.7, "label": 1, "amount": 185.0, "country": "US"},
            {"event_ts": "2026-01-04T00:00:00", "prediction": 0.85, "label": 1, "amount": 190.0, "country": "CA"},
            {"event_ts": "2026-01-04T01:00:00", "prediction": 0.75, "label": 1, "amount": 195.0, "country": "US"},
        ]
    ).createOrReplaceTempView(source_table)
    seed_repository = SparkRefreshRepository(
        warehouse=DummyWarehouse(spark),  # type: ignore[arg-type]
        table_names=TableNames(catalog="main", schema="default"),
        spark=spark,
    )
    config = MonitorConfig(
        model_key="fraud_v1",
        display_name="Fraud V1",
        source_table=source_table,
        contract=InferenceContract(
            timestamp_col="event_ts",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount", "country"),
            categorical_columns=("country",),
        ),
        problem_type="classification",
    )
    baseline_profiles = seed_repository.build_daily_profiles(
        config,
        start_date="2026-01-01",
        end_date="2026-01-02",
        computed_at="2026-01-05T00:00:00+00:00",
        include_drift_quality=True,
        include_performance=True,
        existing_bin_specs=None,
    )
    current_profiles = seed_repository.build_daily_profiles(
        config,
        start_date="2026-01-03",
        end_date="2026-01-04",
        computed_at="2026-01-05T00:00:00+00:00",
        include_drift_quality=True,
        include_performance=True,
        existing_bin_specs=baseline_profiles.performance_bin_specs,
    )
    repository = PersistedSparkRefreshRepository(
        spark=spark,
        persisted_quality_rows=baseline_profiles.daily_quality_profile_rows,
        persisted_feature_rows=baseline_profiles.daily_feature_profile_rows,
        persisted_performance_rows=baseline_profiles.daily_performance_profile_rows,
    )

    result = repository.derive_refresh_result_from_daily_profile_rows(
        config=config,
        metadata_list=[{
            "window_id": "fraud_v1:daily:2026-01-03:2026-01-04:2026-01-01:2026-01-02",
            "model_key": "fraud_v1",
            "window_grain": "daily",
            "window_start": "2026-01-03",
            "window_end": "2026-01-04",
            "baseline_start": "2026-01-01",
            "baseline_end": "2026-01-02",
            "baseline_kind": "rolling",
        }],
        current_daily_quality_profile_rows=current_profiles.daily_quality_profile_rows,
        current_daily_feature_profile_rows=current_profiles.daily_feature_profile_rows,
        current_daily_performance_profile_rows=current_profiles.daily_performance_profile_rows,
        derivation_start="2026-01-01",
        derivation_end="2026-01-04",
        computed_at="2026-01-05T00:00:00+00:00",
        prior_open_incidents={},
        include_drift_quality=True,
        include_performance=True,
    )

    assert len(result.window_rows) == 1
    assert result.quality_history_rows[0]["row_count"] == 4
    assert result.drift_rows
    assert {"amount", "country"} <= {row["feature_name"] for row in result.drift_rows}
    numeric_metrics = [
        row["metric_value"]
        for row in result.drift_rows
        if row["feature_name"] == "amount"
    ]
    assert numeric_metrics
    assert all(math.isfinite(float(metric)) for metric in numeric_metrics)
    assert result.performance_rows


def test_spark_refresh_repository_qualifies_model_key_when_joining_quality_profiles_to_window_metadata() -> None:
    spark = _spark()
    repository = PersistedSparkRefreshRepository(
        spark=spark,
        persisted_quality_rows=[],
        persisted_feature_rows=[],
        persisted_performance_rows=[],
    )
    config = MonitorConfig(
        model_key="fraud_v1",
        display_name="Fraud V1",
        source_table="unused_source",
        contract=InferenceContract(
            timestamp_col="event_ts",
            prediction_col="prediction",
            feature_columns=("amount",),
        ),
        problem_type="classification",
    )

    result = repository.derive_refresh_result_from_daily_profile_rows(
        config=config,
        metadata_list=[{
            "window_id": "fraud_v1:daily:2026-01-03:2026-01-03:2026-01-01:2026-01-02",
            "model_key": "fraud_v1",
            "window_grain": "daily",
            "window_start": "2026-01-03",
            "window_end": "2026-01-03",
            "baseline_start": "2026-01-01",
            "baseline_end": "2026-01-02",
            "baseline_kind": "rolling",
        }],
        current_daily_quality_profile_rows=[
            {
                "model_key": "fraud_v1",
                "profile_date": "2026-01-03",
                "row_count": 10,
                "prediction_mean": 0.5,
                "prediction_std": 0.1,
                "null_rates": '{"amount": 0.0}',
                "label_row_count": 0,
                "computed_at": "2026-01-05T00:00:00+00:00",
            }
        ],
        current_daily_feature_profile_rows=[],
        current_daily_performance_profile_rows=[],
        derivation_start="2026-01-03",
        derivation_end="2026-01-03",
        computed_at="2026-01-05T00:00:00+00:00",
        prior_open_incidents={},
        include_drift_quality=False,
        include_performance=False,
    )

    assert result.quality_history_rows == [{
        "model_key": "fraud_v1",
        "window_id": "fraud_v1:daily:2026-01-03:2026-01-03:2026-01-01:2026-01-02",
        "window_start": "2026-01-03",
        "window_end": "2026-01-03",
        "baseline_start": "2026-01-01",
        "baseline_end": "2026-01-02",
        "row_count": 10,
        "prediction_mean": 0.5,
        "prediction_std": 0.1,
        "null_rates": '{"amount": 0.0}',
        "computed_at": "2026-01-05T00:00:00+00:00",
    }]


def test_spark_refresh_repository_derives_recovered_incident_history_without_new_drift_rows() -> None:
    spark = _spark()
    repository = PersistedSparkRefreshRepository(
        spark=spark,
        persisted_quality_rows=[],
        persisted_feature_rows=[],
        persisted_performance_rows=[],
    )
    config = MonitorConfig(
        model_key="fraud_v1",
        display_name="Fraud V1",
        source_table="unused_source",
        contract=InferenceContract(
            timestamp_col="event_ts",
            prediction_col="prediction",
            feature_columns=("amount",),
        ),
        problem_type="classification",
    )

    result = repository.derive_refresh_result_from_daily_profile_rows(
        config=config,
        metadata_list=[{
            "window_id": "fraud_v1:daily:2026-01-03:2026-01-03:2026-01-01:2026-01-02",
            "model_key": "fraud_v1",
            "window_grain": "daily",
            "window_start": "2026-01-03",
            "window_end": "2026-01-03",
            "baseline_start": "2026-01-01",
            "baseline_end": "2026-01-02",
            "baseline_kind": "rolling",
        }],
        current_daily_quality_profile_rows=[
            {
                "model_key": "fraud_v1",
                "profile_date": "2026-01-03",
                "row_count": 10,
                "prediction_mean": 0.5,
                "prediction_std": 0.1,
                "null_rates": '{"amount": 0.0}',
                "label_row_count": 0,
                "computed_at": "2026-01-05T00:00:00+00:00",
            }
        ],
        current_daily_feature_profile_rows=[],
        current_daily_performance_profile_rows=[],
        derivation_start="2026-01-01",
        derivation_end="2026-01-03",
        computed_at="2026-01-05T00:00:00+00:00",
        prior_open_incidents={
            ("fraud_v1", "amount", "psi"): {
                "model_key": "fraud_v1",
                "feature_name": "amount",
                "metric_name": "psi",
                "severity": "warning",
                "status": "open",
                "metric_value": 0.12,
                "window_end": "2026-01-02",
                "observed_at": "2026-01-02T00:00:00+00:00",
            }
        },
        include_drift_quality=True,
        include_performance=False,
    )

    assert result.incident_rows == []
    assert result.incident_history_rows == [{
        "model_key": "fraud_v1",
        "feature_name": "amount",
        "metric_name": "psi",
        "event_type": "recovered",
        "severity": "warning",
        "status": "closed",
        "metric_value": 0.0,
        "window_id": "fraud_v1:daily:2026-01-03:2026-01-03:2026-01-01:2026-01-02",
        "window_start": "2026-01-03",
        "window_end": "2026-01-03",
        "baseline_start": "2026-01-01",
        "baseline_end": "2026-01-02",
        "observed_at": "2026-01-05 00:00:00",
    }]


def test_spark_refresh_repository_reads_and_replaces_performance_bin_specs_via_spark() -> None:
    spark = _spark()
    table_names = TableNames(catalog="main", schema="default")
    repository = RecordingSparkPersistenceRepository(
        spark=spark,
        table_names=table_names,
        table_frames={
            table_names.performance_bin_specs: spark.createDataFrame([
                {
                    "model_key": "payments_risk_v1",
                    "feature_name": "amount",
                    "edges_json": "[0.0, 1.5, 3.0]",
                    "computed_at": "2026-01-20T00:00:00+00:00",
                }
            ]),
        },
    )

    assert repository.get_performance_bin_specs("payments_risk_v1") == {"amount": (0.0, 1.5, 3.0)}

    repository.replace_performance_bin_specs("payments_risk_v1", {"amount": (0.0, 2.0, 4.0)})

    assert repository.deleted == [
        (table_names.performance_bin_specs, "model_key = 'payments_risk_v1'"),
    ]
    appended_rows = repository.appended[-1][1]
    assert json.loads(str(appended_rows[0]["edges_json"])) == [0.0, 2.0, 4.0]


def test_spark_refresh_repository_rewrites_quality_summary_from_spark_daily_profiles() -> None:
    spark = _spark()
    table_names = TableNames(catalog="main", schema="default")
    daily_quality_df = (
        spark.createDataFrame([
            {
                "model_key": "payments_risk_v1",
                "profile_date": "2026-01-19",
                "row_count": 100,
                "prediction_mean": 0.2,
                "prediction_std": 0.1,
                "null_rates": '{"amount": 0.0}',
                "label_row_count": 90,
                "computed_at": "2026-01-19T00:00:00+00:00",
                "source_run_id": "run-1",
            },
            {
                "model_key": "payments_risk_v1",
                "profile_date": "2026-01-20",
                "row_count": 50,
                "prediction_mean": 0.6,
                "prediction_std": 0.2,
                "null_rates": '{"amount": 20.0}',
                "label_row_count": 40,
                "computed_at": "2026-01-20T00:00:00+00:00",
                "source_run_id": "run-2",
            },
        ])
        .withColumn("profile_date", F.to_date("profile_date"))
        .withColumn("computed_at", F.to_timestamp("computed_at"))
    )
    repository = RecordingSparkPersistenceRepository(
        spark=spark,
        table_names=table_names,
        table_frames={table_names.daily_quality_profiles: daily_quality_df},
    )

    repository._rewrite_quality_summary("payments_risk_v1")

    assert repository.deleted == [
        (table_names.quality_metrics, "model_key = 'payments_risk_v1'"),
    ]
    quality_rows = next(rows for table_name, rows in repository.appended if table_name == table_names.quality_metrics)
    summary = quality_rows[0]
    assert summary["total_rows"] == 150
    assert str(summary["min_date"]) == "2026-01-19"
    assert str(summary["max_date"]) == "2026-01-20"
    assert round(float(summary["prediction_mean"]), 4) == 0.3333
    assert json.loads(str(summary["daily_volume"])) == {"2026-01-19": 100, "2026-01-20": 50}
    assert json.loads(str(summary["null_rates"])) == {"amount": 6.67}


def test_spark_refresh_repository_replace_all_refresh_results_uses_spark_table_writes() -> None:
    spark = _spark()
    table_names = TableNames(catalog="main", schema="default")
    repository = RecordingSparkAppendRepository(spark=spark, table_names=table_names)

    repository.replace_all_refresh_results(
        "payments_risk_v1",
        RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[],
            quality_history_rows=[],
            window_rows=[
                {
                    "window_id": "window-1",
                    "model_key": "payments_risk_v1",
                    "window_grain": "daily",
                    "window_start": "2026-01-20",
                    "window_end": "2026-01-20",
                    "baseline_start": "2026-01-13",
                    "baseline_end": "2026-01-19",
                    "baseline_kind": "rolling",
                    "created_at": "2026-01-20T00:00:00+00:00",
                }
            ],
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
            performance_bin_specs={"amount": (0.0, 10.0, 20.0)},
        ),
        source_run_id="run-1",
    )

    deleted_tables = {table_name for table_name, _ in repository.deleted}
    assert table_names.comparison_windows in deleted_tables
    assert table_names.daily_quality_profiles in deleted_tables
    assert table_names.performance_bin_specs in deleted_tables
    appended_tables = {table_name for table_name, _ in repository.appended}
    assert table_names.comparison_windows in appended_tables
    assert table_names.daily_quality_profiles in appended_tables
    assert repository.rewritten == ["payments_risk_v1"]
    assert repository.synced is True
    assert repository.replaced_bin_specs == [("payments_risk_v1", {"amount": (0.0, 10.0, 20.0)})]


def test_spark_refresh_repository_append_refresh_result_clears_recovered_incidents() -> None:
    spark = _spark()
    table_names = TableNames(catalog="main", schema="default")
    repository = RecordingSparkAppendRepository(spark=spark, table_names=table_names)

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
                    "window_start": "2026-01-20",
                    "window_end": "2026-01-20",
                    "baseline_start": "2026-01-13",
                    "baseline_end": "2026-01-19",
                    "observed_at": "2026-01-20T00:00:00+00:00",
                }
            ],
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
            performance_bin_specs={"amount": (0.0, 2.0, 4.0)},
        ),
        source_run_id="run-2",
    )

    assert (table_names.incidents, "model_key = 'payments_risk_v1'") in repository.deleted
    assert any(table_name == table_names.daily_quality_profiles for table_name, _ in repository.appended)
    assert repository.rewritten == ["payments_risk_v1"]
    assert repository.synced is True
    assert repository.replaced_bin_specs == [("payments_risk_v1", {"amount": (0.0, 2.0, 4.0)})]
