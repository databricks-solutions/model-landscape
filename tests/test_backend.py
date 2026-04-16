from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import model_lens.backend as backend_module
from model_lens.backend import DashboardBackend, _null_rate_dict, _safe_json_list
from model_lens.domain.models import BaselinePolicy, InferenceContract, MonitorConfig
from model_lens.services.refresh_jobs import SharedWorkflowScheduleStatus
from model_lens.services.thresholds import get_thresholds


class _FakeWarehouse:
    def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
        model_key = params[0]
        metric_name = params[1] if len(params) > 1 else "f1"
        if "FROM drift_metrics" in sql:
            return pd.DataFrame(
                [
                    {
                        "model_key": model_key,
                        "feature_name": "amount",
                        "metric_name": "psi",
                        "metric_value": 0.12,
                        "window_start": "2026-01-13",
                        "window_end": "2026-01-20",
                        "baseline_start": "2026-01-06",
                        "baseline_end": "2026-01-12",
                        "ref_mean": 8.0,
                        "cur_mean": 18.0,
                        "ref_std": 0.8,
                        "cur_std": 1.8,
                        "ref_null_pct": 0.0,
                        "cur_null_pct": 0.4,
                        "ref_count": 90,
                        "cur_count": 110,
                        "computed_at": "2026-01-20T10:00:00",
                    },
                    {
                        "model_key": model_key,
                        "feature_name": "amount",
                        "metric_name": "js_divergence",
                        "metric_value": 0.07,
                        "window_start": "2026-01-13",
                        "window_end": "2026-01-20",
                        "baseline_start": "2026-01-06",
                        "baseline_end": "2026-01-12",
                        "ref_mean": 8.0,
                        "cur_mean": 18.0,
                        "ref_std": 0.8,
                        "cur_std": 1.8,
                        "ref_null_pct": 0.0,
                        "cur_null_pct": 0.4,
                        "ref_count": 90,
                        "cur_count": 110,
                        "computed_at": "2026-01-20T10:00:00",
                    },
                    {
                        "model_key": model_key,
                        "feature_name": "amount",
                        "metric_name": "psi",
                        "metric_value": 0.21,
                        "window_start": "2026-01-14",
                        "window_end": "2026-01-21",
                        "baseline_start": "2026-01-07",
                        "baseline_end": "2026-01-13",
                        "ref_mean": 10.0,
                        "cur_mean": 20.0,
                        "ref_std": 1.0,
                        "cur_std": 2.0,
                        "ref_null_pct": 0.0,
                        "cur_null_pct": 0.5,
                        "ref_count": 100,
                        "cur_count": 120,
                        "computed_at": "2026-01-21T10:00:00",
                    },
                    {
                        "model_key": model_key,
                        "feature_name": "amount",
                        "metric_name": "js_divergence",
                        "metric_value": 0.11,
                        "window_start": "2026-01-14",
                        "window_end": "2026-01-21",
                        "baseline_start": "2026-01-07",
                        "baseline_end": "2026-01-13",
                        "ref_mean": 10.0,
                        "cur_mean": 20.0,
                        "ref_std": 1.0,
                        "cur_std": 2.0,
                        "ref_null_pct": 0.0,
                        "cur_null_pct": 0.5,
                        "ref_count": 100,
                        "cur_count": 120,
                        "computed_at": "2026-01-21T10:00:00",
                    },
                ]
            )
        if "FROM quality_metrics" in sql:
            return pd.DataFrame(
                [
                    {
                        "model_key": model_key,
                        "total_rows": 840,
                        "min_date": "2026-01-01",
                        "max_date": "2026-01-21",
                        "prediction_mean": 0.44,
                        "prediction_std": 0.13,
                        "daily_volume": '{"2026-01-21": 40}',
                        "null_rates": '{"amount": 0.0, "velocity_7d": 1.2}',
                        "computed_at": "2026-01-21T10:00:00",
                    }
                ]
            )
        if "FROM quality_history" in sql:
            return pd.DataFrame(
                [
                    {
                        "model_key": model_key,
                        "window_id": "rolling|2026-01-06|2026-01-12|2026-01-13|2026-01-20",
                        "window_start": "2026-01-13",
                        "window_end": "2026-01-20",
                        "baseline_start": "2026-01-06",
                        "baseline_end": "2026-01-12",
                        "row_count": 110,
                        "prediction_mean": 0.42,
                        "prediction_std": 0.12,
                        "null_rates": '{"amount": 0.0, "velocity_7d": 0.8}',
                        "computed_at": "2026-01-20T10:00:00",
                    },
                    {
                        "model_key": model_key,
                        "window_id": "rolling|2026-01-07|2026-01-13|2026-01-14|2026-01-21",
                        "window_start": "2026-01-14",
                        "window_end": "2026-01-21",
                        "baseline_start": "2026-01-07",
                        "baseline_end": "2026-01-13",
                        "row_count": 120,
                        "prediction_mean": 0.44,
                        "prediction_std": 0.13,
                        "null_rates": '{"amount": 0.0, "velocity_7d": 1.2}',
                        "computed_at": "2026-01-21T10:00:00",
                    },
                ]
            )
        if "FROM daily_quality_profiles" in sql:
            return pd.DataFrame(
                [
                    {
                        "model_key": model_key,
                        "profile_date": "2026-01-20",
                        "row_count": 110,
                        "prediction_mean": 0.42,
                        "prediction_std": 0.12,
                        "null_rates": '{"amount": 0.0, "velocity_7d": 0.8}',
                        "computed_at": "2026-01-20T10:00:00",
                    },
                    {
                        "model_key": model_key,
                        "profile_date": "2026-01-21",
                        "row_count": 120,
                        "prediction_mean": 0.44,
                        "prediction_std": 0.13,
                        "null_rates": '{"amount": 0.0, "velocity_7d": 1.2}',
                        "computed_at": "2026-01-21T10:00:00",
                    },
                ]
            )
        if "FROM performance_metrics" in sql:
            return pd.DataFrame(
                [
                    {
                        "model_key": model_key,
                        "feature_name": "amount",
                        "bin_label": "[0, 100)",
                        "baseline_metric": 0.84,
                        "current_metric": 0.84,
                        "delta": 0.0,
                        "volume_pct": 55.0,
                        "contribution": 0.0,
                        "metric_name": metric_name,
                        "window_start": "2026-01-13",
                        "window_end": "2026-01-20",
                        "computed_at": "2026-01-20T10:00:00",
                    },
                    {
                        "model_key": model_key,
                        "feature_name": "velocity_7d",
                        "bin_label": "[0, 5)",
                        "baseline_metric": 0.78,
                        "current_metric": 0.78,
                        "delta": 0.0,
                        "volume_pct": 45.0,
                        "contribution": 0.0,
                        "metric_name": metric_name,
                        "window_start": "2026-01-13",
                        "window_end": "2026-01-20",
                        "computed_at": "2026-01-20T10:00:00",
                    },
                    {
                        "model_key": model_key,
                        "feature_name": "amount",
                        "bin_label": "[0, 100)",
                        "baseline_metric": 0.84,
                        "current_metric": 0.84,
                        "delta": 0.0,
                        "volume_pct": 55.0,
                        "contribution": 0.0,
                        "metric_name": metric_name,
                        "window_start": "2026-01-14",
                        "window_end": "2026-01-21",
                        "computed_at": "2026-01-21T10:00:00",
                    },
                    {
                        "model_key": model_key,
                        "feature_name": "velocity_7d",
                        "bin_label": "[0, 5)",
                        "baseline_metric": 0.78,
                        "current_metric": 0.78,
                        "delta": 0.0,
                        "volume_pct": 45.0,
                        "contribution": 0.0,
                        "metric_name": metric_name,
                        "window_start": "2026-01-14",
                        "window_end": "2026-01-21",
                        "computed_at": "2026-01-21T10:00:00",
                    },
                ]
            )
        raise AssertionError(f"Unexpected query: {sql}")


def _with_published_generation(repository, generation_id: str = "published-1"):
    if not callable(getattr(repository, "get_latest_published_generation_id", None)):
        repository.get_latest_published_generation_id = lambda model_key: generation_id
    return repository


def _make_backend() -> DashboardBackend:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount", "velocity_7d"),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
        labels_table="main.model_lens_demo.labels",
        labels_join_col="entity_id",
        labels_order_col="label_timestamp",
    )
    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(
            [
                {
                    "model_key": "fraud_model_demo",
                    "display_name": "Fraud Model Demo",
                    "max_psi": 0.21,
                    "feature_count": 2,
                    "latest_window_end": "2026-01-21",
                    "total_rows": 840,
                    "latest_data_date": "2026-01-21",
                    "last_refresh_at": "2026-01-21T10:00:00",
                    "open_incident_count": 1,
                }
            ]
        ),
        get_open_incidents=lambda: pd.DataFrame(),
        load_monitor_frame=lambda config: pd.DataFrame(),
    )
    return DashboardBackend(repository=_with_published_generation(repository))


def test_list_models_merges_monitor_configs_with_summary() -> None:
    backend = _make_backend()

    models = backend.list_models()

    assert models == [
        {
            "id": "fraud_model_demo",
            "name": "Fraud Model Demo",
            "description": "main.model_lens_demo.inference_logs | model_id=fraud_model_v1",
            "versions": [],
            "feature_count": 2,
            "slice_columns": ["region"],
            "has_labels": True,
            "baseline_days": 7,
            "baseline_kind": "rolling",
            "baseline_label": "Rolling: 7 days",
            "max_psi": 0.21,
            "total_rows": 840,
            "open_incident_count": 1,
            "freshness_status": "pending_bootstrap",
            "last_run_status": "",
        }
    ]


def test_get_drift_results_pivots_long_metrics_into_feature_period_rows() -> None:
    backend = _make_backend()

    drift = backend.get_drift_results("fraud_model_demo", granularity="daily")

    assert list(drift["feature"]) == ["amount", "amount"]
    assert list(drift["period"]) == ["2026-01-20", "2026-01-21"]
    assert drift.iloc[1]["psi"] == 0.21
    assert drift.iloc[1]["js_divergence"] == 0.11
    assert drift.iloc[1]["kl_divergence"] == 0.0


def test_get_drift_results_requires_a_published_generation() -> None:
    backend = _make_backend()
    backend.repository.get_latest_published_generation_id = lambda model_id: None

    drift = backend.get_drift_results("fraud_model_demo", granularity="daily")

    assert drift.empty


def test_get_drift_results_aggregates_weekly_history_with_latest_overlay_and_summed_counts() -> None:
    backend = _make_backend()

    drift = backend.get_drift_results("fraud_model_demo", granularity="weekly")

    assert list(drift["feature"]) == ["amount"]
    assert drift.iloc[0]["psi"] == 0.21
    assert drift.iloc[0]["js_divergence"] == 0.11
    assert drift.iloc[0]["window_end"] == "2026-01-21"
    assert drift.iloc[0]["ref_mean"] == 10.0
    assert drift.iloc[0]["ref_count"] == 190
    assert drift.iloc[0]["cur_count"] == 230


def test_get_drift_results_limits_to_recent_windows_in_sql() -> None:
    class _RecordingWarehouse(_FakeWarehouse):
        def __init__(self) -> None:
            self.query_param_calls: list[tuple[str, tuple]] = []

        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            self.query_param_calls.append((sql, params))
            return super().query_params(sql, params)

    backend = _make_backend()
    warehouse = _RecordingWarehouse()
    backend.repository._warehouse = warehouse

    backend.get_drift_results("fraud_model_demo")

    sql, _ = warehouse.query_param_calls[-1]
    assert "recent_windows" in sql
    assert "LIMIT 400" in sql


def test_get_quality_stats_parses_json_payloads() -> None:
    backend = _make_backend()

    quality = backend.get_quality_stats("fraud_model_demo")

    assert quality["total_rows"] == 840
    assert quality["daily_volume"] == {"2026-01-21": 40}
    assert quality["null_rates"] == {"amount": 0.0, "velocity_7d": 1.2}


def test_get_quality_history_returns_windowed_rows_with_null_rate_metadata() -> None:
    backend = _make_backend()

    history = backend.get_quality_history("fraud_model_demo")

    assert list(history["period"]) == ["2026-01-20", "2026-01-21"]
    assert list(history["row_count"]) == [110, 120]
    assert history.iloc[1]["null_rates_dict"] == {"amount": 0.0, "velocity_7d": 1.2}
    assert history.iloc[1]["max_null_rate"] == 1.2


def test_get_quality_history_limits_to_recent_windows_in_sql() -> None:
    class _RecordingWarehouse(_FakeWarehouse):
        def __init__(self) -> None:
            self.query_param_calls: list[tuple[str, tuple]] = []

        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            self.query_param_calls.append((sql, params))
            return super().query_params(sql, params)

    backend = _make_backend()
    warehouse = _RecordingWarehouse()
    backend.repository._warehouse = warehouse

    backend.get_quality_history("fraud_model_demo")

    sql, _ = warehouse.query_param_calls[-1]
    assert "recent_history" in sql
    assert "LIMIT 400" in sql


def test_get_quality_history_falls_back_to_daily_profiles_when_window_history_is_missing() -> None:
    class MissingWindowHistoryWarehouse(_FakeWarehouse):
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM quality_history" in sql:
                return pd.DataFrame()
            return super().query_params(sql, params)

    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount", "velocity_7d"),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )
    repository = SimpleNamespace(
        _warehouse=MissingWindowHistoryWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            daily_quality_profiles="daily_quality_profiles",
            performance_metrics="performance_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    history = backend.get_quality_history("fraud_model_demo")

    assert list(history["period"]) == ["2026-01-20", "2026-01-21"]
    assert list(history["row_count"]) == [110, 120]
    assert history.iloc[1]["null_rates_dict"] == {"amount": 0.0, "velocity_7d": 1.2}


def test_get_null_rate_history_explodes_top_features_over_time() -> None:
    backend = _make_backend()

    history = backend.get_null_rate_history("fraud_model_demo")

    assert set(history["feature"]) == {"amount", "velocity_7d"}
    assert list(history["period"].unique()) == ["2026-01-20", "2026-01-21"]
    assert history[history["feature"] == "velocity_7d"]["null_rate"].tolist() == [0.8, 1.2]


def test_get_performance_summary_keeps_zero_delta_rows_visible() -> None:
    backend = _make_backend()

    performance = backend.get_performance_summary("fraud_model_demo", metric_name="rmse")

    assert performance["timeline"] == [
        {"period": "2026-01-20", "rmse": 0.813},
        {"period": "2026-01-21", "rmse": 0.813},
    ]
    assert performance["timeline_unavailable_reason"] == ""
    assert len(performance["latest_bins"]) == 2
    assert len(performance["all_bins"]) == 4
    assert set(performance["contributors"]["feature"]) == {"amount", "velocity_7d"}
    assert performance["has_significant_degradation"] is False
    assert performance["worst_weighted_delta"] == 0.0


def test_get_performance_rows_limits_to_recent_windows_in_sql() -> None:
    class _RecordingWarehouse(_FakeWarehouse):
        def __init__(self) -> None:
            self.query_param_calls: list[tuple[str, tuple]] = []

        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            self.query_param_calls.append((sql, params))
            return super().query_params(sql, params)

    backend = _make_backend()
    warehouse = _RecordingWarehouse()
    backend.repository._warehouse = warehouse

    backend.get_performance_rows("fraud_model_demo", metric_name="f1")

    sql, _ = warehouse.query_param_calls[-1]
    assert "recent_windows" in sql
    assert "LIMIT 180" in sql


def test_get_performance_summary_supports_alternate_metric_names() -> None:
    backend = _make_backend()

    rmse = backend.get_performance_summary("fraud_model_demo", metric_name="rmse")
    precision = backend.get_performance_summary("fraud_model_demo", metric_name="precision")

    assert rmse["timeline"][0]["rmse"] == 0.813
    assert precision["timeline"] == [
        {"period": "2026-01-20", "precision": 0.813},
        {"period": "2026-01-21", "precision": 0.813},
    ]
    assert precision["timeline_unavailable_reason"] == ""


def test_get_performance_summary_prefers_daily_label_metrics_and_keeps_null_gaps() -> None:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )
    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        table_names=SimpleNamespace(
            performance_metrics="performance_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        get_daily_label_metric_rows=lambda model_id, start_date=None, end_date=None: [
            {
                "model_key": model_id,
                "profile_date": "2026-01-20",
                "actual_positive_count": 4,
                "actual_negative_count": 6,
                "predicted_positive_count": 0,
                "predicted_negative_count": 10,
                "tp": 0,
                "fp": 0,
                "fn": 4,
                "tn": 6,
                "precision": None,
                "recall": None,
                "f1": None,
                "accuracy": 0.6,
            },
            {
                "model_key": model_id,
                "profile_date": "2026-01-21",
                "actual_positive_count": 5,
                "actual_negative_count": 5,
                "predicted_positive_count": 4,
                "predicted_negative_count": 6,
                "tp": 3,
                "fp": 1,
                "fn": 2,
                "tn": 4,
                "precision": 0.75,
                "recall": 0.6,
                "f1": 0.6667,
                "accuracy": 0.7,
            },
        ],
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    performance = backend.get_performance_summary("fraud_model_demo", metric_name="precision")

    assert performance["timeline"] == [
        {"period": "2026-01-20", "precision": None},
        {"period": "2026-01-21", "precision": 0.75},
    ]


def test_get_performance_summary_falls_back_to_comparison_window_rows_when_daily_label_metrics_are_empty() -> None:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )
    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        table_names=SimpleNamespace(
            performance_metrics="performance_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        get_daily_label_metric_rows=lambda model_id, start_date=None, end_date=None: [],
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    performance = backend.get_performance_summary("fraud_model_demo", metric_name="precision")

    assert performance["timeline"] == [
        {"period": "2026-01-20", "precision": 0.813},
        {"period": "2026-01-21", "precision": 0.813},
    ]
    assert performance["timeline_unavailable_reason"] == ""


def test_get_latest_window_metrics_aggregates_latest_daily_label_facts() -> None:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )

    class _SnapshotWarehouse(_FakeWarehouse):
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM comparison_windows" in sql:
                return pd.DataFrame(
                    [
                        {
                            "baseline_start": "2026-01-07",
                            "baseline_end": "2026-01-13",
                            "window_start": "2026-01-14",
                            "window_end": "2026-01-21",
                        }
                    ]
                )
            return super().query_params(sql, params)

    repository = SimpleNamespace(
        _warehouse=_SnapshotWarehouse(),
        table_names=SimpleNamespace(comparison_windows="comparison_windows"),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        get_daily_label_metric_rows=lambda model_id, start_date=None, end_date=None: [
            {
                "model_key": model_id,
                "profile_date": "2026-01-20",
                "tp": 3,
                "fp": 1,
                "fn": 2,
                "tn": 4,
                "precision": 0.75,
                "recall": 0.6,
                "f1": 0.6667,
                "accuracy": 0.7,
            },
            {
                "model_key": model_id,
                "profile_date": "2026-01-21",
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 8,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.6667,
                "accuracy": 0.9,
            },
        ],
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    snapshot = backend.get_latest_window_metrics("fraud_model_demo")

    assert snapshot["supported"] is True
    assert snapshot["window_start"] == "2026-01-14"
    assert snapshot["window_end"] == "2026-01-21"
    assert snapshot["metrics"] == {
        "precision": 0.8,
        "recall": 0.5714,
        "f1": 0.6667,
        "accuracy": 0.8,
    }


def test_get_latest_window_metrics_falls_back_to_comparison_window_rows() -> None:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )

    class _SnapshotWarehouse(_FakeWarehouse):
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM comparison_windows" in sql:
                return pd.DataFrame(
                    [
                        {
                            "baseline_start": "2026-01-07",
                            "baseline_end": "2026-01-13",
                            "window_start": "2026-01-14",
                            "window_end": "2026-01-21",
                        }
                    ]
                )
            return super().query_params(sql, params)

    repository = SimpleNamespace(
        _warehouse=_SnapshotWarehouse(),
        table_names=SimpleNamespace(comparison_windows="comparison_windows", performance_metrics="performance_metrics"),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        get_daily_label_metric_rows=lambda model_id, start_date=None, end_date=None: [],
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    snapshot = backend.get_latest_window_metrics("fraud_model_demo")

    assert snapshot["metrics"] == {
        "precision": 0.813,
        "recall": 0.813,
        "f1": 0.813,
        "accuracy": 0.813,
    }
    assert snapshot["message"] == "Showing recent performance trends."


def test_get_quality_stats_and_history_support_class_filters_from_daily_profiles() -> None:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )
    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        table_names=SimpleNamespace(),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        get_daily_class_quality_profile_rows=lambda model_id, start_date=None, end_date=None, class_basis=None, class_value=None: [
            {
                "model_key": model_id,
                "profile_date": "2026-01-20",
                "class_basis": class_basis,
                "class_value": class_value,
                "row_count": 40,
                "prediction_mean": 0.6,
                "prediction_std": 0.1,
                "null_rates": '{"amount": 0.0}',
                "label_row_count": 40,
                "computed_at": "2026-01-20T10:00:00",
            },
            {
                "model_key": model_id,
                "profile_date": "2026-01-21",
                "class_basis": class_basis,
                "class_value": class_value,
                "row_count": 60,
                "prediction_mean": 0.8,
                "prediction_std": 0.2,
                "null_rates": '{"amount": 5.0}',
                "label_row_count": 60,
                "computed_at": "2026-01-21T10:00:00",
            },
        ],
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    quality = backend.get_quality_stats(
        "fraud_model_demo",
        start_date="2026-01-20",
        end_date="2026-01-21",
        class_basis="actual",
        class_value="positive",
    )
    history = backend.get_quality_history(
        "fraud_model_demo",
        start_date="2026-01-20",
        end_date="2026-01-21",
        class_basis="actual",
        class_value="positive",
    )

    assert quality["total_rows"] == 100
    assert quality["daily_volume"] == {"2026-01-20": 40, "2026-01-21": 60}
    assert quality["null_rates"] == {"amount": 3.0}
    assert list(history["period"]) == ["2026-01-20", "2026-01-21"]
    assert list(history["row_count"]) == [40, 60]


def test_get_drift_results_supports_class_filtered_daily_feature_profiles() -> None:
    class _WindowWarehouse(_FakeWarehouse):
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM comparison_windows" in sql:
                return pd.DataFrame(
                    [
                        {
                            "window_id": "rolling|2026-01-01|2026-01-02|2026-01-03|2026-01-04",
                            "model_key": params[0],
                            "window_grain": "day",
                            "window_start": "2026-01-03",
                            "window_end": "2026-01-04",
                            "baseline_start": "2026-01-01",
                            "baseline_end": "2026-01-02",
                            "baseline_kind": "rolling",
                        }
                    ]
                )
            return super().query_params(sql, params)

    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
        ),
        baseline=BaselinePolicy(n_days=2),
        model_id_value="fraud_model_v1",
    )
    repository = SimpleNamespace(
        _warehouse=_WindowWarehouse(),
        table_names=SimpleNamespace(
            comparison_windows="comparison_windows",
            drift_metrics="drift_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        get_daily_class_feature_profile_rows=lambda model_id, start_date=None, end_date=None, class_basis=None, class_value=None: [
            {
                "model_key": model_id,
                "profile_date": "2026-01-01",
                "class_basis": class_basis,
                "class_value": class_value,
                "feature_name": "amount",
                "feature_kind": "numeric",
                "row_count": 3,
                "non_null_count": 3,
                "null_pct": 0.0,
                "mean": 1.0,
                "std": 0.1,
                "distribution_json": '{"sample_values":[0.9,1.0,1.1]}',
            },
            {
                "model_key": model_id,
                "profile_date": "2026-01-02",
                "class_basis": class_basis,
                "class_value": class_value,
                "feature_name": "amount",
                "feature_kind": "numeric",
                "row_count": 3,
                "non_null_count": 3,
                "null_pct": 0.0,
                "mean": 1.1,
                "std": 0.1,
                "distribution_json": '{"sample_values":[1.0,1.1,1.2]}',
            },
            {
                "model_key": model_id,
                "profile_date": "2026-01-03",
                "class_basis": class_basis,
                "class_value": class_value,
                "feature_name": "amount",
                "feature_kind": "numeric",
                "row_count": 3,
                "non_null_count": 3,
                "null_pct": 0.0,
                "mean": 5.0,
                "std": 0.1,
                "distribution_json": '{"sample_values":[4.9,5.0,5.1]}',
            },
            {
                "model_key": model_id,
                "profile_date": "2026-01-04",
                "class_basis": class_basis,
                "class_value": class_value,
                "feature_name": "amount",
                "feature_kind": "numeric",
                "row_count": 3,
                "non_null_count": 3,
                "null_pct": 0.0,
                "mean": 5.1,
                "std": 0.1,
                "distribution_json": '{"sample_values":[5.0,5.1,5.2]}',
            },
        ],
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    drift = backend.get_drift_results(
        "fraud_model_demo",
        granularity="daily",
        start_date="2026-01-03",
        end_date="2026-01-04",
        class_basis="actual",
        class_value="positive",
    )

    assert list(drift["feature"]) == ["amount"]
    assert drift.iloc[0]["period"] == "2026-01-04"
    assert drift.iloc[0]["psi"] > 0.0


def test_feature_detail_load_uses_bounded_sampled_frame() -> None:
    calls: list[dict[str, object]] = []
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount", "velocity_7d"),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )

    def load_monitor_frame(config_arg, **kwargs):
        del config_arg
        calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "event_ts": f"2026-01-{day:02d}T00:00:00",
                    "amount": float(day),
                }
                for day in range(7, 22)
            ]
        )

    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        load_monitor_frame=load_monitor_frame,
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    baseline, current = backend.get_feature_distribution("fraud_model_demo", "amount")

    assert not baseline.empty
    assert not current.empty
    assert len(calls) == 1
    assert calls[0]["start_date"] == "2026-01-07"
    assert calls[0]["end_date"] == "2026-01-21"
    assert calls[0]["feature_columns"] == ("amount",)
    assert calls[0]["sample_rows_per_day"] > 0
    assert calls[0]["max_total_rows"] > 0


def test_feature_detail_returns_empty_when_only_unbounded_load_is_supported() -> None:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )

    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        load_monitor_frame=lambda config_arg: pd.DataFrame([{"event_ts": "2026-01-20T00:00:00", "amount": 10.0}]),
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    details = backend.get_feature_distribution_details("fraud_model_demo", "amount")

    assert details["baseline"].empty
    assert details["current"].empty
    assert details["distribution_source"] == "unavailable_unsafe_bounded_read"


def test_feature_detail_skips_raw_fallback_without_hard_row_cap() -> None:
    calls: list[dict[str, object]] = []
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )

    def load_monitor_frame(config_arg, *, start_date=None, end_date=None, feature_columns=None):
        del config_arg
        calls.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "feature_columns": feature_columns,
            }
        )
        return pd.DataFrame(
            [
                {"event_ts": "2026-01-20T00:00:00", "amount": 10.0},
            ]
        )

    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        load_monitor_frame=load_monitor_frame,
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    details = backend.get_feature_distribution_details("fraud_model_demo", "amount")

    assert details["baseline"].empty
    assert details["current"].empty
    assert details["distribution_source"] == "unavailable_unsafe_bounded_read"
    assert calls == []


def test_current_window_detail_reads_use_current_window_bounds_only() -> None:
    calls: list[dict[str, object]] = []
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount", "velocity_7d"),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )

    class CurrentWindowWarehouse:
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM comparison_windows" in sql:
                return pd.DataFrame(
                    [
                        {
                            "baseline_start": "2026-01-07",
                            "baseline_end": "2026-01-13",
                            "window_start": "2026-01-14",
                            "window_end": "2026-01-21",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected query: {sql}")

    def load_monitor_frame(config_arg, **kwargs):
        del config_arg
        calls.append(kwargs)
        return pd.DataFrame(
            [
                {"event_ts": "2026-01-14T00:00:00", "amount": 10.0, "region": "west", "prediction": 0.2},
                {"event_ts": "2026-01-21T00:00:00", "amount": 20.0, "region": "east", "prediction": 0.8},
            ]
        )

    repository = SimpleNamespace(
        _warehouse=CurrentWindowWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
            comparison_windows="comparison_windows",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        load_monitor_frame=load_monitor_frame,
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    breakdown = backend.get_dimension_breakdown("fraud_model_demo", "amount", "region")
    prediction = backend.get_prediction_distribution("fraud_model_demo")

    assert not breakdown.empty
    assert prediction.tolist() == [0.2, 0.8]
    assert len(calls) == 2
    assert calls[0]["start_date"] == "2026-01-14"
    assert calls[0]["end_date"] == "2026-01-21"
    assert calls[0]["feature_columns"] == ("amount", "region")
    assert calls[1]["start_date"] == "2026-01-14"
    assert calls[1]["end_date"] == "2026-01-21"
    assert calls[1]["sample_rows_per_day"] > 0
    assert calls[1]["max_total_rows"] > 0


def test_current_window_returns_empty_when_only_unbounded_load_is_supported() -> None:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )

    class FallbackWindowWarehouse:
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM comparison_windows" in sql:
                return pd.DataFrame(
                    [
                        {
                            "baseline_start": "2026-01-07",
                            "baseline_end": "2026-01-13",
                            "window_start": "2026-01-14",
                            "window_end": "2026-01-21",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected query: {sql}")

    def load_monitor_frame(config_arg):
        del config_arg
        return pd.DataFrame(
            [
                {"event_ts": "2026-01-13T23:59:59", "prediction": 0.1},
                {"event_ts": "2026-01-14T00:00:00", "prediction": 0.2},
                {"event_ts": "2026-01-21T18:45:00", "prediction": 0.8},
                {"event_ts": "2026-01-22T00:00:00", "prediction": 0.9},
            ]
        )

    repository = SimpleNamespace(
        _warehouse=FallbackWindowWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
            comparison_windows="comparison_windows",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        load_monitor_frame=load_monitor_frame,
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    prediction = backend.get_prediction_distribution("fraud_model_demo")

    assert prediction.tolist() == []


def test_feature_distribution_daily_profile_query_is_bounded_to_latest_window_dates() -> None:
    queries: list[tuple[str, tuple]] = []
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )

    class DailyFeatureWarehouse:
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            queries.append((sql, params))
            if "FROM comparison_windows" in sql:
                return pd.DataFrame(
                    [
                        {
                            "baseline_start": "2026-01-07",
                            "baseline_end": "2026-01-13",
                            "window_start": "2026-01-14",
                            "window_end": "2026-01-21",
                        }
                    ]
                )
            if "FROM daily_feature_profiles" in sql:
                return pd.DataFrame(
                    [
                        {
                            "profile_date": "2026-01-10",
                            "distribution_json": '{"sample_values":[1.0,2.0]}',
                        },
                        {
                            "profile_date": "2026-01-20",
                            "distribution_json": '{"sample_values":[3.0,4.0]}',
                        },
                    ]
                )
            raise AssertionError(f"Unexpected query: {sql}")

    repository = SimpleNamespace(
        _warehouse=DailyFeatureWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
            comparison_windows="comparison_windows",
            daily_feature_profiles="daily_feature_profiles",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    baseline, current = backend.get_feature_distribution("fraud_model_demo", "amount")

    assert baseline.tolist() == [1.0, 2.0]
    assert current.tolist() == [3.0, 4.0]
    profile_query = next(sql for sql, _ in queries if "FROM daily_feature_profiles" in sql)
    profile_params = next(params for sql, params in queries if "FROM daily_feature_profiles" in sql)
    assert "profile_date BETWEEN CAST(%s AS DATE) AND CAST(%s AS DATE)" in profile_query
    assert profile_params == ("fraud_model_demo", "amount", "2026-01-07", "2026-01-21", "published-1")


def test_get_overview_rows_uses_bulk_historical_snapshot_queries() -> None:
    queries: list[str] = []

    class OverviewWarehouse:
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            queries.append(sql)
            if "ROW_NUMBER() OVER" in sql and "FROM quality_metrics" in sql:
                return pd.DataFrame(
                    [
                        {
                            "model_key": "fraud_model_demo",
                            "total_rows": 840,
                            "min_date": "2026-01-01",
                            "max_date": "2026-01-21",
                            "prediction_mean": 0.44,
                            "prediction_std": 0.13,
                            "daily_volume": '{"2026-01-21": 40}',
                            "null_rates": '{"amount": 0.0, "velocity_7d": 1.2}',
                            "computed_at": "2026-01-21T10:00:00",
                        },
                        {
                            "model_key": "chargeback_model_demo",
                            "total_rows": 420,
                            "min_date": "2026-01-05",
                            "max_date": "2026-01-21",
                            "prediction_mean": 0.31,
                            "prediction_std": 0.09,
                            "daily_volume": '{"2026-01-21": 22}',
                            "null_rates": '{"amount": 0.7}',
                            "computed_at": "2026-01-21T10:00:00",
                        },
                    ]
                )
            if "feature_metric_history AS" in sql and "FROM drift_metrics" in sql:
                return pd.DataFrame(
                    [
                        {
                            "model_key": "fraud_model_demo",
                            "feature_name": "amount",
                            "metric_name": "psi",
                            "metric_value": 10.97,
                        },
                        {
                            "model_key": "fraud_model_demo",
                            "feature_name": "amount",
                            "metric_name": "js_divergence",
                            "metric_value": 0.61,
                        },
                        {
                            "model_key": "fraud_model_demo",
                            "feature_name": "velocity_7d",
                            "metric_name": "psi",
                            "metric_value": 0.05,
                        },
                        {
                            "model_key": "fraud_model_demo",
                            "feature_name": "velocity_7d",
                            "metric_name": "js_divergence",
                            "metric_value": 0.03,
                        },
                        {
                            "model_key": "chargeback_model_demo",
                            "feature_name": "amount",
                            "metric_name": "psi",
                            "metric_value": 4.93,
                        },
                        {
                            "model_key": "chargeback_model_demo",
                            "feature_name": "amount",
                            "metric_name": "js_divergence",
                            "metric_value": 0.42,
                        },
                    ]
                )
            raise AssertionError(f"Unexpected overview query: {sql}")

    fraud_config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount", "velocity_7d"),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )
    chargeback_config = MonitorConfig(
        model_key="chargeback_model_demo",
        display_name="Chargeback Model Demo",
        source_table="main.model_lens_demo.chargebacks",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col=None,
            prediction_col="prediction",
            label_col=None,
            feature_columns=("amount",),
            slice_columns=(),
            categorical_columns=(),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value=None,
    )

    repository = SimpleNamespace(
        _warehouse=OverviewWarehouse(),
        table_names=SimpleNamespace(
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
            refresh_runs="refresh_runs",
        ),
        list_monitor_configs=lambda status="active": [fraud_config, chargeback_config],
        get_monitor_summary=lambda: pd.DataFrame(
            [
                {
                    "model_key": "fraud_model_demo",
                    "display_name": "Fraud Model Demo",
                    "max_psi": 0.21,
                    "feature_count": 2,
                    "latest_window_end": "2026-01-21",
                    "total_rows": 840,
                    "latest_data_date": "2026-01-21",
                    "last_refresh_at": "2026-01-21T10:00:00",
                    "open_incident_count": 1,
                },
                {
                    "model_key": "chargeback_model_demo",
                    "display_name": "Chargeback Model Demo",
                    "max_psi": 0.08,
                    "feature_count": 1,
                    "latest_window_end": "2026-01-21",
                    "total_rows": 420,
                    "latest_data_date": "2026-01-21",
                    "last_refresh_at": "2026-01-21T10:00:00",
                    "open_incident_count": 0,
                },
            ]
        ),
        get_open_incidents=lambda: pd.DataFrame(),
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    rows = backend.get_overview_rows()

    assert len(rows) == 2
    fraud_row = next(row for row in rows if row["model_id"] == "fraud_model_demo")
    assert fraud_row["max_psi"] == 10.97
    assert round(fraud_row["avg_psi"], 2) == 5.51
    assert fraud_row["avg_js"] == 0.32
    assert fraud_row["drifting_features"] == 1
    assert fraud_row["total_features"] == 2
    assert fraud_row["top_drifter"] == "amount"
    assert fraud_row["max_null_rate"] == 1.2
    assert fraud_row["computing"] is False

    chargeback_row = next(row for row in rows if row["model_id"] == "chargeback_model_demo")
    assert chargeback_row["max_psi"] == 4.93
    assert chargeback_row["drifting_features"] == 1
    assert chargeback_row["max_null_rate"] == 0.7
    assert chargeback_row["computing"] is False
    quality_query = next(sql for sql in queries if "ROW_NUMBER() OVER" in sql and "FROM quality_metrics" in sql)
    normalized_quality_query = " ".join(quality_query.split())
    assert ") latest_quality WHERE row_num = 1" in normalized_quality_query
    assert "PARTITION BY quality.model_key ORDER BY quality.computed_at DESC" in normalized_quality_query
    drift_query = next(sql for sql in queries if "feature_metric_history AS" in sql and "FROM drift_metrics" in sql)
    normalized_drift_query = " ".join(drift_query.split())
    assert "MAX(drift.metric_value) AS metric_value" in normalized_drift_query
    assert "GROUP BY drift.model_key, drift.feature_name, drift.metric_name" in normalized_drift_query
    assert "FROM feature_metric_history" in normalized_drift_query
    assert not any("ORDER BY window_end, feature_name, metric_name" in sql for sql in queries)


def test_get_overview_rows_marks_models_without_drift_as_computing() -> None:
    class OverviewWarehouse:
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM quality_metrics" in sql:
                return pd.DataFrame(
                    [
                        {
                            "model_key": "fraud_model_demo",
                            "total_rows": 840,
                            "min_date": "2026-01-01",
                            "max_date": "2026-01-21",
                            "prediction_mean": 0.44,
                            "prediction_std": 0.13,
                            "daily_volume": '{"2026-01-21": 40}',
                            "null_rates": '{"amount": 0.0}',
                            "computed_at": "2026-01-21T10:00:00",
                        }
                    ]
                )
            if "WITH feature_metric_history AS" in sql and "FROM drift_metrics" in sql:
                return pd.DataFrame(
                    [
                        {
                            "model_key": "fraud_model_demo",
                            "feature_name": "amount",
                            "metric_name": "psi",
                            "metric_value": 0.12,
                        }
                    ]
                )
            return pd.DataFrame()

    fraud_config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            feature_columns=("amount", "velocity_7d"),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )
    pending_config = MonitorConfig(
        model_key="spoof_model_demo",
        display_name="Spoof Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            feature_columns=("device_score", "ip_risk", "country_score"),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="spoof_model_v1",
    )

    repository = SimpleNamespace(
        _warehouse=OverviewWarehouse(),
        table_names=SimpleNamespace(
            quality_metrics="quality_metrics",
            drift_metrics="drift_metrics",
        ),
        list_monitor_configs=lambda status="active": [fraud_config, pending_config],
        get_monitor_summary=lambda: pd.DataFrame(),
        list_monitor_runtime_states=lambda keys: {},
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    rows = backend.get_overview_rows(metric="psi")
    rows_by_id = {row["model_id"]: row for row in rows}

    assert rows_by_id["fraud_model_demo"]["computing"] is False
    assert rows_by_id["fraud_model_demo"]["drifting_features"] == 1
    assert rows_by_id["spoof_model_demo"]["computing"] is True
    assert rows_by_id["spoof_model_demo"]["total_features"] == 3
    assert rows_by_id["spoof_model_demo"]["top_drifter"] == "Computing/Pending"


def test_get_overview_rows_uses_monitor_threshold_overrides_for_drifting_features() -> None:
    class OverviewWarehouse:
        def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
            if "FROM quality_metrics" in sql:
                return pd.DataFrame(
                    [
                        {
                            "model_key": "fraud_model_demo",
                            "total_rows": 840,
                            "min_date": "2026-01-01",
                            "max_date": "2026-01-21",
                            "prediction_mean": 0.44,
                            "prediction_std": 0.13,
                            "daily_volume": '{"2026-01-21": 40}',
                            "null_rates": '{"amount": 0.0}',
                            "computed_at": "2026-01-21T10:00:00",
                        }
                    ]
                )
            if "WITH feature_metric_history AS" in sql and "FROM drift_metrics" in sql:
                return pd.DataFrame(
                    [
                        {
                            "model_key": "fraud_model_demo",
                            "feature_name": "amount",
                            "metric_name": "psi",
                            "metric_value": 0.12,
                        }
                    ]
                )
            return pd.DataFrame()

    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            feature_columns=("amount",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
        threshold_overrides={"psi": {"warning": 0.2, "critical": 0.4}},
    )
    repository = SimpleNamespace(
        _warehouse=OverviewWarehouse(),
        table_names=SimpleNamespace(
            quality_metrics="quality_metrics",
            drift_metrics="drift_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame(),
        list_monitor_runtime_states=lambda keys: {},
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    rows = backend.get_overview_rows(metric="psi")

    assert rows[0]["drifting_features"] == 0
    assert rows[0]["threshold_warning"] == 0.2
    assert rows[0]["threshold_critical"] == 0.4


def test_get_overview_rows_returns_empty_without_active_monitors() -> None:
    repository = SimpleNamespace(
        list_monitor_configs=lambda status="active": [],
        get_monitor_summary=lambda: (_ for _ in ()).throw(AssertionError("summary should not be queried")),
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    assert backend.list_models() == []
    assert backend.get_overview_rows() == []


def test_safe_json_helpers_skip_malformed_numeric_values() -> None:
    assert _safe_json_list([1, "bad", None, "4.5"]) == [1.0, 4.5]
    assert _null_rate_dict('{"amount": 1.2, "country": "oops", "velocity_7d": null}') == {"amount": 1.2}


def test_shared_thresholds_cover_metric_specific_warning_logic() -> None:
    warning, critical = get_thresholds("js_divergence")

    assert warning == 0.05
    assert critical == 0.15


def test_get_reference_data_includes_recent_incident_history_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backend_module,
        "resolve_shared_workflow_schedule_status",
        lambda: SharedWorkflowScheduleStatus(
            configured=True,
            resolved=True,
            job_id=123,
            job_name="model-lens-refresh",
            scheduler_mode="cron",
            current_expression="0 0 * * * ?",
            current_interval_hours=1,
            current_label="Every 1 Hour",
            timezone_id="UTC",
            paused=False,
            editable=True,
            supported=True,
            checked_at="2026-01-21T10:06:00Z",
            management_available=True,
        ),
    )
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )
    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        table_names=SimpleNamespace(
            catalog="model_observability",
            schema="control_plane",
            drift_metrics="drift_metrics",
            quality_metrics="quality_metrics",
            quality_history="quality_history",
            performance_metrics="performance_metrics",
        ),
        list_monitor_configs=lambda status="active": [config],
        get_monitor_summary=lambda: pd.DataFrame([{"model_key": "fraud_model_demo", "total_rows": 840}]),
        get_monitor_runtime_state=lambda model_id: None,
        get_recent_refresh_runs=lambda model_id, limit=12: [
            {
                "scope": "bootstrap",
                "status": "completed",
                "started_at": "2026-01-21T10:00:00",
                "completed_at": "2026-01-21T10:06:00",
                "source_metadata_ms": 90_000,
                "daily_profiles_ms": 120_000,
                "derivation_ms": 90_000,
                "persistence_ms": 60_000,
                "total_duration_ms": 360_000,
            },
            {
                "scope": "drift_quality",
                "status": "completed",
                "started_at": "2026-01-20T10:00:00",
                "completed_at": "2026-01-20T10:05:00",
                "source_metadata_ms": 75_000,
                "daily_profiles_ms": 130_000,
                "derivation_ms": 65_000,
                "persistence_ms": 30_000,
                "total_duration_ms": 300_000,
            },
            {
                "scope": "drift_quality",
                "status": "completed",
                "started_at": "2026-01-19T10:00:00",
                "completed_at": "2026-01-19T10:05:10",
                "source_metadata_ms": 80_000,
                "daily_profiles_ms": 140_000,
                "derivation_ms": 60_000,
                "persistence_ms": 30_000,
                "total_duration_ms": 310_000,
            },
        ],
        get_recent_incident_history=lambda model_id, limit=8: [
            {
                "event_type": "opened",
                "feature_name": "amount",
                "metric_name": "psi",
                "severity": "warning",
                "status": "open",
                "metric_value": 0.12,
                "window_end": "2026-01-21",
                "observed_at": "2026-01-21T10:00:00",
            }
        ],
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    reference = backend.get_reference_data("fraud_model_demo")

    assert reference["recent_runs"][0]["scope"] == "bootstrap"
    assert reference["refresh_diagnostics"]["summary"]["dominant_bottleneck"] == "Mixed"
    assert reference["refresh_diagnostics"]["summary"]["successful_run_count"] == 3
    assert reference["recent_incident_history"] == [
        {
            "event_type": "opened",
            "feature_name": "amount",
            "metric_name": "psi",
            "severity": "warning",
            "status": "open",
            "metric_value": 0.12,
            "window_end": "2026-01-21",
            "observed_at": "2026-01-21T10:00:00",
        }
    ]


def test_get_incidents_data_enriches_rows_with_monitor_names() -> None:
    config = MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.model_lens_demo.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="label",
            feature_columns=("amount",),
            slice_columns=("region",),
            categorical_columns=("region",),
        ),
        baseline=BaselinePolicy(n_days=7),
        model_id_value="fraud_model_v1",
    )
    repository = SimpleNamespace(
        _warehouse=_FakeWarehouse(),
        list_monitor_configs=lambda status=None: [config],
        get_open_incidents=lambda: pd.DataFrame(
            [
                {
                    "model_key": "fraud_model_demo",
                    "feature_name": "amount",
                    "metric_name": "psi",
                    "severity": "critical",
                    "metric_value": 0.22,
                    "window_end": "2026-01-21",
                    "observed_at": "2026-01-21T10:00:00",
                }
            ]
        ),
        get_recent_incident_history_all=lambda limit=50: [
            {
                "model_key": "fraud_model_demo",
                "event_type": "opened",
                "feature_name": "amount",
                "metric_name": "psi",
                "severity": "critical",
                "status": "open",
                "metric_value": 0.22,
                "window_end": "2026-01-21",
                "observed_at": "2026-01-21T10:00:00",
            }
        ],
    )
    backend = DashboardBackend(repository=_with_published_generation(repository))

    incidents = backend.get_incidents_data(limit_history=20)

    assert incidents["models"] == [{"id": "fraud_model_demo", "name": "Fraud Model Demo", "status": "active"}]
    assert incidents["open_incidents"].iloc[0]["display_name"] == "Fraud Model Demo"
    assert incidents["open_incidents"].iloc[0]["status"] == "open"
    assert incidents["history"].iloc[0]["display_name"] == "Fraud Model Demo"
