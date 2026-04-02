from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from model_lens.backend import DashboardBackend
from model_lens.domain.models import BaselinePolicy, InferenceContract, MonitorConfig


class _FakeWarehouse:
    def query_params(self, sql: str, params: tuple) -> pd.DataFrame:
        model_key = params[0]
        if "FROM drift_metrics" in sql:
            return pd.DataFrame(
                [
                    {
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
                        "metric_name": "f1",
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
                        "metric_name": "f1",
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
                        "metric_name": "f1",
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
                        "metric_name": "f1",
                        "window_start": "2026-01-14",
                        "window_end": "2026-01-21",
                        "computed_at": "2026-01-21T10:00:00",
                    },
                ]
            )
        raise AssertionError(f"Unexpected query: {sql}")


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
    return DashboardBackend(repository=repository)


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
    backend = DashboardBackend(repository=repository)

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

    performance = backend.get_performance_summary("fraud_model_demo", metric_name="f1")

    assert performance["timeline"]
    assert len(performance["timeline"]) == 2
    assert len(performance["latest_bins"]) == 2
    assert len(performance["all_bins"]) == 4
    assert set(performance["contributors"]["feature"]) == {"amount", "velocity_7d"}
    assert performance["has_significant_degradation"] is False
    assert performance["worst_weighted_delta"] == 0.0


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
    backend = DashboardBackend(repository=repository)

    baseline, current = backend.get_feature_distribution("fraud_model_demo", "amount")

    assert not baseline.empty
    assert not current.empty
    assert len(calls) == 1
    assert calls[0]["start_date"] == "2026-01-07"
    assert calls[0]["end_date"] == "2026-01-21"
    assert calls[0]["feature_columns"] == ("amount",)
    assert calls[0]["sample_rows_per_day"] > 0
    assert calls[0]["max_total_rows"] > 0
