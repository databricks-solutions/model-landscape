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
        }
    ]


def test_get_drift_results_pivots_long_metrics_into_feature_period_rows() -> None:
    backend = _make_backend()

    drift = backend.get_drift_results("fraud_model_demo", granularity="daily")

    assert list(drift["feature"]) == ["amount"]
    assert list(drift["period"]) == ["2026-01-21"]
    assert drift.iloc[0]["psi"] == 0.21
    assert drift.iloc[0]["js_divergence"] == 0.11
    assert drift.iloc[0]["kl_divergence"] == 0.0


def test_get_quality_stats_parses_json_payloads() -> None:
    backend = _make_backend()

    quality = backend.get_quality_stats("fraud_model_demo")

    assert quality["total_rows"] == 840
    assert quality["daily_volume"] == {"2026-01-21": 40}
    assert quality["null_rates"] == {"amount": 0.0, "velocity_7d": 1.2}


def test_get_performance_summary_keeps_zero_delta_rows_visible() -> None:
    backend = _make_backend()

    performance = backend.get_performance_summary("fraud_model_demo", metric_name="f1")

    assert performance["timeline"]
    assert len(performance["latest_bins"]) == 2
    assert len(performance["all_bins"]) == 2
    assert set(performance["contributors"]["feature"]) == {"amount", "velocity_7d"}
    assert performance["has_significant_degradation"] is False
    assert performance["worst_weighted_delta"] == 0.0
