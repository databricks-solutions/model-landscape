from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pandas as pd

from model_lens.analytics.performance import (
    compute_bin_edges,
    compute_classification_metrics,
    compute_daily_classification_metrics,
)
from model_lens.domain.models import MonitorConfig
from model_lens.services.inference_contracts import build_inference_contract as build_contract
from model_lens.services.onboarding import build_default_baseline
from model_lens.services.refresh_engine import build_daily_performance_profile_rows, build_performance_bin_specs


def _classification_contract() -> object:
    frame = pd.DataFrame(
        [
            {
                "event_ts": datetime(2026, 1, 1) + timedelta(hours=offset),
                "model_id": "m1",
                "prediction": 0.8 if offset % 2 else 0.2,
                "label": offset % 2,
                "amount": float(offset + 1),
            }
            for offset in range(12)
        ]
    )
    return build_contract(
        columns=list(frame.columns),
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        feature_columns=["amount"],
    ), frame


def test_monitor_config_defaults_classification_performance_metrics() -> None:
    contract, _ = _classification_contract()

    config = MonitorConfig(
        model_key="m1",
        display_name="Model 1",
        source_table="cat.sch.logs",
        contract=contract,
        baseline=build_default_baseline(),
        problem_type="classification",
    )

    assert config.performance_metric_names == ("f1", "precision", "recall")
    assert config.default_performance_metric == "f1"
    assert config.performance_binning_mode == "quantile"
    assert config.performance_binning_clip_percentile is None


def test_monitor_config_rejects_invalid_performance_binning_config() -> None:
    contract, _ = _classification_contract()

    with pytest.raises(ValueError):
        MonitorConfig(
            model_key="m1",
            display_name="Model 1",
            source_table="cat.sch.logs",
            contract=contract,
            baseline=build_default_baseline(),
            performance_binning_mode="weird",
        )

    with pytest.raises(ValueError):
        MonitorConfig(
            model_key="m1",
            display_name="Model 1",
            source_table="cat.sch.logs",
            contract=contract,
            baseline=build_default_baseline(),
            performance_binning_clip_percentile=50.0,
        )


def test_compute_bin_edges_supports_quantile_default_and_optional_clip() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]).to_numpy(dtype=float)

    quantile_edges = compute_bin_edges(values, n_bins=4)
    fixed_edges = compute_bin_edges(values, n_bins=4, mode="fixed_width")
    clipped_edges = compute_bin_edges(values, n_bins=4, mode="fixed_width", clip_percentile=10.0)

    assert len(quantile_edges) >= 2
    assert fixed_edges[-1] == 1000.0
    assert clipped_edges[-1] < fixed_edges[-1]


def test_build_performance_bin_specs_uses_quantile_binning_by_default() -> None:
    contract, frame = _classification_contract()
    frame = frame.copy()
    frame["amount"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 1000.0]
    config = MonitorConfig(
        model_key="m1",
        display_name="Model 1",
        source_table="cat.sch.logs",
        contract=contract,
        baseline=build_default_baseline(),
    )

    specs = build_performance_bin_specs(config=config, inference_df=frame, n_bins=4)

    assert "amount" in specs
    assert specs["amount"][-1] == 1000.0
    assert specs["amount"][1] < 100.0


def test_monitor_config_rejects_cross_problem_performance_metric_names() -> None:
    contract, _ = _classification_contract()

    with pytest.raises(ValueError):
        MonitorConfig(
            model_key="m1",
            display_name="Model 1",
            source_table="cat.sch.logs",
            contract=contract,
            baseline=build_default_baseline(),
            problem_type="classification",
            performance_metric_names=("rmse",),
        )


def test_daily_performance_profiles_include_optional_accuracy_when_configured() -> None:
    contract, frame = _classification_contract()
    config = MonitorConfig(
        model_key="m1",
        display_name="Model 1",
        source_table="cat.sch.logs",
        contract=contract,
        baseline=build_default_baseline(),
        problem_type="classification",
        performance_metric_names=("f1", "precision", "recall", "accuracy"),
        default_performance_metric="accuracy",
    )

    rows = build_daily_performance_profile_rows(
        config=config,
        inference_df=frame,
        computed_at="2026-01-02T00:00:00Z",
    )

    assert {row["metric_name"] for row in rows} == {"f1", "precision", "recall", "accuracy"}


def test_daily_performance_profiles_skip_undefined_classification_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "event_ts": datetime(2026, 1, 1) + timedelta(hours=offset),
                "model_id": "m1",
                "prediction": 0.1,
                "label": 0,
                "amount": float(offset + 1),
            }
            for offset in range(12)
        ]
    )
    contract = build_contract(
        columns=list(frame.columns),
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        feature_columns=["amount"],
    )
    config = MonitorConfig(
        model_key="m1",
        display_name="Model 1",
        source_table="cat.sch.logs",
        contract=contract,
        baseline=build_default_baseline(),
        problem_type="classification",
        performance_metric_names=("f1", "precision", "recall", "accuracy"),
        default_performance_metric="accuracy",
    )

    rows = build_daily_performance_profile_rows(
        config=config,
        inference_df=frame,
        computed_at="2026-01-02T00:00:00Z",
    )

    assert rows
    assert {row["metric_name"] for row in rows} == {"accuracy"}


def test_daily_performance_profiles_skip_precision_recall_and_f1_when_no_detections_occur() -> None:
    frame = pd.DataFrame(
        [
            {
                "event_ts": datetime(2026, 1, 1) + timedelta(hours=offset),
                "model_id": "m1",
                "prediction": 0.1,
                "label": 1 if offset < 4 else 0,
                "amount": float(offset + 1),
            }
            for offset in range(12)
        ]
    )
    contract = build_contract(
        columns=list(frame.columns),
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        feature_columns=["amount"],
    )
    config = MonitorConfig(
        model_key="m1",
        display_name="Model 1",
        source_table="cat.sch.logs",
        contract=contract,
        baseline=build_default_baseline(),
        problem_type="classification",
        performance_metric_names=("f1", "precision", "recall", "accuracy"),
        default_performance_metric="accuracy",
    )

    rows = build_daily_performance_profile_rows(
        config=config,
        inference_df=frame,
        computed_at="2026-01-02T00:00:00Z",
    )

    assert rows
    assert {row["metric_name"] for row in rows} == {"accuracy"}


def test_classification_metrics_prefer_discrete_prediction_labels_over_prediction_score_col() -> None:
    frame = pd.DataFrame(
        [
            {"prediction": "negative", "prediction_score": 0.9, "label": 1},
            {"prediction": "positive", "prediction_score": 0.1, "label": 0},
            {"prediction": "positive", "prediction_score": 0.8, "label": 1},
            {"prediction": "negative", "prediction_score": 0.2, "label": 0},
        ]
    )

    with_score = compute_classification_metrics(
        frame,
        prediction_col="prediction",
        label_col="label",
        prediction_score_col="prediction_score",
    )
    without_score = compute_classification_metrics(
        frame,
        prediction_col="prediction",
        label_col="label",
    )

    assert without_score == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "accuracy": 0.5}
    assert with_score == without_score


def test_classification_metrics_use_prediction_score_col_when_prediction_labels_are_not_binary() -> None:
    frame = pd.DataFrame(
        [
            {"prediction": "watch", "prediction_score": 0.9, "label": 1},
            {"prediction": "watch", "prediction_score": 0.1, "label": 0},
            {"prediction": "watch", "prediction_score": 0.8, "label": 1},
            {"prediction": "watch", "prediction_score": 0.2, "label": 0},
        ]
    )

    with_score = compute_classification_metrics(
        frame,
        prediction_col="prediction",
        label_col="label",
        prediction_score_col="prediction_score",
    )
    without_score = compute_classification_metrics(
        frame,
        prediction_col="prediction",
        label_col="label",
    )

    assert with_score == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}
    assert without_score == {}


def test_classification_metrics_return_nulls_for_undefined_precision_recall_and_f1() -> None:
    frame = pd.DataFrame(
        [
            {"prediction": 0.1, "label": 0},
            {"prediction": 0.2, "label": 0},
            {"prediction": 0.3, "label": 0},
        ]
    )

    metrics = compute_classification_metrics(
        frame,
        prediction_col="prediction",
        label_col="label",
    )

    assert metrics == {"precision": None, "recall": None, "f1": None, "accuracy": 1.0}


def test_daily_classification_metrics_hide_classification_scores_when_no_positive_detections_exist() -> None:
    frame = pd.DataFrame(
        [
            {"prediction": 0.1, "label": 1},
            {"prediction": 0.2, "label": 1},
            {"prediction": 0.3, "label": 0},
        ]
    )

    metrics = compute_daily_classification_metrics(
        frame,
        prediction_col="prediction",
        label_col="label",
    )

    assert metrics == {
        "actual_positive_count": 2,
        "actual_negative_count": 1,
        "predicted_positive_count": 0,
        "predicted_negative_count": 3,
        "tp": 0,
        "fp": 0,
        "fn": 2,
        "tn": 1,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": 0.3333,
    }
