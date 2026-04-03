from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pandas as pd

from model_lens.domain.models import MonitorConfig
from model_lens.services.contracts import build_contract
from model_lens.services.onboarding import build_default_baseline
from model_lens.services.refresh_engine import build_daily_performance_profile_rows


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
