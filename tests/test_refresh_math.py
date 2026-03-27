from datetime import datetime, timedelta

import pandas as pd

from model_lens.domain.models import MonitorConfig
from model_lens.services.contracts import build_contract
from model_lens.services.onboarding import build_default_baseline
from model_lens.workflows.refresh_job import refresh_monitor, split_baseline_current


def test_split_baseline_current_avoids_overlap() -> None:
    start = datetime(2026, 1, 1)
    frame = pd.DataFrame({
        "event_ts": [start + timedelta(days=index) for index in range(10)],
        "model_id": ["m1"] * 10,
        "prediction": [0.1 * index for index in range(10)],
        "f1": [float(index) for index in range(10)],
    })
    baseline, current = split_baseline_current(frame, "event_ts", 7)
    assert len(baseline) == 7
    assert len(current) == 3
    assert baseline["event_ts"].max() < current["event_ts"].min()


def test_refresh_monitor_produces_deduplicated_incidents() -> None:
    start = datetime(2026, 1, 1)
    frame = pd.DataFrame({
        "event_ts": [start + timedelta(days=index) for index in range(20)],
        "model_id": ["m1"] * 20,
        "prediction": [0.2] * 10 + [0.8] * 10,
        "label": [0] * 10 + [1] * 10,
        "f1": [1.0] * 10 + [10.0] * 10,
        "f2": [1.0] * 10 + [20.0] * 10,
    })
    contract = build_contract(
        columns=list(frame.columns),
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        feature_columns=["f1", "f2"],
    )
    result = refresh_monitor(
        MonitorConfig(
            model_key="m1",
            display_name="Model 1",
            source_table="cat.sch.logs",
            contract=contract,
            baseline=build_default_baseline(),
        ),
        inference_df=frame,
    )
    assert result.drift_rows
    assert len({(row["model_key"], row["feature_name"], row["metric_name"]) for row in result.incident_rows}) == len(result.incident_rows)
