from datetime import datetime, timedelta

import pandas as pd

from model_lens.domain.models import MonitorConfig
from model_lens.services.inference_contracts import build_inference_contract as build_contract
from model_lens.services.onboarding import build_default_baseline, build_fixed_baseline
from model_lens.services.refresh_engine import (
    build_daily_feature_profile_rows,
    build_daily_performance_profile_rows,
    build_daily_quality_profile_rows,
    build_performance_bin_specs,
    derive_refresh_result_from_daily_profiles,
    generate_window_metadata,
    generate_window_pairs,
    refresh_monitor,
    refresh_monitor_backfill,
    split_baseline_current,
)


def test_split_baseline_current_uses_two_recent_adjacent_windows() -> None:
    start = datetime(2026, 1, 1)
    frame = pd.DataFrame({
        "event_ts": [start + timedelta(days=index) for index in range(10)],
        "model_id": ["m1"] * 10,
        "prediction": [0.1 * index for index in range(10)],
        "f1": [float(index) for index in range(10)],
    })
    baseline, current = split_baseline_current(frame, "event_ts", 3)
    assert len(baseline) == 3
    assert len(current) == 3
    assert baseline["event_ts"].max() < current["event_ts"].min()
    assert baseline["event_ts"].min().date().isoformat() == "2026-01-05"
    assert current["event_ts"].min().date().isoformat() == "2026-01-08"


def test_generate_window_pairs_backfills_all_rolling_windows() -> None:
    start = datetime(2026, 1, 1)
    frame = pd.DataFrame({
        "event_ts": [start + timedelta(days=index) for index in range(21)],
        "model_id": ["m1"] * 21,
        "prediction": [0.1 * index for index in range(21)],
        "f1": [float(index) for index in range(21)],
    })

    pairs = generate_window_pairs(frame, "event_ts", build_default_baseline(7))

    assert len(pairs) == 8
    assert {key: pairs[0][2][key] for key in ("baseline_start", "baseline_end", "window_start", "window_end")} == {
        "baseline_start": "2026-01-01",
        "baseline_end": "2026-01-07",
        "window_start": "2026-01-08",
        "window_end": "2026-01-14",
    }
    assert pairs[0][2]["baseline_kind"] == "rolling"
    assert pairs[0][2]["window_grain"] == "daily"
    assert {key: pairs[-1][2][key] for key in ("baseline_start", "baseline_end", "window_start", "window_end")} == {
        "baseline_start": "2026-01-08",
        "baseline_end": "2026-01-14",
        "window_start": "2026-01-15",
        "window_end": "2026-01-21",
    }


def test_refresh_monitor_backfill_produces_all_window_rows() -> None:
    start = datetime(2026, 1, 1)
    rows = []
    for day in range(21):
        regime = 0 if day < 7 else 1 if day < 14 else 2
        for offset in range(2):
            rows.append({
                "event_ts": start + timedelta(days=day, hours=offset),
                "model_id": "m1",
                "prediction": [0.15, 0.35, 0.55, 0.85][regime + offset if regime < 2 else 2 + offset],
                "label": 0 if regime < 2 else 1,
                "f1": float((day * 2) + offset + (regime * 5)),
                "f2": float((day * 3) + offset + (regime * 7)),
            })
    frame = pd.DataFrame(rows)
    contract = build_contract(
        columns=list(frame.columns),
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        feature_columns=["f1", "f2"],
    )

    result = refresh_monitor_backfill(
        MonitorConfig(
            model_key="m1",
            display_name="Model 1",
            source_table="cat.sch.logs",
            contract=contract,
            baseline=build_default_baseline(),
        ),
        inference_df=frame,
    )

    assert len({row["window_end"] for row in result.drift_rows}) == 8
    assert len(result.drift_rows) == 8 * 2 * 3
    assert len({row["window_end"] for row in result.performance_rows}) == 7
    assert {row["metric_name"] for row in result.performance_rows} == {"precision"}
    assert len(result.window_rows) == 8
    assert isinstance(result.incident_history_rows, list)
    assert all(row["window_id"] for row in result.incident_history_rows)
    assert result.window_rows[0]["window_id"].startswith("m1|rolling|2026-01-01|2026-01-07|2026-01-08|2026-01-14")
    assert result.quality_rows[0]["max_date"] == "2026-01-21"
    assert all(row["window_end"] == "2026-01-21" for row in result.incident_rows)


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
    assert result.drift_rows[0]["window_start"] == "2026-01-14"
    if result.performance_rows:
        assert result.performance_rows[0]["window_start"] == "2026-01-14"


def test_split_baseline_current_supports_fixed_baseline_range() -> None:
    start = datetime(2026, 1, 1)
    frame = pd.DataFrame({
        "event_ts": [start + timedelta(days=index) for index in range(20)],
        "model_id": ["m1"] * 20,
        "prediction": [0.1 * index for index in range(20)],
        "f1": [float(index) for index in range(20)],
    })

    baseline, current = split_baseline_current(
        frame,
        "event_ts",
        build_fixed_baseline("2026-01-01", "2026-01-05"),
    )

    assert baseline["event_ts"].min().date().isoformat() == "2026-01-01"
    assert baseline["event_ts"].max().date().isoformat() == "2026-01-05"
    assert current["event_ts"].min().date().isoformat() == "2026-01-16"
    assert current["event_ts"].max().date().isoformat() == "2026-01-20"


def test_generate_window_pairs_supports_fixed_baseline_history() -> None:
    start = datetime(2026, 1, 1)
    frame = pd.DataFrame({
        "event_ts": [start + timedelta(days=index) for index in range(20)],
        "model_id": ["m1"] * 20,
        "prediction": [0.1 * index for index in range(20)],
        "f1": [float(index) for index in range(20)],
    })

    pairs = generate_window_pairs(
        frame,
        "event_ts",
        build_fixed_baseline("2026-01-01", "2026-01-05"),
    )

    assert len(pairs) == 11
    assert {key: pairs[0][2][key] for key in ("baseline_start", "baseline_end", "window_start", "window_end")} == {
        "baseline_start": "2026-01-01",
        "baseline_end": "2026-01-05",
        "window_start": "2026-01-06",
        "window_end": "2026-01-10",
    }
    assert pairs[0][2]["baseline_kind"] == "fixed"
    assert {key: pairs[-1][2][key] for key in ("baseline_start", "baseline_end", "window_start", "window_end")} == {
        "baseline_start": "2026-01-01",
        "baseline_end": "2026-01-05",
        "window_start": "2026-01-16",
        "window_end": "2026-01-20",
    }


def test_refresh_monitor_backfill_supports_regression_metrics_and_categorical_drift() -> None:
    start = datetime(2026, 1, 1)
    rows = []
    for day in range(21):
        for offset in range(2):
            rows.append({
                "event_ts": start + timedelta(days=day, hours=offset),
                "model_id": "m1",
                "prediction": float(day + offset),
                "label": float(day + (offset * 0.5)),
                "amount": float((day * 2) + offset),
                "segment": "baseline" if day < 10 else "current",
            })
    frame = pd.DataFrame(rows)
    contract = build_contract(
        columns=list(frame.columns),
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        feature_columns=["amount", "segment"],
        categorical_columns=["segment"],
    )

    result = refresh_monitor_backfill(
        MonitorConfig(
            model_key="m1",
            display_name="Regression Model",
            source_table="cat.sch.logs",
            contract=contract,
            baseline=build_default_baseline(),
            problem_type="regression",
        ),
        inference_df=frame,
    )

    assert {"rmse", "mae"} == {row["metric_name"] for row in result.performance_rows}
    assert any(row["feature_name"] == "segment" and row["metric_name"] == "psi" for row in result.drift_rows)


def test_daily_profile_builders_emit_quality_feature_and_performance_rows() -> None:
    start = datetime(2026, 1, 1)
    frame = pd.DataFrame([
        {
            "event_ts": start + timedelta(hours=offset),
            "model_id": "m1",
            "prediction": 0.1 + (offset * 0.1),
            "label": offset % 2,
            "amount": 10.0 + offset,
            "segment": "a" if offset < 6 else "b",
        }
        for offset in range(12)
    ] + [
        {
            "event_ts": start + timedelta(days=1, hours=offset),
            "model_id": "m1",
            "prediction": 0.2 + (offset * 0.05),
            "label": (offset + 1) % 2,
            "amount": 20.0 + offset,
            "segment": "b" if offset < 6 else "c",
        }
        for offset in range(12)
    ])
    contract = build_contract(
        columns=list(frame.columns),
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        feature_columns=["amount", "segment"],
        categorical_columns=["segment"],
    )
    config = MonitorConfig(
        model_key="m1",
        display_name="Model 1",
        source_table="cat.sch.logs",
        contract=contract,
        baseline=build_default_baseline(),
    )

    quality_rows = build_daily_quality_profile_rows(config=config, inference_df=frame, computed_at="2026-01-02T00:00:00Z")
    feature_rows = build_daily_feature_profile_rows(config=config, inference_df=frame, computed_at="2026-01-02T00:00:00Z")
    performance_rows = build_daily_performance_profile_rows(config=config, inference_df=frame, computed_at="2026-01-02T00:00:00Z")

    assert {row["profile_date"] for row in quality_rows} == {"2026-01-01", "2026-01-02"}
    assert {row["feature_name"] for row in feature_rows} == {"amount", "segment"}
    assert {row["feature_kind"] for row in feature_rows} == {"numeric", "categorical"}
    assert performance_rows
    assert {row["metric_name"] for row in performance_rows} == {"f1", "precision", "recall"}


def test_daily_performance_profiles_reuse_canonical_bin_specs_across_runs() -> None:
    bootstrap_frame = pd.DataFrame([
        {
            "event_ts": datetime(2026, 1, 1, hour=offset),
            "model_id": "m1",
            "prediction": 0.8 if offset % 2 else 0.2,
            "label": offset % 2,
            "amount": float(offset),
        }
        for offset in range(12)
    ])
    incremental_frame = pd.DataFrame([
        {
            "event_ts": datetime(2026, 1, 2, hour=offset),
            "model_id": "m1",
            "prediction": 0.8 if offset % 2 else 0.2,
            "label": offset % 2,
            "amount": float(100 + offset),
        }
        for offset in range(12)
    ])
    contract = build_contract(
        columns=list(bootstrap_frame.columns),
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
    )

    bootstrap_specs = build_performance_bin_specs(
        config=config,
        inference_df=bootstrap_frame,
        n_bins=4,
    )
    incremental_specs = build_performance_bin_specs(
        config=config,
        inference_df=incremental_frame,
        existing_specs=bootstrap_specs,
        n_bins=4,
    )
    reused_rows = build_daily_performance_profile_rows(
        config=config,
        inference_df=incremental_frame,
        computed_at="2026-01-02T00:00:00Z",
        bin_specs=incremental_specs,
        n_bins=4,
    )
    ad_hoc_rows = build_daily_performance_profile_rows(
        config=config,
        inference_df=incremental_frame,
        computed_at="2026-01-02T00:00:00Z",
        n_bins=4,
    )

    assert incremental_specs == bootstrap_specs
    assert reused_rows
    assert ad_hoc_rows
    assert {row["bin_label"] for row in reused_rows} != {row["bin_label"] for row in ad_hoc_rows}
    assert all("100" not in row["bin_label"] for row in reused_rows)


def test_daily_profiles_can_derive_window_history_without_raw_window_reloads() -> None:
    start = datetime(2026, 1, 1)
    rows = []
    for day in range(21):
        regime = 0 if day < 7 else 1 if day < 14 else 2
        for offset in range(2):
            rows.append({
                "event_ts": start + timedelta(days=day, hours=offset),
                "model_id": "m1",
                "prediction": [0.15, 0.35, 0.55, 0.85][regime + offset if regime < 2 else 2 + offset],
                "label": 0 if regime < 2 else 1,
                "amount": float((day * 2) + offset + (regime * 4)),
                "segment": "baseline" if day < 10 else "current",
            })
    frame = pd.DataFrame(rows)
    contract = build_contract(
        columns=list(frame.columns),
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        label_col="label",
        feature_columns=["amount", "segment"],
        categorical_columns=["segment"],
    )
    config = MonitorConfig(
        model_key="m1",
        display_name="Model 1",
        source_table="cat.sch.logs",
        contract=contract,
        baseline=build_default_baseline(),
    )
    computed_at = "2026-01-22T00:00:00Z"

    quality_rows = build_daily_quality_profile_rows(config=config, inference_df=frame, computed_at=computed_at)
    feature_rows = build_daily_feature_profile_rows(config=config, inference_df=frame, computed_at=computed_at)
    performance_rows = build_daily_performance_profile_rows(config=config, inference_df=frame, computed_at=computed_at)
    metadata_list = generate_window_metadata(
        min_date="2026-01-01",
        max_date="2026-01-21",
        baseline=config.baseline,
        model_key=config.model_key,
    )

    result = derive_refresh_result_from_daily_profiles(
        config=config,
        metadata_list=metadata_list,
        daily_quality_profile_rows=quality_rows,
        daily_feature_profile_rows=feature_rows,
        daily_performance_profile_rows=performance_rows,
        computed_at=computed_at,
    )

    assert len(result.window_rows) == 8
    assert len(result.quality_history_rows) == 8
    assert len({row["window_end"] for row in result.drift_rows}) == 8
    assert any(row["feature_name"] == "segment" and row["metric_name"] == "psi" for row in result.drift_rows)
    assert {row["metric_name"] for row in result.performance_rows} == {"precision"}
    assert all(row["window_id"] for row in result.performance_rows)
    assert all(row["window_id"] for row in result.incident_history_rows)
