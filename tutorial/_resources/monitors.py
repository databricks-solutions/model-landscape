# Databricks notebook source
"""Tutorial helper: build a MonitorConfig from the common 80% of fields."""

from __future__ import annotations

from model_landscape.domain.models import (
    BaselinePolicy,
    InferenceContract,
    MLflowLineage,
    MonitorConfig,
)


FRAUD_FEATURES: tuple[str, ...] = (
    "transaction_amount", "device_trust_score", "distance_from_home_km",
    "velocity_24h", "hour_of_day", "is_weekend", "account_age_days",
    "merchant_category", "region",
)

MAINT_FEATURES: tuple[str, ...] = (
    "vibration_mm_s", "temperature_c", "pressure_kpa", "rpm",
    "oil_viscosity", "power_output_kw", "ambient_temp_c",
    "operating_hours_since_service", "load_factor",
    "equipment_class", "site",
)


def make_monitor_config(
    *,
    model_key: str,
    display_name: str,
    source_table: str,
    problem_type: str,
    feature_columns: tuple[str, ...],
    timestamp_col: str = "event_ts",
    prediction_col: str = "prediction",
    entity_id_col: str = "entity_id",
    model_version_col: str = "model_version",
    prediction_score_col: str | None = None,
    slice_columns: tuple[str, ...] = (),
    categorical_columns: tuple[str, ...] = (),
    labels_table: str | None = None,
    labels_join_col: str | None = None,
    labels_order_col: str | None = None,
    mlflow_experiment: str | None = None,
    baseline_days: int = 7,
    created_by: str = "tutorial",
) -> MonitorConfig:
    """Tutorial helper: build a MonitorConfig from the common 80% of fields.

    Most tutorial monitors share the same baseline policy (rolling 7-day),
    cadence presets (daily drift, daily 7-day-repair performance), and
    schedule flag. This collapses those defaults so notebook callsites can
    focus on what's *different* about each monitor (problem type, columns).
    """
    return MonitorConfig(
        model_key=model_key,
        display_name=display_name,
        source_table=source_table,
        problem_type=problem_type,
        contract=InferenceContract(
            timestamp_col=timestamp_col,
            prediction_col=prediction_col,
            prediction_score_col=prediction_score_col,
            entity_id_col=entity_id_col,
            model_version_col=model_version_col,
            feature_columns=tuple(feature_columns),
            slice_columns=tuple(slice_columns),
            categorical_columns=tuple(categorical_columns),
        ),
        baseline=BaselinePolicy(kind="rolling", n_days=baseline_days),
        labels_table=labels_table,
        labels_join_col=labels_join_col,
        labels_order_col=labels_order_col,
        drift_cadence_preset="daily",
        performance_cadence_preset="daily_7d_repair",
        schedule_enabled=True,
        mlflow=MLflowLineage(experiment_name=mlflow_experiment) if mlflow_experiment else None,
        created_by=created_by,
    )
