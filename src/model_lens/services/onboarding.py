from __future__ import annotations

from model_lens.domain.models import BaselinePolicy, MonitorConfig


def build_monitor_config_payload(config: MonitorConfig) -> dict:
    return {
        "model_key": config.model_key,
        "display_name": config.display_name,
        "source_table": config.source_table,
        "timestamp_col": config.contract.timestamp_col,
        "model_id_col": config.contract.model_id_col,
        "prediction_col": config.contract.prediction_col,
        "model_version_col": config.contract.model_version_col or "",
        "prediction_score_col": config.contract.prediction_score_col or "",
        "label_col": config.contract.label_col or "",
        "entity_id_col": config.contract.entity_id_col or "",
        "feature_columns": list(config.contract.feature_columns),
        "slice_columns": list(config.contract.slice_columns),
        "categorical_columns": list(config.contract.categorical_columns),
        "baseline_kind": config.baseline.kind,
        "baseline_n_days": config.baseline.n_days,
        "baseline_max_comparison_days": config.baseline.max_comparison_days,
        "problem_type": config.problem_type,
        "labels_table": config.labels_table or "",
        "labels_join_col": config.labels_join_col or "",
        "created_by": config.created_by,
        "status": "active",
    }


def build_default_baseline(n_days: int = 7, max_comparison_days: int = 90) -> BaselinePolicy:
    if n_days < 1:
        raise ValueError("n_days must be positive")
    return BaselinePolicy(
        kind="first_n_days",
        n_days=n_days,
        max_comparison_days=max_comparison_days,
    )
