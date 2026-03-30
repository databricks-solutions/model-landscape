from __future__ import annotations

from model_lens.domain.models import BaselinePolicy, MonitorConfig


def build_monitor_config_payload(config: MonitorConfig) -> dict:
    return {
        "model_key": config.model_key,
        "display_name": config.display_name,
        "source_table": config.source_table,
        "timestamp_col": config.contract.timestamp_col,
        "model_id_col": config.contract.model_id_col,
        "model_id_value": config.model_id_value or "",
        "prediction_col": config.contract.prediction_col,
        "model_version_col": config.contract.model_version_col or "",
        "model_version_value": config.model_version_value or "",
        "prediction_score_col": config.contract.prediction_score_col or "",
        "label_col": config.contract.label_col or "",
        "entity_id_col": config.contract.entity_id_col or "",
        "feature_columns": list(config.contract.feature_columns),
        "slice_columns": list(config.contract.slice_columns),
        "categorical_columns": list(config.contract.categorical_columns),
        "baseline_kind": config.baseline.kind,
        "baseline_n_days": config.baseline.n_days,
        "baseline_max_comparison_days": config.baseline.max_comparison_days,
        "baseline_start": config.baseline.baseline_start or "",
        "baseline_end": config.baseline.baseline_end or "",
        "problem_type": config.problem_type,
        "labels_table": config.labels_table or "",
        "labels_join_col": config.labels_join_col or "",
        "labels_order_col": config.labels_order_col or "",
        "mlflow_experiment_name": config.mlflow.experiment_name or "",
        "mlflow_experiment_id": config.mlflow.experiment_id or "",
        "mlflow_run_id": config.mlflow.run_id or "",
        "mlflow_registered_model_name": config.mlflow.registered_model_name or "",
        "mlflow_model_version": config.mlflow.model_version or "",
        "created_by": config.created_by,
        "status": "active",
    }


def build_default_baseline(n_days: int = 7, max_comparison_days: int = 90) -> BaselinePolicy:
    if n_days < 1:
        raise ValueError("n_days must be positive")
    return BaselinePolicy(
        kind="rolling",
        n_days=n_days,
        max_comparison_days=max_comparison_days,
    )


def build_fixed_baseline(
    baseline_start: str,
    baseline_end: str,
    *,
    max_comparison_days: int = 90,
) -> BaselinePolicy:
    return BaselinePolicy(
        kind="fixed",
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        max_comparison_days=max_comparison_days,
    )


def baseline_label(policy: BaselinePolicy) -> str:
    return policy.label
