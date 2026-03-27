from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REQUIRED_INFERENCE_COLUMNS = ("event_ts", "model_id", "prediction")
OPTIONAL_INFERENCE_COLUMNS = ("model_version", "prediction_proba", "label", "entity_id")


@dataclass(frozen=True)
class InferenceContract:
    timestamp_col: str
    model_id_col: str
    prediction_col: str
    model_version_col: str | None = None
    prediction_score_col: str | None = None
    label_col: str | None = None
    entity_id_col: str | None = None
    feature_columns: tuple[str, ...] = field(default_factory=tuple)
    slice_columns: tuple[str, ...] = field(default_factory=tuple)
    categorical_columns: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BaselinePolicy:
    kind: str = "first_n_days"
    n_days: int = 7
    max_comparison_days: int = 90


@dataclass(frozen=True)
class MonitorConfig:
    model_key: str
    display_name: str
    source_table: str
    contract: InferenceContract
    baseline: BaselinePolicy = field(default_factory=BaselinePolicy)
    problem_type: str = "classification"
    labels_table: str | None = None
    labels_join_col: str | None = None
    created_by: str = "app"


@dataclass(frozen=True)
class IncidentKey:
    model_key: str
    feature_name: str
    metric_name: str


@dataclass(frozen=True)
class IncidentRecord:
    key: IncidentKey
    severity: str
    status: str
    metric_value: float
    observed_at: str


@dataclass(frozen=True)
class RefreshResult:
    drift_rows: list[dict[str, Any]]
    quality_rows: list[dict[str, Any]]
    performance_rows: list[dict[str, Any]]
    incident_rows: list[dict[str, Any]]

