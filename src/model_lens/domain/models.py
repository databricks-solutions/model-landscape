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
    kind: str = "rolling_n_days"
    n_days: int = 7
    max_comparison_days: int = 90


@dataclass(frozen=True)
class MLflowLineage:
    experiment_name: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None
    registered_model_name: str | None = None
    model_version: str | None = None

    @property
    def connected(self) -> bool:
        return any(
            [
                self.experiment_name,
                self.experiment_id,
                self.run_id,
                self.registered_model_name,
                self.model_version,
            ]
        )


@dataclass(frozen=True)
class MonitorConfig:
    model_key: str
    display_name: str
    source_table: str
    contract: InferenceContract
    baseline: BaselinePolicy = field(default_factory=BaselinePolicy)
    problem_type: str = "classification"
    model_id_value: str | None = None
    model_version_value: str | None = None
    labels_table: str | None = None
    labels_join_col: str | None = None
    labels_order_col: str | None = None
    mlflow: MLflowLineage = field(default_factory=MLflowLineage)
    created_by: str = "app"


@dataclass(frozen=True)
class MLflowDiscovery:
    lineage: MLflowLineage = field(default_factory=MLflowLineage)
    feature_columns: tuple[str, ...] = field(default_factory=tuple)
    problem_type: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MonitorDiscoveryResult:
    config: MonitorConfig
    columns: tuple[str, ...] = field(default_factory=tuple)
    schema_rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    preview_rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    label_columns: tuple[str, ...] = field(default_factory=tuple)
    confidence: str = "high"
    requires_review: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


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
