from __future__ import annotations

from datetime import date
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
    kind: str = "rolling"
    n_days: int = 7
    max_comparison_days: int = 90
    baseline_start: str | None = None
    baseline_end: str | None = None

    def __post_init__(self) -> None:
        normalized_kind = (self.kind or "rolling").strip().lower()
        if normalized_kind == "rolling_n_days":
            normalized_kind = "rolling"
        if normalized_kind not in {"rolling", "fixed"}:
            raise ValueError("Baseline kind must be 'rolling' or 'fixed'.")
        object.__setattr__(self, "kind", normalized_kind)

        if self.max_comparison_days < 1:
            raise ValueError("max_comparison_days must be positive")

        if normalized_kind == "rolling":
            if self.n_days < 1:
                raise ValueError("n_days must be positive")
            object.__setattr__(self, "baseline_start", None)
            object.__setattr__(self, "baseline_end", None)
            return

        start = _coerce_iso_date(self.baseline_start)
        end = _coerce_iso_date(self.baseline_end)
        if not start or not end:
            raise ValueError("Fixed baselines require baseline_start and baseline_end.")
        if end < start:
            raise ValueError("baseline_end must be on or after baseline_start.")
        object.__setattr__(self, "baseline_start", start)
        object.__setattr__(self, "baseline_end", end)
        object.__setattr__(self, "n_days", (date.fromisoformat(end) - date.fromisoformat(start)).days + 1)

    @property
    def is_fixed(self) -> bool:
        return self.kind == "fixed"

    @property
    def label(self) -> str:
        if self.is_fixed and self.baseline_start and self.baseline_end:
            return f"Fixed: {self.baseline_start} to {self.baseline_end}"
        return f"Rolling: {self.n_days} days"


def _coerce_iso_date(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    return date.fromisoformat(text).isoformat()


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
    label_schema_rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    label_preview_rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    label_validation: dict[str, Any] = field(default_factory=dict)
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
    incident_history_rows: list[dict[str, Any]] = field(default_factory=list)
    quality_history_rows: list[dict[str, Any]] = field(default_factory=list)
    window_rows: list[dict[str, Any]] = field(default_factory=list)
