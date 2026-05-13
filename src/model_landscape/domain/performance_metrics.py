from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceMetricDefinition:
    metric_name: str
    label: str
    problem_type: str
    direction: str
    requires_labels: bool = True
    requires_prediction_score: bool = False


_METRIC_DEFINITIONS = (
    PerformanceMetricDefinition("f1", "F1 Score", "classification", "higher_is_better"),
    PerformanceMetricDefinition("precision", "Precision", "classification", "higher_is_better"),
    PerformanceMetricDefinition("recall", "Recall", "classification", "higher_is_better"),
    PerformanceMetricDefinition("accuracy", "Accuracy", "classification", "higher_is_better"),
    PerformanceMetricDefinition("rmse", "RMSE", "regression", "lower_is_better"),
    PerformanceMetricDefinition("mae", "MAE", "regression", "lower_is_better"),
)

DEFAULT_CLASSIFICATION_PERFORMANCE_METRICS = ("f1", "precision", "recall")
DEFAULT_REGRESSION_PERFORMANCE_METRICS = ("rmse", "mae")


def normalize_problem_type(problem_type: str | None) -> str:
    normalized = str(problem_type or "classification").strip().lower()
    return "regression" if normalized == "regression" else "classification"


def performance_metric_definitions(problem_type: str | None = None) -> tuple[PerformanceMetricDefinition, ...]:
    normalized_problem_type = normalize_problem_type(problem_type)
    return tuple(definition for definition in _METRIC_DEFINITIONS if definition.problem_type == normalized_problem_type)


def default_performance_metric_names(problem_type: str | None) -> tuple[str, ...]:
    normalized_problem_type = normalize_problem_type(problem_type)
    if normalized_problem_type == "regression":
        return DEFAULT_REGRESSION_PERFORMANCE_METRICS
    return DEFAULT_CLASSIFICATION_PERFORMANCE_METRICS


def default_primary_performance_metric(problem_type: str | None) -> str:
    return default_performance_metric_names(problem_type)[0]


def performance_metric_label(metric_name: str) -> str:
    normalized_metric_name = str(metric_name or "").strip().lower()
    for definition in _METRIC_DEFINITIONS:
        if definition.metric_name == normalized_metric_name:
            return definition.label
    return normalized_metric_name.upper()


def performance_metric_options(problem_type: str | None) -> list[dict[str, str]]:
    return [
        {"label": definition.label, "value": definition.metric_name}
        for definition in performance_metric_definitions(problem_type)
    ]


def normalize_performance_metric_names(
    problem_type: str | None,
    metric_names: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    allowed = {definition.metric_name for definition in performance_metric_definitions(problem_type)}
    if not metric_names:
        return default_performance_metric_names(problem_type)
    normalized: list[str] = []
    for metric_name in metric_names:
        normalized_metric_name = str(metric_name or "").strip().lower()
        if not normalized_metric_name:
            continue
        if normalized_metric_name not in allowed:
            raise ValueError(
                f"Unsupported performance metric {metric_name!r} for problem type {normalize_problem_type(problem_type)!r}."
            )
        if normalized_metric_name not in normalized:
            normalized.append(normalized_metric_name)
    return tuple(normalized or default_performance_metric_names(problem_type))


def resolve_default_performance_metric(
    problem_type: str | None,
    metric_names: tuple[str, ...] | list[str] | None,
    default_metric: str | None,
) -> str:
    normalized_metric_names = normalize_performance_metric_names(problem_type, metric_names)
    requested_default = str(default_metric or "").strip().lower()
    if requested_default and requested_default in normalized_metric_names:
        return requested_default
    normalized_problem_type = normalize_problem_type(problem_type)
    if default_metric:
        allowed = {definition.metric_name for definition in performance_metric_definitions(normalized_problem_type)}
        if requested_default and requested_default not in allowed:
            return default_primary_performance_metric(normalized_problem_type)
    preferred_default = default_primary_performance_metric(normalized_problem_type)
    if preferred_default in normalized_metric_names:
        return preferred_default
    return normalized_metric_names[0]
