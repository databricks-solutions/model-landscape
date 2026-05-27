from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "psi": {"warning": 0.1, "critical": 0.2},
    "js_divergence": {"warning": 0.05, "critical": 0.15},
    "kl_divergence": {"warning": 0.1, "critical": 0.3},
    "null_rate": {"warning": 1.0, "critical": 5.0},
}


Severity = Literal["healthy", "warning", "critical"]
ThresholdMap = dict[str, dict[str, float]]
THRESHOLD_METRICS: tuple[str, ...] = tuple(DEFAULT_THRESHOLDS.keys())


def _coerce_threshold_value(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    return numeric


def normalize_threshold_overrides(overrides: Mapping[str, Any] | None = None) -> ThresholdMap:
    normalized: ThresholdMap = {}
    for metric in THRESHOLD_METRICS:
        raw_metric = overrides.get(metric) if isinstance(overrides, Mapping) else None
        if not isinstance(raw_metric, Mapping):
            continue
        warning = _coerce_threshold_value(raw_metric.get("warning"))
        critical = _coerce_threshold_value(raw_metric.get("critical"))
        if warning is None and critical is None:
            continue
        if warning is None or critical is None:
            raise ValueError(
                f"{metric} threshold overrides must provide both warning and critical values."
            )
        if critical <= warning:
            raise ValueError(f"{metric} critical threshold must be greater than warning.")
        normalized[metric] = {
            "warning": float(warning),
            "critical": float(critical),
        }
    return normalized


def merged_thresholds(overrides: Mapping[str, Any] | None = None) -> ThresholdMap:
    resolved: ThresholdMap = {
        metric: {"warning": float(values["warning"]), "critical": float(values["critical"])}
        for metric, values in DEFAULT_THRESHOLDS.items()
    }
    for metric, values in normalize_threshold_overrides(overrides).items():
        resolved[metric] = values
    return resolved


def metric_thresholds(
    metric: str = "psi", overrides: Mapping[str, Any] | None = None
) -> dict[str, float]:
    return merged_thresholds(overrides).get(metric, DEFAULT_THRESHOLDS["psi"])


def get_thresholds(
    metric: str = "psi", overrides: Mapping[str, Any] | None = None
) -> tuple[float, float]:
    thresholds = metric_thresholds(metric, overrides)
    return float(thresholds["warning"]), float(thresholds["critical"])


def drift_severity(
    value: float | int | None, metric: str = "psi", overrides: Mapping[str, Any] | None = None
) -> Severity:
    warning, critical = get_thresholds(metric, overrides)
    numeric = float(value or 0.0)
    if numeric >= critical:
        return "critical"
    if numeric >= warning:
        return "warning"
    return "healthy"
