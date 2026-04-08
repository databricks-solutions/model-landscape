from __future__ import annotations

from typing import Literal


DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "psi": {"warning": 0.1, "critical": 0.2},
    "js_divergence": {"warning": 0.05, "critical": 0.15},
    "kl_divergence": {"warning": 0.1, "critical": 0.3},
    "null_rate": {"warning": 1.0, "critical": 5.0},
}


Severity = Literal["healthy", "warning", "critical"]


def metric_thresholds(metric: str = "psi") -> dict[str, float]:
    return DEFAULT_THRESHOLDS.get(metric, DEFAULT_THRESHOLDS["psi"])


def get_thresholds(metric: str = "psi") -> tuple[float, float]:
    thresholds = metric_thresholds(metric)
    return float(thresholds["warning"]), float(thresholds["critical"])


def drift_severity(value: float | int | None, metric: str = "psi") -> Severity:
    warning, critical = get_thresholds(metric)
    numeric = float(value or 0.0)
    if numeric >= critical:
        return "critical"
    if numeric >= warning:
        return "warning"
    return "healthy"
