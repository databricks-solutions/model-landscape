from __future__ import annotations

from datetime import datetime, timezone


DEFAULT_THRESHOLDS = {
    "psi": {"warning": 0.1, "critical": 0.25},
    "js_divergence": {"warning": 0.05, "critical": 0.15},
    "kl_divergence": {"warning": 0.1, "critical": 0.25},
}


def build_incidents(drift_rows: list[dict], thresholds: dict | None = None) -> list[dict]:
    active_thresholds = thresholds or DEFAULT_THRESHOLDS
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    incidents: dict[tuple[str, str, str], dict] = {}

    for row in drift_rows:
        metric_name = row["metric_name"]
        metric_thresholds = active_thresholds.get(metric_name)
        if not metric_thresholds:
            continue

        metric_value = float(row["metric_value"])
        severity = None
        if metric_value >= metric_thresholds["critical"]:
            severity = "critical"
        elif metric_value >= metric_thresholds["warning"]:
            severity = "warning"

        if not severity:
            continue

        key = (row["model_key"], row["feature_name"], metric_name)
        existing = incidents.get(key)
        candidate = {
            "model_key": row["model_key"],
            "feature_name": row["feature_name"],
            "metric_name": metric_name,
            "severity": severity,
            "status": "open",
            "metric_value": metric_value,
            "observed_at": now,
            "window_end": row["window_end"],
        }
        if existing is None or metric_value > existing["metric_value"]:
            incidents[key] = candidate

    return list(incidents.values())

