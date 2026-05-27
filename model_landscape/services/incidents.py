from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from model_landscape.services.thresholds import merged_thresholds

_SEVERITY_RANK = {"warning": 1, "critical": 2}


def _observed_at(row: dict[str, Any]) -> str:
    recorded = str(row.get("computed_at") or row.get("observed_at") or "").strip()
    if recorded:
        return recorded
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _incident_candidate(
    row: dict[str, Any], thresholds: dict[str, dict[str, float]]
) -> dict[str, Any] | None:
    active_thresholds = merged_thresholds(thresholds)
    metric_name = str(row["metric_name"])
    metric_thresholds = active_thresholds.get(metric_name)
    if not metric_thresholds:
        return None

    metric_value = float(row["metric_value"])
    severity = None
    if metric_value >= metric_thresholds["critical"]:
        severity = "critical"
    elif metric_value >= metric_thresholds["warning"]:
        severity = "warning"

    if not severity:
        return None

    return {
        "model_key": row["model_key"],
        "feature_name": row["feature_name"],
        "metric_name": metric_name,
        "severity": severity,
        "status": "open",
        "metric_value": metric_value,
        "window_id": row.get("window_id", ""),
        "window_start": row.get("window_start", ""),
        "window_end": row["window_end"],
        "baseline_start": row.get("baseline_start", ""),
        "baseline_end": row.get("baseline_end", ""),
        "observed_at": _observed_at(row),
    }


def _incident_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["model_key"]), str(row["feature_name"]), str(row["metric_name"]))


def _incident_map(
    drift_rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    incidents: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in drift_rows:
        candidate = _incident_candidate(row, thresholds)
        if not candidate:
            continue
        key = _incident_key(candidate)
        existing = incidents.get(key)
        candidate_rank = _SEVERITY_RANK.get(str(candidate["severity"]), 0)
        existing_rank = _SEVERITY_RANK.get(str(existing["severity"]), 0) if existing else 0
        if (
            existing is None
            or candidate_rank > existing_rank
            or float(candidate["metric_value"]) > float(existing["metric_value"])
        ):
            incidents[key] = candidate
    return incidents


def build_incidents(drift_rows: list[dict[str, Any]], thresholds: dict | None = None) -> list[dict]:
    active_thresholds = merged_thresholds(thresholds)
    return list(_incident_map(drift_rows, active_thresholds).values())


def build_incident_history(
    drift_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    *,
    prior_open_incidents: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    thresholds: dict | None = None,
) -> list[dict[str, Any]]:
    active_thresholds = merged_thresholds(thresholds)
    rows_by_window: dict[str, list[dict[str, Any]]] = {}
    for row in drift_rows:
        rows_by_window.setdefault(str(row.get("window_id") or ""), []).append(row)

    previous = {key: dict(value) for key, value in (prior_open_incidents or {}).items()}
    history_rows: list[dict[str, Any]] = []

    ordered_windows = sorted(
        window_rows,
        key=lambda row: (
            str(row.get("window_end") or ""),
            str(row.get("window_start") or ""),
            str(row.get("window_id") or ""),
        ),
    )
    for window in ordered_windows:
        window_id = str(window.get("window_id") or "")
        current = _incident_map(rows_by_window.get(window_id, []), active_thresholds)
        observed_at = str(window.get("created_at") or "").strip() or datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")
        for key in sorted(set(previous) | set(current)):
            previous_row = previous.get(key)
            current_row = current.get(key)
            if current_row and not previous_row:
                event_type = "opened"
                event_row = current_row
                status = "open"
                metric_value = float(current_row["metric_value"])
                severity = str(current_row["severity"])
            elif current_row and previous_row:
                current_rank = _SEVERITY_RANK.get(str(current_row["severity"]), 0)
                previous_rank = _SEVERITY_RANK.get(str(previous_row["severity"]), 0)
                if current_rank > previous_rank:
                    event_type = "escalated"
                elif current_rank < previous_rank:
                    event_type = "downgraded"
                else:
                    event_type = "ongoing"
                event_row = current_row
                status = "open"
                metric_value = float(current_row["metric_value"])
                severity = str(current_row["severity"])
            elif previous_row and not current_row:
                event_type = "recovered"
                event_row = previous_row
                status = "closed"
                metric_value = 0.0
                severity = str(previous_row["severity"])
            else:
                continue

            history_rows.append(
                {
                    "model_key": key[0],
                    "feature_name": key[1],
                    "metric_name": key[2],
                    "event_type": event_type,
                    "severity": severity,
                    "status": status,
                    "metric_value": metric_value,
                    "window_id": window_id,
                    "window_start": str(
                        window.get("window_start") or event_row.get("window_start") or ""
                    ),
                    "window_end": str(
                        window.get("window_end") or event_row.get("window_end") or ""
                    ),
                    "baseline_start": str(
                        window.get("baseline_start") or event_row.get("baseline_start") or ""
                    ),
                    "baseline_end": str(
                        window.get("baseline_end") or event_row.get("baseline_end") or ""
                    ),
                    "observed_at": observed_at,
                }
            )
        previous = current
    return history_rows
