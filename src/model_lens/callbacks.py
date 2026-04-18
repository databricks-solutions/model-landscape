from __future__ import annotations

from dataclasses import replace
import logging
import os
import re
from datetime import datetime, timezone
from threading import Lock
from time import monotonic
from urllib.parse import parse_qs, urlencode

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html, ctx, no_update

from model_lens.backend import DashboardBackend, build_dashboard_backend
from model_lens.config import settings
from model_lens.domain.models import (
    DRIFT_CADENCE_PRESETS,
    MLflowLineage,
    MonitorConfig,
    PERFORMANCE_CADENCE_PRESETS,
)
from model_lens.domain.performance_metrics import (
    default_performance_metric_names,
    default_primary_performance_metric,
    performance_metric_label,
    performance_metric_options,
)
from model_lens.pages import onboarding
from model_lens.services.class_filters import normalize_class_filter, supports_binary_class_filters
from model_lens.services.inference_contracts import build_inference_contract
from model_lens.services.onboarding import baseline_label, build_default_baseline, build_fixed_baseline
from model_lens.services.refresh_jobs import (
    SCHEDULE_INTERVAL_OPTIONS,
    is_refresh_job_configuration_error,
    run_now_permission_guidance,
    trigger_refresh_job,
    update_shared_workflow_schedule,
    validate_workspace_readiness,
    workspace_readiness_payload,
)
from model_lens.services.thresholds import THRESHOLD_METRICS, get_thresholds, merged_thresholds
from model_lens.ui import charts
from model_lens.ui.components import (
    make_chart_card,
    make_empty_state,
    make_metric_card,
    make_model_status_card,
    make_wizard_step,
)


_NUMERIC_TYPE_TOKENS = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal", "numeric", "real")
_THRESHOLD_LABELS = {
    "psi": "PSI",
    "js_divergence": "Jensen-Shannon Divergence",
    "kl_divergence": "KL Divergence",
    "null_rate": "Null Rate (%)",
}
_SCHEDULE_INTERVAL_LABELS = {
    1: "Every 1 Hour",
    3: "Every 3 Hours",
    6: "Every 6 Hours",
    12: "Every 12 Hours",
    24: "Every 24 Hours",
}

logger = logging.getLogger(__name__)
_LAKEBASE_INSTANCE_CACHE_TTL_SECONDS = 60.0
_lakebase_instances_cache: tuple[float, tuple[str, ...]] | None = None
_lakebase_instances_cache_lock = Lock()

def _workspace_lakebase_instances() -> tuple[str, ...]:
    global _lakebase_instances_cache
    if not os.getenv("DATABRICKS_APP_PORT"):
        return ()
    now = monotonic()
    with _lakebase_instances_cache_lock:
        if _lakebase_instances_cache and _lakebase_instances_cache[0] > now:
            return _lakebase_instances_cache[1]
    try:
        from databricks.sdk import WorkspaceClient

        names: list[str] = []
        for instance in WorkspaceClient().database.list_database_instances(page_size=20):
            name = str(getattr(instance, "name", "") or "").strip()
            if name:
                names.append(name)
            if len(names) >= 5:
                break
        result = tuple(names)
    except Exception:
        result = ()
    with _lakebase_instances_cache_lock:
        _lakebase_instances_cache = (now + _LAKEBASE_INSTANCE_CACHE_TTL_SECONDS, result)
    return result


def _invalidate_workspace_lakebase_instances_cache() -> None:
    global _lakebase_instances_cache
    with _lakebase_instances_cache_lock:
        _lakebase_instances_cache = None


_workspace_lakebase_instances.cache_clear = _invalidate_workspace_lakebase_instances_cache  # type: ignore[attr-defined]


def _status_alert(message: str, color: str = "info") -> dbc.Alert:
    return dbc.Alert(message, color=color, className="py-2 mb-3")


def _user_action_error_message(action: str) -> str:
    return f"{action} failed. Check logs and try again."


def _callback_error_message(context: str, error: Exception) -> str:
    del error
    return f"Could not load {context}. Check logs and try again."


def _callback_error_panel(
    context: str,
    error: Exception,
    *,
    icon: str = "fas fa-triangle-exclamation",
) -> html.Div:
    logger.exception("Dashboard callback failed for %s", context, exc_info=error)
    return html.Div(
        [
            _status_alert(_callback_error_message(context, error), "danger"),
            make_empty_state(f"{context.capitalize()} is unavailable right now.", icon=icon),
        ]
    )


def _normalize_top_n(value: object, *, default: int = 10, minimum: int = 1, maximum: int = 50) -> int:
    try:
        numeric = int(value or default)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))


def _historical_drift_feature_ranking(
    drift: pd.DataFrame,
    *,
    metric: str,
    top_n: int,
) -> pd.DataFrame:
    if drift.empty or "feature" not in drift.columns or metric not in drift.columns:
        return pd.DataFrame(columns=["feature", metric])
    working = drift[["feature", metric]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce").fillna(0.0)
    return (
        working.groupby("feature", as_index=False)[metric]
        .max()
        .nlargest(top_n, metric)
        .reset_index(drop=True)
    )


def _format_metric_value(value: float | None, *, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def _resolved_monitor_thresholds(config: MonitorConfig | None) -> dict[str, dict[str, float]]:
    return merged_thresholds(getattr(config, "threshold_overrides", None))


def _threshold_input_id(metric: str, level: str, *, prefix: str = "reference") -> str:
    return f"{prefix}-threshold-{metric}-{level}-input"


def _coerce_threshold_input(metric: str, level: str, value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"{_THRESHOLD_LABELS.get(metric, metric)} {level.title()} is required.")
    numeric_value = float(numeric)
    if numeric_value < 0:
        raise ValueError(f"{_THRESHOLD_LABELS.get(metric, metric)} {level.title()} must be non-negative.")
    return numeric_value


def _collect_threshold_overrides_from_inputs(values: dict[str, tuple[object, object]]) -> dict[str, dict[str, float]]:
    overrides: dict[str, dict[str, float]] = {}
    for metric in THRESHOLD_METRICS:
        warning_raw, critical_raw = values[metric]
        warning_value = _coerce_threshold_input(metric, "warning", warning_raw)
        critical_value = _coerce_threshold_input(metric, "critical", critical_raw)
        if critical_value <= warning_value:
            raise ValueError(f"{_THRESHOLD_LABELS.get(metric, metric)} Critical must be greater than Warning.")
        default_warning, default_critical = get_thresholds(metric)
        if warning_value != default_warning or critical_value != default_critical:
            overrides[metric] = {"warning": warning_value, "critical": critical_value}
    return overrides


def _schedule_interval_options() -> list[dict[str, object]]:
    return [{"label": _SCHEDULE_INTERVAL_LABELS[value], "value": value} for value in SCHEDULE_INTERVAL_OPTIONS]


def _compute_guidance_block(
    *,
    config: MonitorConfig,
    diagnostics: dict[str, object] | None,
    shared_schedule: dict[str, object] | None,
) -> html.Div:
    payload = diagnostics or {}
    summary = payload.get("summary") or {}
    shared = shared_schedule or {}
    interval_hours = int(shared.get("current_interval_hours") or 1) if str(shared.get("current_interval_hours") or "").strip() else None
    footprint = str(summary.get("compute_footprint") or "No compute footprint data yet")
    footprint_category = str(summary.get("compute_footprint_category") or "no_data")
    median_duration = _format_duration_ms(summary.get("median_duration_ms"))
    recent_runs = int(summary.get("recent_run_count") or 0)
    alerts: list[object] = [
        dbc.Alert(
            (
                f"Shared workflow wake interval: {shared.get('current_label') or 'Unavailable'}. "
                f"This only decides how often the scheduler checks for due monitors. "
                f"Per-monitor cadence still decides whether {config.display_name} actually runs."
            ),
            color="secondary",
            className="py-2 mb-2",
        )
    ]
    if interval_hours is not None and interval_hours <= 3:
        alerts.append(
            dbc.Alert(
                f"A {interval_hours}-hour shared wake interval is aggressive. Large workspaces may see more queue pressure when many monitors are due together.",
                color="warning",
                className="py-2 mb-2",
            )
        )
    if config.drift_cadence_preset == "hourly":
        alerts.append(
            dbc.Alert(
                "This monitor checks drift and quality every hour when the shared workflow wakes up. Use hourly cadence only when low-latency detection is worth the additional Spark workload.",
                color="warning",
                className="py-2 mb-2",
            )
        )
    if config.performance_cadence_preset == "6h_3d_repair":
        alerts.append(
            dbc.Alert(
                "Performance repair is set to every 6 hours with a 3-day repair horizon. That is the heaviest labeled-monitor preset in the app.",
                color="warning",
                className="py-2 mb-2",
            )
        )
    if footprint_category in {"elevated", "high"}:
        alerts.append(
            dbc.Alert(
                f"Recent runs already show {footprint} compute footprint (median duration {median_duration}). Higher cadence will increase Spark time and queue contention on the shared workflow.",
                color="danger" if footprint_category == "high" else "warning",
                className="py-2 mb-2",
            )
        )
    elif recent_runs > 0:
        alerts.append(
            dbc.Alert(
                f"Recent runs indicate {footprint} compute footprint (median duration {median_duration}). Use this telemetry as the practical workload signal instead of estimating cost from rows alone.",
                color="info",
                className="py-2 mb-2",
            )
        )
    else:
        alerts.append(
            dbc.Alert(
                "No compute footprint data yet. After a few completed refresh runs, Model Lens will show duration and scanned-row guidance here.",
                color="secondary",
                className="py-2 mb-2",
            )
        )
    return html.Div(alerts, className="mb-3")


def _parse_custom_edges(value: object) -> list[float] | None:
    text = str(value or "").strip()
    if not text:
        return None
    edges: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        numeric = pd.to_numeric(pd.Series([token]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            raise ValueError(f"Invalid edge value: {token}")
        edges.append(float(numeric))
    if len(edges) < 2:
        raise ValueError("Enter at least two numeric edges.")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("Custom bin edges must be strictly increasing.")
    return edges


def _normalize_outlier_mode(value: object) -> str:
    normalized = str(value or "off").strip().lower()
    if normalized in {"percentile_clip", "iqr_fence"}:
        return normalized
    return "off"


def _outlier_control_value(mode: object, raw_value: object) -> float | None:
    normalized_mode = _normalize_outlier_mode(mode)
    if normalized_mode == "off":
        return None
    numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
    if normalized_mode == "percentile_clip":
        if pd.isna(numeric) or float(numeric) <= 0:
            return 1.0
        return min(49.0, float(numeric))
    if pd.isna(numeric) or float(numeric) <= 0:
        return 1.5
    return float(numeric)


def _analysis_filter_summary(
    *,
    start_date: str | None,
    end_date: str | None,
    class_basis: str | None,
    class_value: str | None,
) -> str:
    parts: list[str] = []
    if start_date or end_date:
        parts.append(f"Date range: {start_date or 'start'} to {end_date or 'end'}")
    basis = str(class_basis or "all").strip().lower()
    value = str(class_value or "all").strip().lower()
    if basis in {"actual", "predicted"} and value in {"positive", "negative"}:
        parts.append(f"{basis.title()} {value.title()}")
    return " | ".join(parts)


def _feature_distribution_source_message(source: str) -> str:
    if source == "persisted_histogram":
        return "Distribution source: persisted daily feature profiles (approximate histogram reconstruction)."
    if source == "persisted_samples":
        return "Distribution source: persisted daily feature profiles."
    if source == "bounded_window_read":
        return "Distribution source: bounded source-window read."
    if source == "unavailable_requested_raw":
        return "Requested exact distribution controls need a bounded source-window read, but that path is unavailable for this monitor right now."
    if source == "unavailable_unsafe_bounded_read":
        return "Distribution source unavailable. This repository cannot enforce a hard cap on raw source-window reads, so Model Lens skips the fallback instead of loading an unsafe volume of rows."
    return "Distribution source unavailable. Refresh more history or check bounded source-read support."


def _configured_run_now_permission_hint(*, workflow_kind: str = "shared") -> str:
    configured_job_id = ""
    if workflow_kind == "bootstrap":
        configured_job_id = str(getattr(settings, "bootstrap_refresh_job_id", "") or "").strip()
        if not configured_job_id:
            configured_job_id = str(getattr(settings, "refresh_job_id", "") or "").strip()
    else:
        configured_job_id = str(getattr(settings, "refresh_job_id", "") or "").strip()
    if configured_job_id.isdigit():
        guidance = run_now_permission_guidance(int(configured_job_id))
        if guidance:
            return f" {guidance}"
    return ""


def _setup_retry_message(error: object) -> str:
    del error
    return "Setup failed. Fix the issue and click Setup Control Plane again to retry."


def _refresh_job_unavailable_message(model_key: str, error: Exception) -> str:
    detail = (
        "If the shared refresh workflow already exists and is scheduled, it can still pick up this pending monitor on its next hourly run. "
        "If no shared refresh workflow exists yet, deploy or create it first, then set REFRESH_JOB_ID (preferred) or REFRESH_JOB_NAME in the app environment and redeploy the app."
    )
    if is_refresh_job_configuration_error(error):
        return (
            f"Saved monitor {model_key}. Initial refresh is pending on the shared refresh job; automatic trigger was unavailable. "
            f"{detail}"
        )
    return (
        f"Saved monitor {model_key}. Initial refresh is pending on the shared refresh job; automatic trigger was unavailable. "
        "The shared workflow can still pick it up on its next hourly run, or you can run it manually once job permissions are fixed."
        f"{_configured_run_now_permission_hint(workflow_kind='bootstrap')}"
    )


def _manual_refresh_unavailable_message(model_key: str, error: Exception) -> str:
    detail = (
        "If the shared refresh workflow already exists and is scheduled, it can still pick up this pending monitor on its next hourly run. "
        "If no shared refresh workflow exists yet, deploy or create it first, then set REFRESH_JOB_ID (preferred) or REFRESH_JOB_NAME in the app environment and redeploy the app."
    )
    if is_refresh_job_configuration_error(error):
        return (
            f"Could not trigger the initial refresh for {model_key}. "
            f"{detail}"
        )
    return (
        f"Could not trigger the initial refresh for {model_key}. "
        "The shared workflow can still pick it up on its next hourly run once job permissions are fixed."
        f"{_configured_run_now_permission_hint(workflow_kind='bootstrap')}"
    )


def _status_block(items: list[tuple[str, str]]) -> html.Div:
    return html.Div([_status_alert(message, color) for message, color in items])


def _comparison_history_message(window_count: int, granularity: str = "daily") -> str | None:
    if window_count >= 2:
        return None
    if granularity == "daily":
        return "Only one comparison window is available. Run more refreshes to see trends over time."
    return (
        f"Only one {granularity} comparison window is available. "
        "Run more refreshes to make granularity trends meaningful."
    )


def _default_performance_metric(model_id: str | None, backend: DashboardBackend) -> str:
    if not model_id:
        return default_primary_performance_metric("classification")
    config = backend.get_monitor_config(model_id)
    return _configured_default_performance_metric(config)


def _configured_performance_metric_names(config: object | None) -> list[str]:
    if config is None:
        return list(default_performance_metric_names("classification"))
    problem_type = getattr(config, "problem_type", "classification")
    allowed_values = {
        option["value"]
        for option in performance_metric_options(problem_type)
    }
    metric_names = [
        str(metric_name).strip().lower()
        for metric_name in (getattr(config, "performance_metric_names", ()) or ())
        if str(metric_name).strip().lower() in allowed_values
    ]
    if metric_names:
        return list(dict.fromkeys(metric_names))
    return list(default_performance_metric_names(problem_type))


def _configured_default_performance_metric(config: object | None) -> str:
    problem_type = getattr(config, "problem_type", "classification") if config is not None else "classification"
    metric_names = _configured_performance_metric_names(config)
    requested_default = str(getattr(config, "default_performance_metric", "") or "").strip().lower()
    if requested_default in metric_names:
        return requested_default
    preferred_default = default_primary_performance_metric(problem_type)
    if preferred_default in metric_names:
        return preferred_default
    return metric_names[0]


def _sync_performance_metric_selection(
    *,
    problem_type: str | None,
    selected_metrics: list[str] | tuple[str, ...] | None,
    current_default: str | None,
) -> tuple[list[dict[str, str]], list[str], list[dict[str, str]], str | None]:
    options = performance_metric_options(problem_type)
    allowed_values = {option["value"] for option in options}
    requested_metrics = [
        str(metric_name).strip().lower()
        for metric_name in (selected_metrics or [])
        if str(metric_name).strip().lower() in allowed_values
    ]
    if not requested_metrics:
        requested_metrics = list(default_performance_metric_names(problem_type))
    selected = list(dict.fromkeys(requested_metrics))
    default_options = [option for option in options if option["value"] in set(selected)]
    default_value = str(current_default or "").strip().lower()
    if default_value not in set(selected):
        preferred_default = default_primary_performance_metric(problem_type)
        default_value = preferred_default if preferred_default in set(selected) else selected[0]
    return options, selected, default_options, default_value


def _render_frame(frame: pd.DataFrame, empty_message: str, max_rows: int = 20) -> html.Div:
    if frame is None or frame.empty:
        return html.Div(empty_message, className="text-muted")
    rendered = frame.head(max_rows).copy()
    for column in rendered.columns:
        if pd.api.types.is_datetime64_any_dtype(rendered[column]):
            rendered[column] = rendered[column].astype(str)
    return html.Div(
        dbc.Table.from_dataframe(rendered, striped=True, bordered=True, hover=True, size="sm"),
        style={"overflowX": "auto"},
    )


def _format_runtime_setting_value(field: str, value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "(not configured)"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "(not configured)"
        return stripped
    return str(value)


def _format_duration_ms(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "—"
    milliseconds = float(numeric)
    if milliseconds >= 60_000:
        return f"{milliseconds / 60_000:.1f} min"
    if milliseconds >= 1_000:
        return f"{milliseconds / 1_000:.1f} s"
    return f"{int(milliseconds)} ms"


def _incident_options(data: dict | None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    payload = data or {}
    models = payload.get("models") or []
    open_incidents = payload.get("open_incidents")
    history = payload.get("history")
    model_options = [
        {
            "label": f"{row['name']} ({'Archived' if row.get('status') == 'inactive' else 'Active'})",
            "value": row["id"],
        }
        for row in models
    ]
    metric_values: set[str] = set()
    for frame in (open_incidents, history):
        if isinstance(frame, pd.DataFrame) and not frame.empty and "metric_name" in frame.columns:
            metric_values.update(
                str(value).strip()
                for value in frame["metric_name"].dropna().tolist()
                if str(value).strip()
            )
    metric_options = [{"label": metric, "value": metric} for metric in sorted(metric_values)]
    return model_options, metric_options


def _filter_incident_frame(
    frame: pd.DataFrame,
    *,
    model_id: str | None,
    severity: str | None,
    status: str | None,
    metric_name: str | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    filtered = frame.copy()
    if model_id and "model_key" in filtered.columns:
        filtered = filtered[filtered["model_key"] == model_id]
    severity_value = str(severity or "all").strip().lower()
    if severity_value and severity_value != "all" and "severity" in filtered.columns:
        filtered = filtered[filtered["severity"].astype(str).str.lower() == severity_value]
    status_value = str(status or "all").strip().lower()
    if status_value and status_value != "all" and "status" in filtered.columns:
        filtered = filtered[filtered["status"].astype(str).str.lower() == status_value]
    metric_value = str(metric_name or "").strip()
    if metric_value and "metric_name" in filtered.columns:
        filtered = filtered[filtered["metric_name"].astype(str) == metric_value]
    return filtered


def _render_refresh_diagnostics(diagnostics: dict | None) -> html.Div:
    payload = diagnostics or {}
    summary = payload.get("summary") or {}
    state = str(payload.get("state") or "no_runs").strip() or "no_runs"
    recent_runs = payload.get("recent_runs") or []
    if state == "no_runs":
        return html.Div("No refresh runs yet. No compute footprint data yet.", className="text-muted")

    recommendations = [str(item).strip() for item in summary.get("recommendations") or [] if str(item).strip()]
    recommendation_block = (
        html.Ul([html.Li(text) for text in recommendations], className="text-muted mb-3")
        if recommendations
        else html.Div()
    )
    state_alert = None
    if state == "insufficient_data":
        state_alert = dbc.Alert(
            "Not enough successful timed runs yet to diagnose bottlenecks reliably.",
            color="secondary",
            className="py-2 mb-3",
        )
    elif state == "failure_heavy":
        state_alert = dbc.Alert(
            "Recent failures limit timing guidance. Review recent error messages first.",
            color="warning",
            className="py-2 mb-3",
        )

    summary_cards = dbc.Row(
        [
            dbc.Col(
                make_metric_card(
                    "Recent Median Duration",
                    _format_duration_ms(summary.get("median_duration_ms")),
                    "Successful runs",
                ),
                md=3,
            ),
            dbc.Col(
                make_metric_card(
                    "Dominant Bottleneck",
                    str(summary.get("dominant_bottleneck") or "Insufficient Data"),
                    "Recent successful runs",
                    "warning" if state == "ready" else "secondary",
                ),
                md=3,
            ),
            dbc.Col(
                make_metric_card(
                    "Trend",
                    str(summary.get("trend") or "Not Enough History"),
                    "Recent vs previous runs",
                ),
                md=3,
            ),
            dbc.Col(
                make_metric_card(
                    "Success Rate",
                    f"{float(summary.get('success_rate_pct') or 0.0):.1f}%",
                    f"{int(summary.get('successful_run_count') or 0)}/{int(summary.get('recent_run_count') or 0)} recent runs",
                    "success" if float(summary.get("success_rate_pct") or 0.0) >= 80.0 else "warning",
                ),
                md=3,
            ),
            dbc.Col(
                make_metric_card(
                    "Compute Footprint",
                    str(summary.get("compute_footprint") or "No compute footprint data yet"),
                    "Advisory from median duration and scanned rows",
                    "warning" if str(summary.get("compute_footprint_category") or "") in {"elevated", "high"} else "secondary",
                ),
                md=3,
            ),
        ],
        className="g-3 mb-3",
    )
    recent_runs_frame = (
        pd.DataFrame(recent_runs)[
            [
                "started_at",
                "scope",
                "status",
                "total_duration_ms",
                "dominant_stage",
                "recommendation",
            ]
        ]
        if recent_runs
        else pd.DataFrame()
    )
    if not recent_runs_frame.empty:
        recent_runs_frame = recent_runs_frame.rename(
            columns={
                "started_at": "Started At",
                "scope": "Scope",
                "status": "Status",
                "total_duration_ms": "Total Duration",
                "dominant_stage": "Dominant Stage",
                "recommendation": "Recommendation",
            }
        )
        recent_runs_frame["Total Duration"] = recent_runs_frame["Total Duration"].apply(_format_duration_ms)
    return html.Div(
        [
            state_alert if state_alert is not None else html.Div(),
            summary_cards,
            html.H6("Recommendations", className="text-light mb-2"),
            recommendation_block if recommendations else html.Div("No diagnostics recommendations yet.", className="text-muted mb-3"),
            html.H6("Recent Diagnosed Runs", className="text-light mb-2"),
            _render_frame(recent_runs_frame, "No diagnosed runs yet.", max_rows=8),
        ]
    )


def _render_incidents_page(
    data: dict | None,
    *,
    model_id: str | None,
    severity: str | None,
    status: str | None,
    metric_name: str | None,
) -> html.Div:
    payload = data or {}
    open_incidents = payload.get("open_incidents")
    history = payload.get("history")
    open_filtered = _filter_incident_frame(
        open_incidents if isinstance(open_incidents, pd.DataFrame) else pd.DataFrame(),
        model_id=model_id,
        severity=severity,
        status=status,
        metric_name=metric_name,
    )
    history_filtered = _filter_incident_frame(
        history if isinstance(history, pd.DataFrame) else pd.DataFrame(),
        model_id=model_id,
        severity=severity,
        status=status,
        metric_name=metric_name,
    )
    if open_filtered.empty and history_filtered.empty:
        return make_empty_state("No incidents recorded yet.", icon="fas fa-triangle-exclamation")

    critical_open = (
        int(
            (
                open_filtered["severity"].astype(str).str.lower() == "critical"
            ).sum()
        )
        if not open_filtered.empty and "severity" in open_filtered.columns
        else 0
    )
    affected_monitors = (
        int(open_filtered["model_key"].astype(str).nunique())
        if not open_filtered.empty and "model_key" in open_filtered.columns
        else 0
    )
    summary_row = dbc.Row(
        [
            dbc.Col(make_metric_card("Open Incidents", str(len(open_filtered)), "Current open rows"), md=3),
            dbc.Col(make_metric_card("Critical", str(critical_open), "Open critical incidents", "danger" if critical_open else "secondary"), md=3),
            dbc.Col(make_metric_card("Affected Monitors", str(affected_monitors), "With open incidents"), md=3),
            dbc.Col(make_metric_card("Recent Events", str(len(history_filtered)), "Lifecycle rows in view"), md=3),
        ],
        className="g-3 mb-4",
    )

    open_frame = (
        open_filtered[
            [
                column
                for column in (
                    "display_name",
                    "model_key",
                    "feature_name",
                    "metric_name",
                    "severity",
                    "metric_value",
                    "window_end",
                    "observed_at",
                )
                if column in open_filtered.columns
            ]
        ]
        .rename(
            columns={
                "display_name": "Monitor",
                "model_key": "Model Key",
                "feature_name": "Feature",
                "metric_name": "Metric",
                "severity": "Severity",
                "metric_value": "Value",
                "window_end": "Window End",
                "observed_at": "Observed At",
            }
        )
        if not open_filtered.empty
        else pd.DataFrame()
    )
    history_frame = (
        history_filtered[
            [
                column
                for column in (
                    "display_name",
                    "model_key",
                    "event_type",
                    "feature_name",
                    "metric_name",
                    "severity",
                    "status",
                    "metric_value",
                    "window_end",
                    "observed_at",
                )
                if column in history_filtered.columns
            ]
        ]
        .rename(
            columns={
                "display_name": "Monitor",
                "model_key": "Model Key",
                "event_type": "Event",
                "feature_name": "Feature",
                "metric_name": "Metric",
                "severity": "Severity",
                "status": "Status",
                "metric_value": "Value",
                "window_end": "Window End",
                "observed_at": "Observed At",
            }
        )
        if not history_filtered.empty
        else pd.DataFrame()
    )
    return html.Div(
        [
            summary_row,
            html.H6("Open Incidents", className="text-light mb-2"),
            _render_frame(open_frame, "No open incidents for the current filters.", max_rows=20),
            html.Hr(),
            html.H6("Recent Incident History", className="text-light mb-2"),
            _render_frame(history_frame, "No incident history for the current filters.", max_rows=20),
        ]
    )


def _option_list(columns: list[str], include_blank: bool = False) -> list[dict]:
    options = [{"label": column, "value": column} for column in columns]
    if include_blank:
        return [{"label": "(none)", "value": ""}] + options
    return options


def _guess_column(columns: list[str], patterns: tuple[str, ...], *, fallback_first: bool = True) -> str:
    lowered = {column.lower(): column for column in columns}
    for pattern in patterns:
        for lower, original in lowered.items():
            if pattern in lower:
                return original
    if fallback_first and columns:
        return columns[0]
    return ""


def _guess_defaults(table_name: str, columns: list[str]) -> dict:
    timestamp_col = _guess_column(columns, ("event_ts", "timestamp", "datetime", "date", "time", "_ts"))
    model_id_col = _guess_column(columns, ("model_id", "model", "model_name"), fallback_first=False)
    prediction_col = _guess_column(columns, ("prediction", "score", "probability", "prob"))
    model_version_col = _guess_column(columns, ("model_version", "version"), fallback_first=False)
    prediction_score_col = _guess_column(columns, ("prediction_proba", "prediction_score", "probability", "score"), fallback_first=False)
    entity_id_col = _guess_column(columns, ("entity_id", "request_id", "user_id", "account_id", "id"), fallback_first=False)
    source_label_col = _guess_column(columns, ("label", "target", "actual"), fallback_first=False)
    reserved = {timestamp_col, model_id_col, prediction_col, model_version_col, prediction_score_col, source_label_col}
    features = [column for column in columns if column not in reserved and column != entity_id_col]
    categorical = [
        column for column in features
        if any(token in column.lower() for token in ("country", "segment", "region", "category", "type"))
    ]
    table_leaf = table_name.split(".")[-1] if table_name else "monitor"
    return {
        "display_name": table_leaf.replace("_", " ").title(),
        "model_key": re.sub(r"[^a-zA-Z0-9_]", "_", table_leaf).lower(),
        "timestamp_col": timestamp_col,
        "model_id_col": model_id_col,
        "prediction_col": prediction_col,
        "model_version_col": model_version_col,
        "prediction_score_col": prediction_score_col if prediction_score_col != prediction_col else "",
        "entity_id_col": entity_id_col,
        "source_label_col": source_label_col if source_label_col not in {timestamp_col, model_id_col, prediction_col} else "",
        "features": features,
        "categorical": [column for column in categorical if column in features],
        "slices": [column for column in categorical if column in features],
    }


def _discovery_defaults(scan_data: dict | None) -> dict | None:
    if not scan_data:
        return None
    discovery = scan_data.get("discovery")
    return discovery if isinstance(discovery, dict) and discovery else None


def _schema_frame(scan_data: dict | None) -> pd.DataFrame:
    if not scan_data or not scan_data.get("schema"):
        return pd.DataFrame(columns=["col_name", "data_type"])
    return pd.DataFrame(scan_data["schema"])


def _schema_types(scan_data: dict | None) -> dict[str, str]:
    frame = _schema_frame(scan_data)
    if frame.empty or "col_name" not in frame.columns:
        return {}
    return {str(row["col_name"]): str(row.get("data_type") or "") for _, row in frame.iterrows()}


def _non_numeric_features(feature_columns: list[str] | None, scan_data: dict | None) -> list[str]:
    if not feature_columns:
        return []
    types = _schema_types(scan_data)
    non_numeric: list[str] = []
    for column in feature_columns:
        data_type = types.get(column, "").lower()
        if data_type and not any(token in data_type for token in _NUMERIC_TYPE_TOKENS):
            non_numeric.append(column)
    return non_numeric


def _feature_candidates(scan_data: dict | None, reserved_columns: list[str | None]) -> list[str]:
    if not scan_data:
        return []
    reserved = {column for column in reserved_columns if column}
    return [column for column in scan_data.get("columns", []) if column not in reserved]


def _session_config(data: dict | None) -> dict:
    payload = data or {}
    return {
        "control_plane_catalog": (payload.get("control_plane_catalog") or settings.control_plane_catalog).strip(),
        "control_plane_schema": (payload.get("control_plane_schema") or settings.control_plane_schema).strip(),
        "lakebase_instance_name": (payload.get("lakebase_instance_name") or settings.lakebase_instance_name).strip(),
        "lakebase_database_name": (payload.get("lakebase_database_name") or settings.lakebase_database_name).strip(),
        "lakebase_schema": (payload.get("lakebase_schema") or settings.lakebase_schema).strip(),
    }


def _control_plane_ready(
    ready_state: dict | None,
    *,
    control_plane_catalog: str | None,
    control_plane_schema: str | None,
    lakebase_instance_name: str | None,
    lakebase_database_name: str | None,
    lakebase_schema: str | None,
) -> bool:
    if not ready_state:
        return False
    requested_lakebase = bool((lakebase_instance_name or "").strip() or (lakebase_database_name or "").strip())
    recorded_catalog = str(ready_state.get("control_plane_catalog") or "").strip()
    recorded_schema = str(ready_state.get("control_plane_schema") or "").strip()
    recorded_lakebase_instance = str(ready_state.get("lakebase_instance_name") or "").strip()
    recorded_lakebase_database = str(ready_state.get("lakebase_database_name") or "").strip()
    recorded_lakebase_schema = str(ready_state.get("lakebase_schema") or "").strip()
    return bool(
        recorded_catalog
        and recorded_schema
        and recorded_catalog == str(control_plane_catalog or "").strip()
        and recorded_schema == str(control_plane_schema or "").strip()
        and (
            not requested_lakebase
            or (
                recorded_lakebase_instance == str(lakebase_instance_name or "").strip()
                and recorded_lakebase_database == str(lakebase_database_name or "").strip()
                and recorded_lakebase_schema == str(lakebase_schema or "").strip()
            )
        )
    )


def _workspace_readiness_mode(readiness_state: dict | None) -> str:
    return str((readiness_state or {}).get("overall_mode") or "not_ready").strip().lower() or "not_ready"


def _workspace_readiness_for_session(ready_state: dict | None, session_data: dict | None) -> dict[str, object]:
    session = _session_config(session_data)
    readiness = validate_workspace_readiness(
        control_plane_ready=_ready_for_session(ready_state, session),
        lakebase_requested=bool(session["lakebase_instance_name"] or session["lakebase_database_name"]),
    )
    payload = workspace_readiness_payload(readiness)
    payload.update(
        {
            "control_plane_catalog": session["control_plane_catalog"],
            "control_plane_schema": session["control_plane_schema"],
            "lakebase_instance_name": session["lakebase_instance_name"],
            "lakebase_database_name": session["lakebase_database_name"],
            "lakebase_schema": session["lakebase_schema"],
        }
    )
    return payload


def _render_workspace_readiness(readiness_state: dict | None) -> html.Div:
    readiness = readiness_state or {}
    mode = _workspace_readiness_mode(readiness)
    mode_message = {
        "fully_ready": "Workspace readiness is green. The app can save a monitor and trigger bootstrap immediately.",
        "scheduler_only": "Workspace readiness is good enough for onboarding. The app will save monitors and the scheduled shared workflow will pick them up.",
        "not_ready": "Workspace readiness is blocked. Fix the missing wiring below before onboarding a monitor.",
    }.get(mode, "Workspace readiness is blocked. Fix the missing wiring below before onboarding a monitor.")
    mode_color = {
        "fully_ready": "success",
        "scheduler_only": "secondary",
        "not_ready": "danger",
    }.get(mode, "danger")

    configured_via = str(readiness.get("refresh_workflow_configured_via") or "none").strip() or "none"
    configured_value = str(readiness.get("refresh_workflow_configured_value") or "").strip()
    if configured_via == "id":
        wiring_text = f"REFRESH_JOB_ID={configured_value or '(unset)'}"
    elif configured_via == "name":
        wiring_text = f"REFRESH_JOB_NAME={configured_value or '(unset)'}"
    else:
        wiring_text = "(not configured)"

    resolved_name = str(readiness.get("refresh_workflow_name") or "").strip()
    resolved_id = readiness.get("refresh_workflow_id")
    resolved_text = (
        f"{resolved_name or '(unnamed workflow)'} (job_id={resolved_id})"
        if resolved_id is not None
        else "Not resolved"
    )

    bootstrap_mode = str(readiness.get("bootstrap_workflow_mode") or "shared_default").strip() or "shared_default"
    bootstrap_resolved = bool(readiness.get("bootstrap_workflow_resolved"))
    bootstrap_name = str(readiness.get("bootstrap_workflow_name") or "").strip()
    bootstrap_id = readiness.get("bootstrap_workflow_id")
    if bootstrap_mode == "shared_default":
        bootstrap_text = "Uses shared refresh workflow (default)"
    elif bootstrap_resolved and bootstrap_id is not None:
        bootstrap_text = f"{bootstrap_name or '(unnamed workflow)'} (job_id={bootstrap_id})"
    elif bootstrap_resolved:
        bootstrap_text = bootstrap_name or "Ready"
    else:
        bootstrap_configured_via = str(readiness.get("bootstrap_workflow_configured_via") or "none").strip() or "none"
        bootstrap_configured_value = str(readiness.get("bootstrap_workflow_configured_value") or "").strip()
        if bootstrap_configured_via == "id":
            bootstrap_text = f"Configured via BOOTSTRAP_REFRESH_JOB_ID={bootstrap_configured_value or '(unset)'} but not resolved"
        elif bootstrap_configured_via == "name":
            bootstrap_text = f"Configured via BOOTSTRAP_REFRESH_JOB_NAME={bootstrap_configured_value or '(unset)'} but not resolved"
        else:
            bootstrap_text = "Not configured"

    scheduled_text = "Available"
    if not readiness.get("scheduler_path_available"):
        scheduled_text = "Unavailable"
    elif str(readiness.get("scheduler_mode") or "").strip():
        scheduled_text = f"Available ({str(readiness.get('scheduler_mode')).strip()})"

    run_now_available = readiness.get("run_now_available")
    if run_now_available is True:
        immediate_text = "Available"
    elif run_now_available is False and resolved_id is not None:
        immediate_text = f"Grant CAN_MANAGE_RUN on job {resolved_id}"
    elif run_now_available is None and readiness.get("scheduler_path_available"):
        immediate_text = "Verification unavailable"
    elif readiness.get("scheduler_path_available"):
        immediate_text = "Scheduler only"
    else:
        immediate_text = "Unavailable"

    bootstrap_run_now_available = readiness.get("bootstrap_run_now_available")
    if bootstrap_mode == "shared_default":
        bootstrap_immediate_text = "Uses shared refresh workflow permissions"
    elif bootstrap_run_now_available is True:
        bootstrap_immediate_text = "Available"
    elif bootstrap_run_now_available is False and bootstrap_id is not None:
        bootstrap_immediate_text = f"Grant CAN_MANAGE_RUN on job {bootstrap_id}"
    elif bootstrap_run_now_available is None and bootstrap_id is not None:
        bootstrap_immediate_text = "Verification unavailable"
    else:
        bootstrap_immediate_text = "Unavailable"

    lakebase_text = "Warehouse-only"
    if readiness.get("lakebase_ready") is False:
        lakebase_text = "Incomplete optional config"
    elif settings.use_lakebase_read_model:
        lakebase_text = "Configured"

    checks = pd.DataFrame(
        [
            {"check": "Control Plane", "status": "Ready" if readiness.get("control_plane_ready") else "Blocked"},
            {"check": "SQL Warehouse", "status": "Ready" if readiness.get("warehouse_ready") else "Missing"},
            {
                "check": "Shared Refresh Workflow",
                "status": "Ready" if readiness.get("refresh_workflow_resolved") else "Missing",
            },
            {"check": "App Workflow Wiring", "status": wiring_text},
            {"check": "Resolved Workflow", "status": resolved_text},
            {"check": "Optional Bootstrap Lane", "status": bootstrap_text},
            {"check": "Scheduled Bootstrap", "status": scheduled_text},
            {"check": "Immediate Bootstrap", "status": immediate_text},
            {"check": "Bootstrap Trigger", "status": bootstrap_immediate_text},
            {"check": "Optional Lakebase", "status": lakebase_text},
        ]
    )

    blocking_issues = [str(item) for item in readiness.get("blocking_issues", []) if str(item).strip()]
    warnings = [str(item) for item in readiness.get("warnings", []) if str(item).strip()]
    issue_block = html.Div(
        [
            *[
                dbc.Alert(message, color="danger", className="py-2 mb-2")
                for message in blocking_issues
            ],
            *[
                dbc.Alert(message, color="warning", className="py-2 mb-2")
                for message in warnings
            ],
        ]
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("Workspace Readiness", className="mb-3"),
                dbc.Alert(mode_message, color=mode_color, className="py-2 mb-3"),
                _render_frame(checks, "No readiness checks available.", max_rows=14),
                html.Div(issue_block, className="mt-3"),
            ]
        ),
        className="border-secondary",
    )


def _selected_model_from_search(search: str | None) -> str | None:
    if not search:
        return None
    parsed = parse_qs(search.lstrip("?"))
    values = parsed.get("model", [])
    if not values:
        return None
    selected = str(values[0]).strip()
    return selected or None


def _selected_feature_from_search(search: str | None) -> str | None:
    if not search:
        return None
    parsed = parse_qs(search.lstrip("?"))
    values = parsed.get("feature", [])
    if not values:
        return None
    selected = str(values[0]).strip()
    return selected or None


def _monitor_status_text(status: str | None) -> str:
    return "Archived" if str(status or "").strip().lower() == "inactive" else "Active"


def _monitor_option_label(display_name: str | None, model_key: str | None, *, status: str | None = None) -> str:
    name_text = str(display_name or model_key or "Unnamed Monitor").strip()
    key_text = str(model_key or "").strip()
    if not key_text:
        return name_text
    suffix = f", {_monitor_status_text(status)}" if status is not None else ""
    return f"{name_text} ({key_text}{suffix})"


def _list_all_monitor_configs(backend: DashboardBackend) -> list[MonitorConfig]:
    repository = getattr(backend, "repository", None)
    if repository and hasattr(repository, "list_monitor_configs"):
        try:
            return list(repository.list_monitor_configs(status=None))
        except TypeError:
            return list(repository.list_monitor_configs())
    return []


def _existing_monitor_keys(backend: DashboardBackend) -> set[str]:
    configs = _list_all_monitor_configs(backend)
    if configs:
        return {
            str(config.model_key).strip()
            for config in configs
            if str(getattr(config, "model_key", "")).strip()
        }
    list_reference_models = getattr(backend, "list_reference_models", None)
    if callable(list_reference_models):
        try:
            models = list_reference_models(status=None)
        except TypeError:
            models = list_reference_models()
        return {
            str(model.get("id") or "").strip()
            for model in models
            if str(model.get("id") or "").strip()
        }
    return set()


def _suggest_unique_model_key(base_key: str | None, existing_keys: set[str]) -> str:
    candidate = str(base_key or "").strip()
    if not candidate:
        return ""
    if candidate not in existing_keys:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing_keys:
        suffix += 1
    return f"{candidate}_{suffix}"


def _resolve_reference_model_id(global_model_id: str | None, reference_model_id: str | None) -> str | None:
    selected = str(reference_model_id or "").strip()
    if selected:
        return selected
    selected = str(global_model_id or "").strip()
    return selected or None


def _get_monitor_config_for_reference(backend: DashboardBackend, model_id: str) -> MonitorConfig | None:
    try:
        return backend.get_monitor_config(model_id, status=None)
    except TypeError:
        return backend.get_monitor_config(model_id)
    except AttributeError:
        return None


def _resolve_monitor_config_by_key(
    backend: DashboardBackend,
    model_key: str,
    *,
    fail_if_ambiguous: bool = False,
) -> tuple[MonitorConfig | None, str | None]:
    normalized_key = str(model_key or "").strip()
    if not normalized_key:
        return None, "Select a monitor before continuing."
    configs = _list_all_monitor_configs(backend)
    if configs:
        matches = [
            config
            for config in configs
            if str(getattr(config, "model_key", "")).strip() == normalized_key
        ]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            if fail_if_ambiguous:
                return None, f"Multiple monitors currently share model key {normalized_key}. Resolve that collision before using lifecycle actions."
            return matches[0], None
        return None, f"Selected monitor {normalized_key} no longer exists."
    config = _get_monitor_config_for_reference(backend, normalized_key)
    if config:
        return config, None
    return None, f"Selected monitor {normalized_key} no longer exists."


def _make_backend(session_data: dict | None) -> DashboardBackend:
    session = _session_config(session_data)
    return build_dashboard_backend(
        catalog=session["control_plane_catalog"],
        schema=session["control_plane_schema"],
        lakebase_instance_name=session["lakebase_instance_name"] or None,
        lakebase_database_name=session["lakebase_database_name"] or None,
        lakebase_schema=session["lakebase_schema"] or None,
    )


def _ready_for_session(ready_state: dict | None, session_data: dict | None) -> bool:
    if not ready_state:
        return False
    session = _session_config(session_data)
    recorded_catalog = str(ready_state.get("control_plane_catalog") or "").strip()
    recorded_schema = str(ready_state.get("control_plane_schema") or "").strip()
    recorded_lakebase_instance = str(ready_state.get("lakebase_instance_name") or "").strip()
    recorded_lakebase_database = str(ready_state.get("lakebase_database_name") or "").strip()
    recorded_lakebase_schema = str(ready_state.get("lakebase_schema") or "").strip()
    requested_lakebase = bool(session["lakebase_instance_name"] or session["lakebase_database_name"])
    return bool(
        recorded_catalog
        and recorded_schema
        and recorded_catalog == session["control_plane_catalog"]
        and recorded_schema == session["control_plane_schema"]
        and (
            not requested_lakebase
            or (
                recorded_lakebase_instance == session["lakebase_instance_name"]
                and recorded_lakebase_database == session["lakebase_database_name"]
                and recorded_lakebase_schema == session["lakebase_schema"]
            )
        )
    )


def _step_style(is_active: bool) -> dict:
    return {} if is_active else {"display": "none"}


def _source_labels_join_col(scan_data: dict | None, entity_id_col: str | None, labels_join_col: str | None) -> str | None:
    if not isinstance(scan_data, dict):
        return None
    columns = {str(column).strip() for column in scan_data.get("columns", []) if str(column).strip()}
    shared_join_col = (labels_join_col or "").strip()
    if shared_join_col and shared_join_col in columns:
        return shared_join_col
    entity_join_col = (entity_id_col or "").strip()
    if entity_join_col and entity_join_col in columns:
        return entity_join_col
    return None


def _monitor_contract_ready(
    *,
    scan_data: dict | None,
    display_name: str | None,
    model_key: str | None,
    timestamp_col: str | None,
    model_id_col: str | None,
    prediction_col: str | None,
    model_id_value: str | None = None,
    model_version_col: str | None,
    model_version_value: str | None,
    entity_id_col: str | None,
    source_label_col: str | None,
    external_label_col: str | None,
    labels_table: str | None,
    labels_join_col: str | None,
    feature_columns: list[str] | None,
    baseline_kind: str | None,
    baseline_days: int | None,
    baseline_start: str | None,
    baseline_end: str | None,
) -> bool:
    if not scan_data or not scan_data.get("columns"):
        return False
    if not all(
        [
            (display_name or "").strip(),
            (model_key or "").strip(),
            timestamp_col,
            prediction_col,
            feature_columns,
        ]
    ):
        return False
    if (model_id_value or "").strip() and not model_id_col:
        return False
    try:
        if (baseline_kind or "rolling") == "fixed":
            build_fixed_baseline((baseline_start or "").strip(), (baseline_end or "").strip())
        else:
            build_default_baseline(int(baseline_days or 7))
    except (TypeError, ValueError):
        return False
    if (model_version_value or "").strip() and not model_version_col:
        return False
    if (labels_table or "").strip():
        return bool(
            _source_labels_join_col(scan_data, entity_id_col, labels_join_col)
            and (labels_join_col or "").strip()
            and ((external_label_col or "").strip() or (source_label_col or "").strip())
        )
    return True


def _review_summary(
    *,
    control_plane_catalog: str | None,
    control_plane_schema: str | None,
    source_table: str | None,
    display_name: str | None,
    model_key: str | None,
    model_id_col: str | None,
    feature_columns: list[str] | None,
    categorical_columns: list[str] | None,
    slice_columns: list[str] | None,
    model_id_value: str | None,
    model_version_value: str | None,
    labels_table: str | None,
    source_label_col: str | None,
    external_label_col: str | None,
    baseline_kind: str | None,
    baseline_days: int | None,
    baseline_start: str | None,
    baseline_end: str | None,
    problem_type: str | None,
    performance_metric_names: list[str] | tuple[str, ...] | None,
    default_performance_metric: str | None,
    drift_cadence_preset: str | None,
    performance_cadence_preset: str | None,
    schedule_enabled: bool,
    lakebase_instance_name: str | None,
    lakebase_database_name: str | None,
    mlflow_experiment_name: str | None,
    mlflow_registered_model_name: str | None,
) -> html.Div:
    label_source = str(labels_table or "").strip() if str(labels_table or "").strip() else (str(source_label_col or "").strip() or "none")
    problem_type_text = str(problem_type or "classification").strip() or "classification"
    performance_metric_text = ", ".join(performance_metric_label(metric_name) for metric_name in (performance_metric_names or ())) or "Default"
    default_performance_metric_text = performance_metric_label(default_performance_metric or default_primary_performance_metric(problem_type_text))
    drift_cadence_text = str(drift_cadence_preset or "6h").strip() or "6h"
    performance_cadence_text = str(performance_cadence_preset or "disabled").strip() or "disabled"
    baseline_policy = (
        build_fixed_baseline(str(baseline_start or "").strip(), str(baseline_end or "").strip())
        if str(baseline_kind or "rolling").strip() == "fixed" and str(baseline_start or "").strip() and str(baseline_end or "").strip()
        else build_default_baseline(int(baseline_days or 7))
    )
    rows = [
        ("Control Plane", f"{(control_plane_catalog or '').strip()}.{(control_plane_schema or '').strip()}"),
        ("Source Table", (source_table or "").strip() or "Not scanned yet"),
        ("Display Name", (display_name or "").strip() or "Not set"),
        ("Model Key", (model_key or "").strip() or "Not set"),
        ("Problem Type", problem_type_text.title()),
        ("Tracked Performance Metrics", performance_metric_text),
        ("Default Performance Metric", default_performance_metric_text),
        ("Baseline Policy", baseline_label(baseline_policy)),
        ("Scheduled Refreshes", "Enabled" if schedule_enabled else "Manual only"),
        ("Drift Cadence", drift_cadence_text),
        ("Performance Cadence", performance_cadence_text),
        ("Feature Columns", str(len(feature_columns or []))),
        ("Categorical Columns", str(len(categorical_columns or []))),
        ("Slice Columns", str(len(slice_columns or []))),
        (
            "Model Scope",
            (model_id_value or "").strip()
            or ("Table-scoped monitor" if not str(model_id_col or "").strip() else "All model_id values"),
        ),
        ("Version Scope", (model_version_value or "").strip() or "All versions"),
        ("Labels", label_source),
        (
            "MLflow",
            (mlflow_registered_model_name or "").strip()
            or (mlflow_experiment_name or "").strip()
            or "Not linked",
        ),
        (
            "Lakebase Session",
            f"{(lakebase_instance_name or '').strip()} / {(lakebase_database_name or '').strip()}"
            if (lakebase_instance_name or "").strip() or (lakebase_database_name or "").strip()
            else "Warehouse-only",
        ),
    ]
    frame = pd.DataFrame(rows, columns=["setting", "value"])
    return _render_frame(frame, empty_message="")


def _render_labels_discovery(
    *,
    labels_table: str,
    label_schema: pd.DataFrame,
    label_preview: pd.DataFrame,
    label_validation: dict | None,
    join_col: str,
    label_col: str,
    order_col: str,
) -> html.Div:
    validation = label_validation or {}
    matched_rows = int(validation.get("matched_rows", 0) or 0)
    inference_rows = int(validation.get("inference_rows", 0) or 0)
    unmatched_rows = int(validation.get("unmatched_rows", 0) or 0)
    duplicate_join_keys = int(validation.get("duplicate_join_keys", 0) or 0)
    alerts: list[dbc.Alert] = []
    if inference_rows > 0 and matched_rows == 0:
        alerts.append(
            dbc.Alert(
                "No rows matched between inference and labels tables on this join column. Check the join column selection.",
                color="danger",
                className="mb-3",
            )
        )
    elif unmatched_rows > 0:
        alerts.append(
            dbc.Alert(
                f"Labels matched {matched_rows} of {inference_rows} inference rows; {unmatched_rows} remain unmatched.",
                color="warning",
                className="mb-3",
            )
        )
    if duplicate_join_keys > 0 and not order_col:
        alerts.append(
            dbc.Alert(
                "Labels table contains duplicate join keys without an order column. Add an order column so Model Lens can choose the latest label per entity.",
                color="warning",
                className="mb-3",
            )
        )
    mapping_frame = pd.DataFrame(
        [
            {"field": "join_column", "value": join_col or "Not detected"},
            {"field": "label_column", "value": label_col or "Not detected"},
            {"field": "order_column", "value": order_col or "Not detected"},
        ]
    )
    validation_frame = pd.DataFrame(
        [
            {"check": "matched_rows", "value": int(validation.get("matched_rows", 0) or 0)},
            {"check": "unmatched_rows", "value": int(validation.get("unmatched_rows", 0) or 0)},
            {"check": "match_rate_pct", "value": float(validation.get("match_rate_pct", 0.0) or 0.0)},
            {"check": "duplicate_join_keys", "value": int(validation.get("duplicate_join_keys", 0) or 0)},
            {
                "check": "distinct_label_values",
                "value": ", ".join(validation.get("distinct_label_values", ())) or "(none)",
            },
            {
                "check": "binary_compatible",
                "value": "yes" if validation.get("binary_compatible") else "no",
            },
        ]
    )
    return html.Div(
        [
            html.Hr(),
            html.H6(f"Labels Table Preview: {labels_table}", className="mb-2"),
            html.P(
                "Model Lens scanned the labels table, inferred the join and label columns, and validated the join against the source table.",
                className="text-muted",
            ),
            *alerts,
            html.H6("Detected Label Mapping", className="mb-2"),
            _render_frame(mapping_frame, "No label mapping detected."),
            html.Hr(),
            html.H6("Join Validation", className="mb-2"),
            _render_frame(validation_frame, "No join validation available."),
            html.Hr(),
            html.H6("Labels Schema", className="mb-2"),
            _render_frame(label_schema, "No label schema metadata returned."),
            html.Hr(),
            html.H6("Labels Sample Rows", className="mb-2"),
            _render_frame(label_preview, "No labels preview rows returned.", max_rows=5),
        ]
    )


def _deployment_mode_prompt(session_data: dict | None):
    session = _session_config(session_data)
    if session["lakebase_database_name"] or session["lakebase_instance_name"]:
        return dbc.Badge(
            "Mode: Lakebase",
            color="success",
            pill=True,
            className="px-3 py-2",
        )
    return dbc.Badge(
        "Mode: Warehouse",
        color="secondary",
        pill=True,
        className="px-3 py-2",
    )


def _model_banner(model_id: str | None, backend: DashboardBackend):
    if not model_id:
        return html.Div()
    model = backend.get_model_map().get(model_id)
    if not model:
        return html.Div()
    return dbc.Alert(
        [
            html.I(className="fas fa-robot me-2"),
            html.Strong(model["name"]),
            html.Span(f" — {model.get('description', '')[:120]}", className="ms-1 text-muted") if model.get("description") else None,
        ],
        color="dark",
        className="py-2 mb-3 border",
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("onboarding-current-step", "data"),
        Input("wizard-back-btn", "n_clicks"),
        Input("wizard-next-btn", "n_clicks"),
        State("onboarding-current-step", "data"),
        prevent_initial_call=True,
    )
    def navigate_onboarding_wizard(_, __, current_step):
        try:
            step = int(current_step or 1)
        except (TypeError, ValueError):
            step = 1
        if ctx.triggered_id == "wizard-back-btn":
            return max(1, step - 1)
        return min(len(onboarding.STEP_LABELS), step + 1)

    @app.callback(
        Output("wizard-steps-indicator", "children"),
        Output("wizard-step-guidance", "children"),
        Output("wizard-step-workspace", "style"),
        Output("wizard-step-source", "style"),
        Output("wizard-step-contract", "style"),
        Output("wizard-step-review", "style"),
        Output("wizard-back-btn", "style"),
        Output("wizard-next-btn", "style"),
        Output("wizard-next-btn", "disabled"),
        Output("wizard-next-btn", "children"),
        Output("save-monitor-btn", "disabled"),
        Output("onboarding-review-summary", "children"),
        Input("onboarding-current-step", "data"),
        Input("control-plane-ready-store", "data"),
        Input("workspace-readiness-store", "data"),
        Input("control-plane-catalog-input", "value"),
        Input("control-plane-schema-input", "value"),
        Input("source-table-input", "value"),
        Input("scan-data", "data"),
        Input("display-name-input", "value"),
        Input("model-key-input", "value"),
        Input("timestamp-col-dropdown", "value"),
        Input("model-id-col-dropdown", "value"),
        Input("prediction-col-dropdown", "value"),
        Input("model-version-col-dropdown", "value"),
        Input("model-id-value-input", "value"),
        Input("model-version-value-input", "value"),
        Input("entity-id-col-dropdown", "value"),
        Input("source-label-col-dropdown", "value"),
        Input("labels-table-input", "value"),
        Input("labels-join-col-input", "value"),
        Input("external-label-col-input", "value"),
        Input("feature-cols-dropdown", "value"),
        Input("categorical-cols-dropdown", "value"),
        Input("slice-cols-dropdown", "value"),
        Input("problem-type-dropdown", "value"),
        Input("baseline-kind-input", "value"),
        Input("baseline-days-input", "value"),
        Input("baseline-fixed-range-input", "start_date"),
        Input("baseline-fixed-range-input", "end_date"),
        Input("review-drift-cadence-select", "value"),
        Input("review-performance-cadence-select", "value"),
        Input("review-schedule-enabled-toggle", "value"),
        Input("review-performance-metrics-dropdown", "value"),
        Input("review-default-performance-metric-select", "value"),
        Input("mlflow-experiment-input", "value"),
        Input("mlflow-registered-model-input", "value"),
        Input("lakebase-instance-input", "value"),
        Input("lakebase-database-input", "value"),
        Input("lakebase-schema-input", "value"),
    )
    def render_onboarding_wizard(
        current_step,
        control_plane_ready_state,
        workspace_readiness_state,
        control_plane_catalog,
        control_plane_schema,
        source_table,
        scan_data,
        display_name,
        model_key,
        timestamp_col,
        model_id_col,
        prediction_col,
        model_version_col,
        model_id_value,
        model_version_value,
        entity_id_col,
        source_label_col,
        labels_table,
        labels_join_col,
        external_label_col,
        feature_columns,
        categorical_columns,
        slice_columns,
        problem_type,
        baseline_kind,
        baseline_days,
        baseline_start,
        baseline_end,
        review_drift_cadence,
        review_performance_cadence,
        review_schedule_enabled,
        review_performance_metrics,
        review_default_performance_metric,
        mlflow_experiment_name,
        mlflow_registered_model_name,
        lakebase_instance_name,
        lakebase_database_name,
        lakebase_schema,
    ):
        try:
            step = max(1, min(int(current_step or 1), len(onboarding.STEP_LABELS)))
            control_plane_ready = _control_plane_ready(
                control_plane_ready_state,
                control_plane_catalog=control_plane_catalog,
                control_plane_schema=control_plane_schema,
                lakebase_instance_name=lakebase_instance_name,
                lakebase_database_name=lakebase_database_name,
                lakebase_schema=lakebase_schema,
            )
            readiness_mode = _workspace_readiness_mode(workspace_readiness_state)
            workspace_ready = control_plane_ready and readiness_mode in {"scheduler_only", "fully_ready"}
            source_ready = bool(scan_data and scan_data.get("columns"))
            contract_ready = _monitor_contract_ready(
                scan_data=scan_data,
                display_name=display_name,
                model_key=model_key,
                timestamp_col=timestamp_col,
                model_id_col=model_id_col,
                prediction_col=prediction_col,
                model_id_value=model_id_value,
                model_version_col=model_version_col,
                model_version_value=model_version_value,
                entity_id_col=entity_id_col,
                source_label_col=source_label_col,
                external_label_col=external_label_col,
                labels_table=labels_table,
                labels_join_col=labels_join_col,
                feature_columns=feature_columns,
                baseline_kind=baseline_kind,
                baseline_days=baseline_days,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
            )
            next_disabled = {
                1: not workspace_ready,
                2: not source_ready,
                3: not contract_ready,
            }.get(step, False)
            readiness_issues = [str(item) for item in (workspace_readiness_state or {}).get("blocking_issues", []) if str(item).strip()]
            primary_readiness_issue = readiness_issues[0] if readiness_issues else ""
            guidance = {
                1: (
                    (
                        "Workspace setup is fully ready. The app can save a monitor and trigger bootstrap immediately."
                        if readiness_mode == "fully_ready"
                        else (
                            "Workspace setup is ready in scheduler-only mode. The app can save monitors, and the scheduled shared refresh workflow will pick them up."
                            if readiness_mode == "scheduler_only"
                            else (
                                primary_readiness_issue
                                or "Confirm the control-plane namespace, verify the warehouse and shared refresh workflow, and use Validate Workspace Wiring before onboarding a monitor."
                            )
                        )
                    ),
                    "success" if readiness_mode == "fully_ready" else ("info" if readiness_mode == "scheduler_only" else "secondary"),
                ),
                2: (
                    "Enter the inference table and click Discover. Optional labels and MLflow inputs help Model Lens infer a better draft.",
                    "success" if source_ready else "secondary",
                ),
                3: (
                    "Confirm the inferred draft. Most monitors should only need name, problem type, and baseline before continuing.",
                    "success" if contract_ready else "secondary",
                ),
                4: (
                    (
                        "Activate the monitor. Model Lens saves the config and triggers bootstrap immediately through the shared refresh workflow."
                        if readiness_mode == "fully_ready"
                        else (
                            "Activate the monitor. Model Lens saves the config, marks bootstrap pending, and the scheduled shared refresh workflow picks it up."
                            if readiness_mode == "scheduler_only"
                            else "Activation is blocked until the workspace readiness checks pass."
                        )
                    ),
                    "primary" if readiness_mode == "fully_ready" else ("secondary" if readiness_mode == "scheduler_only" else "warning"),
                ),
            }
            next_labels = {
                1: "Continue to Source",
                2: "Continue to Contract",
                3: "Continue to Review",
            }
            review = _review_summary(
                control_plane_catalog=control_plane_catalog,
                control_plane_schema=control_plane_schema,
                source_table=(scan_data or {}).get("table_name") or source_table,
                display_name=display_name,
                model_key=model_key,
                model_id_col=model_id_col,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                slice_columns=slice_columns,
                model_id_value=model_id_value,
                model_version_value=model_version_value,
                labels_table=labels_table,
                source_label_col=source_label_col,
                external_label_col=external_label_col,
                baseline_kind=baseline_kind,
                baseline_days=baseline_days,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                problem_type=problem_type,
                performance_metric_names=review_performance_metrics,
                default_performance_metric=review_default_performance_metric,
                drift_cadence_preset=review_drift_cadence,
                performance_cadence_preset=review_performance_cadence,
                schedule_enabled="enabled" in (review_schedule_enabled or []),
                lakebase_instance_name=lakebase_instance_name,
                lakebase_database_name=lakebase_database_name,
                mlflow_experiment_name=mlflow_experiment_name,
                mlflow_registered_model_name=mlflow_registered_model_name,
            )
            return (
                [make_wizard_step(index + 1, label, step) for index, label in enumerate(onboarding.STEP_LABELS)],
                _status_alert(*guidance[step]),
                _step_style(step == 1),
                _step_style(step == 2),
                _step_style(step == 3),
                _step_style(step == 4),
                {"display": "none"} if step == 1 else {},
                {"display": "none"} if step == 4 else {},
                next_disabled,
                next_labels.get(step, "Continue"),
                not (workspace_ready and contract_ready),
                review,
            )
        except Exception as error:
            logger.exception("Failed to render onboarding wizard", exc_info=error)
            try:
                fallback_step = max(1, min(int(current_step or 1), len(onboarding.STEP_LABELS)))
            except (TypeError, ValueError):
                fallback_step = 1
            return (
                [make_wizard_step(index + 1, label, fallback_step) for index, label in enumerate(onboarding.STEP_LABELS)],
                _status_alert(_callback_error_message("onboarding wizard", error), "danger"),
                _step_style(fallback_step == 1),
                _step_style(fallback_step == 2),
                _step_style(fallback_step == 3),
                _step_style(fallback_step == 4),
                {"display": "none"} if fallback_step == 1 else {},
                {"display": "none"} if fallback_step == 4 else {},
                True,
                "Continue",
                True,
                html.Div(),
            )

    @app.callback(
        Output("review-performance-metrics-dropdown", "options"),
        Output("review-performance-metrics-dropdown", "value"),
        Output("review-default-performance-metric-select", "options"),
        Output("review-default-performance-metric-select", "value"),
        Input("problem-type-dropdown", "value"),
        State("review-performance-metrics-dropdown", "value"),
        State("review-default-performance-metric-select", "value"),
    )
    def sync_review_performance_metrics(problem_type, selected_metrics, current_default):
        return _sync_performance_metric_selection(
            problem_type=problem_type,
            selected_metrics=selected_metrics,
            current_default=current_default,
        )

    @app.callback(
        Output("baseline-days-wrapper", "style"),
        Output("baseline-fixed-range-wrapper", "style"),
        Input("baseline-kind-input", "value"),
    )
    def toggle_baseline_policy_inputs(baseline_kind):
        shared = {"marginBottom": "16px"}
        if (baseline_kind or "rolling") == "fixed":
            return {**shared, "display": "none"}, shared
        return shared, {**shared, "display": "none"}

    @app.callback(
        Output("global-model-select", "options"),
        Output("global-model-select", "value"),
        Input("url", "pathname"),
        Input("url", "search"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("global-model-select", "value"),
    )
    def populate_model_selector(_, search, __, session_data, current_value):
        try:
            backend = _make_backend(session_data)
            models = backend.list_models()
            options = [
                {
                    "label": _monitor_option_label(model.get("name"), model.get("id")),
                    "value": model["id"],
                }
                for model in models
            ]
            if not options:
                return [], None
            values = {option["value"] for option in options}
            requested = _selected_model_from_search(search)
            if requested in values:
                return options, requested
            return options, current_value if current_value in values else options[0]["value"]
        except Exception as error:
            logger.exception("Failed to populate global model selector", exc_info=error)
            return [], None

    @app.callback(
        Output("reference-monitor-select", "options"),
        Output("reference-monitor-select", "value"),
        Input("url", "pathname"),
        Input("reference-monitor-status-filter", "value"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("reference-monitor-select", "value"),
    )
    def populate_reference_model_selector(pathname, status_filter, global_model_id, _, session_data, current_value):
        if pathname != "/reference":
            return no_update, no_update
        try:
            backend = _make_backend(session_data)
            models = backend.list_reference_models(status=status_filter or "active")
            options = [
                {
                    "label": _monitor_option_label(model.get("name"), model.get("id"), status=model.get("status")),
                    "value": model["id"],
                }
                for model in models
            ]
            if not options:
                return [], None
            values = {option["value"] for option in options}
            if global_model_id in values:
                return options, global_model_id
            if current_value in values:
                return options, current_value
            return options, options[0]["value"]
        except Exception as error:
            logger.exception("Failed to populate reference model selector", exc_info=error)
            return [], None

    @app.callback(
        Output("reference-page-status", "children", allow_duplicate=True),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("reference-monitor-select", "value"),
        Input("reference-monitor-status-filter", "value"),
        prevent_initial_call=True,
    )
    def clear_reference_status(pathname, *_):
        if pathname != "/reference":
            return no_update
        return html.Div()

    @app.callback(
        Output("sidebar-status", "children"),
        Output("sidebar-alert-badge", "children"),
        Output("sidebar-mode-banner", "children"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def update_sidebar(model_id, _, session_data):
        try:
            backend = _make_backend(session_data)
            if not model_id:
                return "No active monitor selected.", html.Div(), _deployment_mode_prompt(session_data)
            model = backend.get_model_map().get(model_id)
            if not model:
                return "Selected monitor no longer exists.", html.Div(), _deployment_mode_prompt(session_data)
            badge = dbc.Badge(
                [html.I(className="fas fa-exclamation-triangle me-1"), f"{model['open_incident_count']} open incidents"],
                color="danger" if model["open_incident_count"] else "secondary",
                className="mb-2",
            )
            status = html.Div(
                [
                    html.Small(
                        model["description"],
                        title=model["description"],
                        className="text-muted d-block model-lens-sidebar-description",
                    ),
                    html.Small(
                        f"Features: {model['feature_count']} | Baseline: {model['baseline_label']}",
                        className="text-muted d-block",
                    ),
                    html.Small(f"Monitoring rows: {model['total_rows']}", className="text-muted d-block"),
                ]
            )
            return status, badge, _deployment_mode_prompt(session_data)
        except Exception as error:
            logger.exception("Failed to update sidebar", exc_info=error)
            return html.Div([_status_alert(_callback_error_message("sidebar status", error), "danger")]), html.Div(), _deployment_mode_prompt(session_data)

    @app.callback(
        Output("drift-model-banner", "children"),
        Output("deepdive-model-banner", "children"),
        Output("perf-model-banner", "children"),
        Output("quality-model-banner", "children"),
        Input("global-model-select", "value"),
        Input("session-config-store", "data"),
    )
    def update_analysis_banners(model_id, session_data):
        try:
            backend = _make_backend(session_data)
            banner = _model_banner(model_id, backend)
            return banner, banner, banner, banner
        except Exception as error:
            logger.exception("Failed to update analysis banners", exc_info=error)
            banner = _status_alert(_callback_error_message("analysis banners", error), "danger")
            return banner, banner, banner, banner

    @app.callback(
        Output("scan-data", "data"),
        Output("scan-status", "children"),
        Output("scan-preview", "children"),
        Input("scan-source-btn", "n_clicks"),
        State("source-table-input", "value"),
        State("labels-table-input", "value"),
        State("mlflow-experiment-input", "value"),
        State("mlflow-registered-model-input", "value"),
        State("session-config-store", "data"),
        prevent_initial_call=True,
    )
    def scan_source_table(_, source_table, labels_table, mlflow_experiment_name, mlflow_registered_model_name, session_data):
        if not source_table:
            return no_update, _status_alert("Enter a fully qualified source table name.", "warning"), no_update
        try:
            backend = _make_backend(session_data)
            discovery = backend.discover_monitor(
                source_table=source_table.strip(),
                labels_table=(labels_table or "").strip() or None,
                mlflow_experiment_name=(mlflow_experiment_name or "").strip() or None,
                mlflow_registered_model_name=(mlflow_registered_model_name or "").strip() or None,
            )
        except Exception as error:
            logger.exception("Scan source table failed", exc_info=error)
            return None, _status_alert(_user_action_error_message("Scan"), "danger"), html.Div()
        columns = list(discovery.columns)
        preview = pd.DataFrame(discovery.preview_rows)
        schema = pd.DataFrame(discovery.schema_rows)
        label_preview = pd.DataFrame(discovery.label_preview_rows)
        label_schema = pd.DataFrame(discovery.label_schema_rows)
        label_validation = dict(discovery.label_validation or {})
        numeric_count = 0
        if not schema.empty and "data_type" in schema.columns:
            numeric_count = sum(
                1
                for value in schema["data_type"].astype(str).str.lower()
                if any(token in value for token in _NUMERIC_TYPE_TOKENS)
            )
        store = {
            "table_name": source_table.strip(),
            "columns": columns,
            "preview": preview.fillna("").astype(str).to_dict("records"),
            "schema": schema.fillna("").astype(str).to_dict("records"),
            "labels_preview": label_preview.fillna("").astype(str).to_dict("records"),
            "labels_schema": label_schema.fillna("").astype(str).to_dict("records"),
            "discovery": {
                "display_name": discovery.config.display_name,
                "model_key": discovery.config.model_key,
                "timestamp_col": discovery.config.contract.timestamp_col,
                "model_id_col": discovery.config.contract.model_id_col or "",
                "prediction_col": discovery.config.contract.prediction_col,
                "model_id_value": discovery.config.model_id_value or "",
                "model_version_col": discovery.config.contract.model_version_col or "",
                "model_version_value": discovery.config.model_version_value or "",
                "prediction_score_col": discovery.config.contract.prediction_score_col or "",
                "entity_id_col": discovery.config.contract.entity_id_col or "",
                "source_label_col": "" if discovery.config.labels_table else (discovery.config.contract.label_col or ""),
                "external_label_col": (discovery.config.contract.label_col or "") if discovery.config.labels_table else "",
                "labels_join_col": discovery.config.labels_join_col or "",
                "labels_order_col": discovery.config.labels_order_col or "",
                "labels_validation": label_validation,
                "feature_columns": list(discovery.config.contract.feature_columns),
                "categorical_columns": list(discovery.config.contract.categorical_columns),
                "slice_columns": list(discovery.config.contract.slice_columns),
                "problem_type": discovery.config.problem_type,
                "baseline_kind": discovery.config.baseline.kind,
                "baseline_days": discovery.config.baseline.n_days,
                "baseline_start": discovery.config.baseline.baseline_start or "",
                "baseline_end": discovery.config.baseline.baseline_end or "",
                "confidence": discovery.confidence,
                "requires_review": discovery.requires_review,
                "warnings": list(discovery.warnings),
                "mlflow": {
                    "experiment_name": discovery.config.mlflow.experiment_name or (mlflow_experiment_name or "").strip(),
                    "experiment_id": discovery.config.mlflow.experiment_id or "",
                    "run_id": discovery.config.mlflow.run_id or "",
                    "registered_model_name": discovery.config.mlflow.registered_model_name or (mlflow_registered_model_name or "").strip(),
                    "model_version": discovery.config.mlflow.model_version or "",
                },
            },
        }
        status_items = [
            (
                f"Discovered a {discovery.confidence}-confidence monitor draft from {source_table.strip()} with {len(columns)} columns and {numeric_count} numeric candidates.",
                "success" if not discovery.requires_review else "warning",
            )
        ]
        if discovery.config.labels_table:
            status_items.append(
                (
                    "Detected external labels via "
                    f"{discovery.config.labels_table} (join={discovery.config.labels_join_col or 'n/a'}, "
                    f"label={discovery.config.contract.label_col or 'n/a'}, "
                    f"order={discovery.config.labels_order_col or 'n/a'}).",
                    "info",
                )
            )
            if label_validation:
                matched_rows = int(label_validation.get("matched_rows", 0) or 0)
                inference_rows = int(label_validation.get("inference_rows", 0) or 0)
                unmatched_rows = int(label_validation.get("unmatched_rows", 0) or 0)
                status_items.append(
                    (
                        f"Join validation: matched={matched_rows}, "
                        f"unmatched={unmatched_rows}, "
                        f"duplicate_keys={int(label_validation.get('duplicate_join_keys', 0) or 0)}.",
                        "danger" if inference_rows > 0 and matched_rows == 0 else "info",
                    )
                )
        elif discovery.config.contract.label_col:
            status_items.append(
                (
                    f"Detected source labels in the inference table via {discovery.config.contract.label_col}.",
                    "info",
                )
            )
        if discovery.config.mlflow.connected:
            status_items.append(
                (
                    "Linked MLflow lineage for this draft."
                    if discovery.config.mlflow.registered_model_name or discovery.config.mlflow.experiment_name
                    else "MLflow metadata is available for this draft.",
                    "info",
                )
            )
        status_items.extend((warning, "warning") for warning in discovery.warnings[:4])
        status = _status_block(status_items)
        preview_div = html.Div(
            [
                html.H6("Column Types", className="mb-2"),
                _render_frame(schema[[column for column in ("col_name", "data_type") if column in schema.columns]], "No schema metadata returned."),
                html.Hr(),
                html.H6("Sample Rows", className="mb-2"),
                _render_frame(preview, "No preview rows returned.", max_rows=5),
                _render_labels_discovery(
                    labels_table=discovery.config.labels_table or "",
                    label_schema=label_schema[[column for column in ("col_name", "data_type") if column in label_schema.columns]]
                    if not label_schema.empty
                    else label_schema,
                    label_preview=label_preview,
                    label_validation=label_validation,
                    join_col=discovery.config.labels_join_col or "",
                    label_col=discovery.config.contract.label_col or "",
                    order_col=discovery.config.labels_order_col or "",
                )
                if discovery.config.labels_table
                else html.Div(),
            ]
        )
        return store, status, preview_div

    @app.callback(
        Output("display-name-input", "value"),
        Output("model-key-input", "value"),
        Output("timestamp-col-dropdown", "options"),
        Output("timestamp-col-dropdown", "value"),
        Output("model-id-col-dropdown", "options"),
        Output("model-id-col-dropdown", "value"),
        Output("prediction-col-dropdown", "options"),
        Output("prediction-col-dropdown", "value"),
        Output("model-version-col-dropdown", "options"),
        Output("model-version-col-dropdown", "value"),
        Output("prediction-score-col-dropdown", "options"),
        Output("prediction-score-col-dropdown", "value"),
        Output("entity-id-col-dropdown", "options"),
        Output("entity-id-col-dropdown", "value"),
        Output("source-label-col-dropdown", "options"),
        Output("source-label-col-dropdown", "value"),
        Output("model-id-value-input", "value"),
        Output("model-version-value-input", "value"),
        Output("labels-join-col-input", "value"),
        Output("external-label-col-input", "value"),
        Output("labels-order-col-input", "value"),
        Output("feature-cols-dropdown", "options"),
        Output("feature-cols-dropdown", "value"),
        Output("categorical-cols-dropdown", "options"),
        Output("categorical-cols-dropdown", "value"),
        Output("slice-cols-dropdown", "options"),
        Output("slice-cols-dropdown", "value"),
        Output("problem-type-dropdown", "value"),
        Output("baseline-kind-input", "value"),
        Output("baseline-days-input", "value"),
        Output("baseline-fixed-range-input", "start_date"),
        Output("baseline-fixed-range-input", "end_date"),
        Input("scan-data", "data"),
        Input("session-config-store", "data"),
    )
    def populate_monitor_form(scan_data, session_data):
        if not scan_data:
            empty = [], ""
            blank_options = [{"label": "(none)", "value": ""}]
            return (
                "",
                "",
                *empty,
                *empty,
                *empty,
                blank_options,
                "",
                blank_options,
                "",
                blank_options,
                "",
                blank_options,
                "",
                "",
                "",
                "entity_id",
                "label",
                "label_timestamp",
                [],
                [],
                [],
                [],
                [],
                [],
                "classification",
                "rolling",
                7,
                None,
                None,
            )
        columns = scan_data.get("columns", [])
        defaults = _discovery_defaults(scan_data) or _guess_defaults(scan_data.get("table_name", ""), columns)
        try:
            backend = _make_backend(session_data)
            defaults["model_key"] = _suggest_unique_model_key(defaults.get("model_key"), _existing_monitor_keys(backend))
        except Exception:
            defaults["model_key"] = str(defaults.get("model_key") or "").strip()
        required_options = _option_list(columns)
        optional_options = _option_list(columns, include_blank=True)
        feature_values = defaults.get("feature_columns") or defaults.get("features") or []
        categorical_values = defaults.get("categorical_columns") or defaults.get("categorical") or []
        slice_values = defaults.get("slice_columns") or defaults.get("slices") or []
        feature_options = _option_list(feature_values)
        slice_options = _option_list(
            _feature_candidates(
                scan_data,
                [
                    defaults["timestamp_col"],
                    defaults["model_id_col"],
                    defaults["prediction_col"],
                    defaults.get("model_version_col"),
                    defaults.get("prediction_score_col"),
                    defaults.get("entity_id_col"),
                    defaults.get("source_label_col"),
                ],
            )
        )
        return (
            defaults["display_name"],
            defaults["model_key"],
            required_options,
            defaults["timestamp_col"],
            optional_options,
            defaults["model_id_col"],
            required_options,
            defaults["prediction_col"],
            optional_options,
            defaults["model_version_col"],
            optional_options,
            defaults["prediction_score_col"],
            optional_options,
            defaults["entity_id_col"],
            optional_options,
            defaults["source_label_col"],
            defaults.get("model_id_value", ""),
            defaults.get("model_version_value", ""),
            defaults.get("labels_join_col", "entity_id"),
            defaults.get("external_label_col", "label"),
            defaults.get("labels_order_col", "label_timestamp"),
            feature_options,
            feature_values,
            feature_options,
            categorical_values,
            slice_options,
            slice_values,
            defaults.get("problem_type", "classification"),
            defaults.get("baseline_kind", "rolling"),
            defaults.get("baseline_days", 7),
            defaults.get("baseline_start") or None,
            defaults.get("baseline_end") or None,
        )

    @app.callback(
        Output("feature-cols-dropdown", "options", allow_duplicate=True),
        Output("feature-cols-dropdown", "value", allow_duplicate=True),
        Input("scan-data", "data"),
        Input("timestamp-col-dropdown", "value"),
        Input("model-id-col-dropdown", "value"),
        Input("prediction-col-dropdown", "value"),
        Input("model-version-col-dropdown", "value"),
        Input("prediction-score-col-dropdown", "value"),
        Input("entity-id-col-dropdown", "value"),
        Input("source-label-col-dropdown", "value"),
        State("feature-cols-dropdown", "value"),
        prevent_initial_call=True,
    )
    def sync_feature_options(
        scan_data,
        timestamp_col,
        model_id_col,
        prediction_col,
        model_version_col,
        prediction_score_col,
        entity_id_col,
        source_label_col,
        selected_features,
    ):
        candidates = _feature_candidates(
            scan_data,
            [
                timestamp_col,
                model_id_col,
                prediction_col,
                model_version_col,
                prediction_score_col,
                entity_id_col,
                source_label_col,
            ],
        )
        values = [column for column in (selected_features or []) if column in set(candidates)]
        return _option_list(candidates), values

    @app.callback(
        Output("categorical-cols-dropdown", "options", allow_duplicate=True),
        Output("categorical-cols-dropdown", "value", allow_duplicate=True),
        Output("slice-cols-dropdown", "options", allow_duplicate=True),
        Output("slice-cols-dropdown", "value", allow_duplicate=True),
        Input("scan-data", "data"),
        Input("feature-cols-dropdown", "value"),
        Input("timestamp-col-dropdown", "value"),
        Input("model-id-col-dropdown", "value"),
        Input("prediction-col-dropdown", "value"),
        Input("model-version-col-dropdown", "value"),
        Input("prediction-score-col-dropdown", "value"),
        Input("entity-id-col-dropdown", "value"),
        Input("source-label-col-dropdown", "value"),
        State("categorical-cols-dropdown", "value"),
        State("slice-cols-dropdown", "value"),
        prevent_initial_call=True,
    )
    def sync_feature_dependent_options(
        scan_data,
        feature_columns,
        timestamp_col,
        model_id_col,
        prediction_col,
        model_version_col,
        prediction_score_col,
        entity_id_col,
        source_label_col,
        categorical_columns,
        slice_columns,
    ):
        options = _option_list(feature_columns or [])
        allowed = set(feature_columns or [])
        categorical_values = [column for column in (categorical_columns or []) if column in allowed]
        slice_candidates = _feature_candidates(
            scan_data,
            [
                timestamp_col,
                model_id_col,
                prediction_col,
                model_version_col,
                prediction_score_col,
                entity_id_col,
                source_label_col,
            ],
        )
        slice_allowed = set(slice_candidates)
        slice_values = [column for column in (slice_columns or []) if column in slice_allowed]
        return options, categorical_values, _option_list(slice_candidates), slice_values

    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Output("session-config-store", "data", allow_duplicate=True),
        Output("control-plane-ready-store", "data", allow_duplicate=True),
        Output("workspace-readiness-status", "children", allow_duplicate=True),
        Output("workspace-readiness-store", "data", allow_duplicate=True),
        Input("setup-control-plane-btn", "n_clicks"),
        State("control-plane-catalog-input", "value"),
        State("control-plane-schema-input", "value"),
        State("lakebase-instance-input", "value"),
        State("lakebase-database-input", "value"),
        State("lakebase-schema-input", "value"),
        State("create-catalog-toggle", "value"),
        prevent_initial_call=True,
    )
    def setup_control_plane(
        _,
        control_plane_catalog,
        control_plane_schema,
        lakebase_instance_name,
        lakebase_database_name,
        lakebase_schema,
        create_catalog_value,
    ):
        _invalidate_workspace_lakebase_instances_cache()
        session = {
            "control_plane_catalog": (control_plane_catalog or "").strip(),
            "control_plane_schema": (control_plane_schema or "").strip(),
            "lakebase_instance_name": (lakebase_instance_name or "").strip(),
            "lakebase_database_name": (lakebase_database_name or "").strip(),
            "lakebase_schema": (lakebase_schema or "").strip(),
        }
        backend = _make_backend(session)
        try:
            backend.repository.ensure_control_plane(create_catalog="create_catalog" in (create_catalog_value or []))
        except Exception as error:
            return _status_alert(_setup_retry_message(error), "danger"), no_update, no_update, no_update, no_update, no_update
        readiness_state = _workspace_readiness_for_session(session, session)
        return (
            _status_alert(
                f"Control plane ready at {backend.repository.table_names.catalog}.{backend.repository.table_names.schema}.",
                "success",
            ),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            session,
            session,
            _render_workspace_readiness(readiness_state),
            readiness_state,
        )

    @app.callback(
        Output("workspace-readiness-status", "children", allow_duplicate=True),
        Output("workspace-readiness-store", "data", allow_duplicate=True),
        Input("validate-workspace-wiring-btn", "n_clicks"),
        State("control-plane-catalog-input", "value"),
        State("control-plane-schema-input", "value"),
        State("lakebase-instance-input", "value"),
        State("lakebase-database-input", "value"),
        State("lakebase-schema-input", "value"),
        State("control-plane-ready-store", "data"),
        prevent_initial_call=True,
    )
    def validate_workspace_wiring(
        _,
        control_plane_catalog,
        control_plane_schema,
        lakebase_instance_name,
        lakebase_database_name,
        lakebase_schema,
        ready_state,
    ):
        _invalidate_workspace_lakebase_instances_cache()
        session = {
            "control_plane_catalog": (control_plane_catalog or "").strip(),
            "control_plane_schema": (control_plane_schema or "").strip(),
            "lakebase_instance_name": (lakebase_instance_name or "").strip(),
            "lakebase_database_name": (lakebase_database_name or "").strip(),
            "lakebase_schema": (lakebase_schema or "").strip(),
        }
        readiness_state = _workspace_readiness_for_session(ready_state, session)
        return _render_workspace_readiness(readiness_state), readiness_state

    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Output("session-config-store", "data", allow_duplicate=True),
        Input("save-monitor-btn", "n_clicks"),
        State("scan-data", "data"),
        State("display-name-input", "value"),
        State("model-key-input", "value"),
        State("timestamp-col-dropdown", "value"),
        State("model-id-col-dropdown", "value"),
        State("prediction-col-dropdown", "value"),
        State("model-id-value-input", "value"),
        State("model-version-value-input", "value"),
        State("model-version-col-dropdown", "value"),
        State("prediction-score-col-dropdown", "value"),
        State("entity-id-col-dropdown", "value"),
        State("source-label-col-dropdown", "value"),
        State("labels-table-input", "value"),
        State("labels-join-col-input", "value"),
        State("external-label-col-input", "value"),
        State("labels-order-col-input", "value"),
        State("feature-cols-dropdown", "value"),
        State("categorical-cols-dropdown", "value"),
        State("slice-cols-dropdown", "value"),
        State("problem-type-dropdown", "value"),
        State("baseline-kind-input", "value"),
        State("baseline-days-input", "value"),
        State("baseline-fixed-range-input", "start_date"),
        State("baseline-fixed-range-input", "end_date"),
        State("review-drift-cadence-select", "value"),
        State("review-performance-cadence-select", "value"),
        State("review-schedule-enabled-toggle", "value"),
        State("review-performance-metrics-dropdown", "value"),
        State("review-default-performance-metric-select", "value"),
        State("control-plane-catalog-input", "value"),
        State("control-plane-schema-input", "value"),
        State("lakebase-instance-input", "value"),
        State("lakebase-database-input", "value"),
        State("lakebase-schema-input", "value"),
        State("control-plane-ready-store", "data"),
        State("workspace-readiness-store", "data"),
        prevent_initial_call=True,
    )
    def save_monitor(
        _,
        scan_data,
        display_name,
        model_key,
        timestamp_col,
        model_id_col,
        prediction_col,
        model_id_value,
        model_version_value,
        model_version_col,
        prediction_score_col,
        entity_id_col,
        source_label_col,
        labels_table,
        labels_join_col,
        external_label_col,
        labels_order_col,
        feature_columns,
        categorical_columns,
        slice_columns,
        problem_type,
        baseline_kind,
        baseline_days,
        baseline_start,
        baseline_end,
        review_drift_cadence,
        review_performance_cadence,
        review_schedule_enabled,
        review_performance_metrics,
        review_default_performance_metric,
        control_plane_catalog,
        control_plane_schema,
        lakebase_instance_name,
        lakebase_database_name,
        lakebase_schema,
        ready_state,
        workspace_readiness_state,
    ):
        if not scan_data:
            return _status_alert("Scan a source table before saving a monitor.", "warning"), no_update, no_update
        if not feature_columns:
            return _status_alert("Select at least one feature column.", "warning"), no_update, no_update
        discovery = (scan_data or {}).get("discovery", {}) if isinstance(scan_data, dict) else {}
        labels_table = (labels_table or "").strip()
        external_label_col = (external_label_col or "").strip()
        labels_join_col = (labels_join_col or "").strip()
        labels_order_col = (labels_order_col or "").strip()
        model_id_value = (model_id_value or "").strip()
        model_version_value = (model_version_value or "").strip()
        session = {
            "control_plane_catalog": (control_plane_catalog or "").strip(),
            "control_plane_schema": (control_plane_schema or "").strip(),
            "lakebase_instance_name": (lakebase_instance_name or "").strip(),
            "lakebase_database_name": (lakebase_database_name or "").strip(),
            "lakebase_schema": (lakebase_schema or "").strip(),
        }
        if not _ready_for_session(ready_state, session):
            return _status_alert("Run Setup Control Plane successfully before saving a monitor.", "warning"), no_update, no_update
        if _workspace_readiness_mode(workspace_readiness_state) not in {"scheduler_only", "fully_ready"}:
            issues = [str(item) for item in (workspace_readiness_state or {}).get("blocking_issues", []) if str(item).strip()]
            detail = f" {issues[0]}" if issues else ""
            return _status_alert(
                "Validate Workspace Wiring successfully before saving a monitor."
                f"{detail}",
                "warning",
            ), no_update, no_update
        label_col = external_label_col or source_label_col or None
        source_join_col = _source_labels_join_col(scan_data, entity_id_col, labels_join_col)
        if labels_table and (not source_join_col or not labels_join_col or not label_col):
            return _status_alert(
                "External labels require External Labels Join Column, External Label Column, and either Entity ID Column or the same join column name in the inference table.",
                "warning",
            ), no_update, no_update
        if model_version_value and not model_version_col:
            return _status_alert("Monitored Model Version Value requires a mapped Model Version Column.", "warning"), no_update, no_update
        try:
            contract = build_inference_contract(
                columns=scan_data["columns"],
                timestamp_col=timestamp_col,
                model_id_col=(model_id_col or "").strip() or None,
                prediction_col=prediction_col,
                model_version_col=model_version_col or None,
                prediction_score_col=prediction_score_col or None,
                label_col=label_col,
                entity_id_col=entity_id_col or None,
                feature_columns=feature_columns or [],
                categorical_columns=categorical_columns or [],
                slice_columns=slice_columns or [],
            )
            config = MonitorConfig(
                model_key=(model_key or scan_data["table_name"].split(".")[-1]).strip(),
                display_name=(display_name or model_key or scan_data["table_name"]).strip(),
                source_table=scan_data["table_name"],
                contract=contract,
                baseline=(
                    build_fixed_baseline((baseline_start or "").strip(), (baseline_end or "").strip())
                    if (baseline_kind or "rolling") == "fixed"
                    else build_default_baseline(int(baseline_days or 7))
                ),
                problem_type=problem_type or "classification",
                model_id_value=model_id_value or None,
                model_version_value=model_version_value or None,
                labels_table=labels_table or None,
                labels_join_col=labels_join_col or None,
                labels_order_col=labels_order_col or None,
                performance_metric_names=tuple(review_performance_metrics or ()),
                default_performance_metric=review_default_performance_metric or None,
                drift_cadence_preset=review_drift_cadence or "6h",
                performance_cadence_preset=(
                    review_performance_cadence
                    if label_col
                    else "disabled"
                ),
                schedule_enabled="enabled" in (review_schedule_enabled or []),
                mlflow=MLflowLineage(
                    experiment_name=str(((discovery.get("mlflow") or {}).get("experiment_name") or "")).strip() or None,
                    experiment_id=str(((discovery.get("mlflow") or {}).get("experiment_id") or "")).strip() or None,
                    run_id=str(((discovery.get("mlflow") or {}).get("run_id") or "")).strip() or None,
                    registered_model_name=str(((discovery.get("mlflow") or {}).get("registered_model_name") or "")).strip() or None,
                    model_version=str(((discovery.get("mlflow") or {}).get("model_version") or "")).strip() or None,
                ),
                created_by="app",
            )
            backend = _make_backend(session)
            existing_config, conflict_error = _resolve_monitor_config_by_key(backend, config.model_key)
            if conflict_error and "no longer exists" not in conflict_error.lower():
                return _status_alert(conflict_error, "danger"), no_update, no_update
            if existing_config:
                return (
                    _status_alert(
                        "A monitor with this model key already exists: "
                        f"{existing_config.display_name} ({existing_config.model_key}, {_monitor_status_text(existing_config.status)}). "
                        "Choose a different model key here, or edit the existing monitor in Monitor Settings.",
                        "warning",
                    ),
                    no_update,
                    no_update,
                )
            backend.repository.validate_monitor_source(config)
            backend.repository.upsert_monitor_config(config)
            backend.repository.mark_monitor_bootstrap_pending(config)
        except Exception as error:
            logger.exception("Save monitor failed", exc_info=error)
            return _status_alert(_user_action_error_message("Save"), "danger"), no_update, no_update
        messages: list[tuple[str, str]] = []
        try:
            trigger = trigger_refresh_job(
                model_key=config.model_key,
                control_plane_catalog=session["control_plane_catalog"],
                control_plane_schema=session["control_plane_schema"],
                lakebase_instance_name=session["lakebase_instance_name"],
                lakebase_database_name=session["lakebase_database_name"],
                lakebase_schema=session["lakebase_schema"],
                scope="bootstrap",
            )
            run_id_text = f", run_id={trigger.run_id}" if trigger.run_id is not None else ""
            messages.append((
                f"Saved monitor {config.model_key}. Refresh job triggered asynchronously (job_id={trigger.job_id}{run_id_text}). Check Overview in a minute.",
                "success",
            ))
        except Exception as error:
            messages.append((_refresh_job_unavailable_message(config.model_key, error), "warning"))
        non_numeric = _non_numeric_features(feature_columns or [], scan_data)
        if non_numeric:
            messages.append((
                "These selected feature columns are not numeric and will be stored in the contract but skipped by the current drift engine: "
                f"{', '.join(non_numeric)}.",
                "warning",
            ))
        return _status_block(messages), datetime.now(timezone.utc).isoformat(timespec="seconds"), session

    @app.callback(
        Output("overview-page-body", "children"),
        Input("url", "pathname"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def render_overview(pathname, _, session_data):
        if pathname != "/":
            return no_update
        try:
            backend = _make_backend(session_data)
            overview_data = backend.get_overview_rows(metric="psi")
            if not overview_data:
                return make_empty_state("No monitors onboarded yet. Go to Onboarding to add your first model.", icon="fas fa-plus-circle")
            normalized_overview: list[dict[str, object]] = []
            for row in overview_data:
                normalized_row = dict(row)
                normalized_row["max_metric"] = float(normalized_row.get("max_metric", normalized_row.get("max_psi", 0.0)) or 0.0)
                normalized_row["avg_metric"] = float(normalized_row.get("avg_metric", normalized_row.get("avg_psi", 0.0)) or 0.0)
                normalized_overview.append(normalized_row)

            computing = sum(1 for row in normalized_overview if row["computing"])
            healthy = sum(
                1
                for row in normalized_overview
                if not row["computing"] and row["max_metric"] < float(row.get("threshold_warning") or 0.0)
            )
            warning = sum(
                1
                for row in normalized_overview
                if (
                    not row["computing"]
                    and float(row.get("threshold_warning") or 0.0) <= row["max_metric"] < float(row.get("threshold_critical") or 0.0)
                )
            )
            critical = sum(
                1
                for row in normalized_overview
                if not row["computing"] and row["max_metric"] >= float(row.get("threshold_critical") or 0.0)
            )
            summary_row = dbc.Row(
                [
                    dbc.Col(make_metric_card("Models Monitored", str(len(normalized_overview)), "Active in production"), md=6, lg=4, xl=2),
                    dbc.Col(make_metric_card("Healthy", str(healthy), "Below each monitor's warning threshold", "success"), md=6, lg=4, xl=2),
                    dbc.Col(make_metric_card("Warning", str(warning), "Between each monitor's warning and critical thresholds", "warning"), md=6, lg=4, xl=2),
                    dbc.Col(make_metric_card("Critical", str(critical), "At or above each monitor's critical threshold", "danger"), md=6, lg=4, xl=2),
                    dbc.Col(make_metric_card("Computing/Pending", str(computing), "No drift history yet", "info"), md=6, lg=4, xl=2),
                ],
                className="mb-4 g-3",
            )
            sorted_data = sorted(normalized_overview, key=lambda item: (not item["computing"], item["max_metric"]), reverse=True)
            model_cards = [
                dbc.Col(
                    dcc.Link(
                        make_model_status_card(
                            model_name=row["model_name"],
                            model_id=row["model_id"],
                            description=row["description"],
                            max_metric=row["max_metric"],
                            avg_metric=row["avg_metric"],
                            drifting_count=row["drifting_features"],
                            total_features=row["total_features"],
                            max_null_rate=None if row["computing"] else row["max_null_rate"],
                            has_labels=row["has_labels"],
                            metric_label="PSI",
                            metric_key="psi",
                            thresholds=row.get("thresholds"),
                            computing=row["computing"],
                            freshness_status=row["freshness_status"],
                            last_run_status=row["last_run_status"],
                        ),
                        href=f"/drift?model={row['model_id']}",
                        style={"textDecoration": "none"},
                    ),
                    md=6,
                    lg=4,
                    className="mb-3",
                )
                for row in sorted_data
            ]
            summary_chart = charts.build_multi_model_summary(
                [
                    {
                        "model": row["model_name"],
                        "max_metric": row["max_metric"],
                        "avg_metric": row["avg_metric"],
                        "drifting_features": row["drifting_features"],
                        "computing": row["computing"],
                        "thresholds": row.get("thresholds"),
                    }
                    for row in sorted_data
                ],
                metric="psi",
            )
            details = pd.DataFrame(
                [
                    {
                        "model": row["model_name"],
                        "versions": ", ".join(str(value) for value in row.get("versions", [])),
                        "status": (
                            "Computing/Pending"
                            if row["computing"]
                            else (
                                "Critical"
                                if row["max_metric"] >= float(row.get("threshold_critical") or 0.0)
                                else (
                                    "Warning"
                                    if row["max_metric"] >= float(row.get("threshold_warning") or 0.0)
                                    else "Healthy"
                                )
                            )
                        ),
                        "max_psi": "—" if row["computing"] else round(row["max_metric"], 4),
                        "avg_psi": "—" if row["computing"] else round(row["avg_metric"], 4),
                        "avg_js": "—" if row["computing"] else round(row["avg_js"], 4),
                        "drifting_features": "Computing/Pending" if row["computing"] else f"{row['drifting_features']} / {row['total_features']}",
                        "top_drifter": row["top_drifter"],
                        "max_null_rate": "—" if row["computing"] else round(row["max_null_rate"], 2),
                    }
                    for row in sorted_data
                ]
            )
            return html.Div([summary_row, dbc.Row(model_cards, className="mb-4"), make_chart_card(summary_chart), _render_frame(details, "No overview detail available.")])
        except Exception as error:
            return _callback_error_panel("overview", error, icon="fas fa-plus-circle")

    @app.callback(
        Output("incidents-monitor-filter", "options"),
        Output("incidents-monitor-filter", "value"),
        Output("incidents-metric-filter", "options"),
        Output("incidents-metric-filter", "value"),
        Input("url", "pathname"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("incidents-monitor-filter", "value"),
        State("incidents-metric-filter", "value"),
    )
    def populate_incident_filters(pathname, _, session_data, current_model_id, current_metric_name):
        if pathname != "/incidents":
            return no_update, no_update, no_update, no_update
        try:
            backend = _make_backend(session_data)
            incidents_data = backend.get_incidents_data(limit_history=100)
            model_options, metric_options = _incident_options(incidents_data)
            model_values = {option["value"] for option in model_options}
            metric_values = {option["value"] for option in metric_options}
            resolved_model = current_model_id if current_model_id in model_values else None
            resolved_metric = current_metric_name if current_metric_name in metric_values else None
            return model_options, resolved_model, metric_options, resolved_metric
        except Exception as error:
            logger.exception("Failed to populate incident filters", exc_info=error)
            return [], None, [], None

    @app.callback(
        Output("incidents-page-body", "children"),
        Input("url", "pathname"),
        Input("incidents-monitor-filter", "value"),
        Input("incidents-severity-filter", "value"),
        Input("incidents-status-filter", "value"),
        Input("incidents-metric-filter", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def render_incidents_page(pathname, model_id, severity, status, metric_name, _, session_data):
        if pathname != "/incidents":
            return no_update
        try:
            backend = _make_backend(session_data)
            incidents_data = backend.get_incidents_data(limit_history=100)
            return _render_incidents_page(
                incidents_data,
                model_id=model_id,
                severity=severity,
                status=status,
                metric_name=metric_name,
            )
        except Exception as error:
            return _callback_error_panel("incident history", error)

    @app.callback(
        Output(_threshold_input_id("psi", "warning", prefix="drift"), "value"),
        Output(_threshold_input_id("psi", "critical", prefix="drift"), "value"),
        Output(_threshold_input_id("js_divergence", "warning", prefix="drift"), "value"),
        Output(_threshold_input_id("js_divergence", "critical", prefix="drift"), "value"),
        Output(_threshold_input_id("kl_divergence", "warning", prefix="drift"), "value"),
        Output(_threshold_input_id("kl_divergence", "critical", prefix="drift"), "value"),
        Output(_threshold_input_id("null_rate", "warning", prefix="drift"), "value"),
        Output(_threshold_input_id("null_rate", "critical", prefix="drift"), "value"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def sync_drift_threshold_inputs(pathname, model_id, _, session_data):
        if pathname != "/drift":
            return (no_update,) * 8
        if not model_id:
            return (None,) * 8
        try:
            backend = _make_backend(session_data)
            config = backend.get_monitor_config(model_id)
            resolved_thresholds = _resolved_monitor_thresholds(config)
            values: list[float] = []
            for metric in THRESHOLD_METRICS:
                warning, critical = get_thresholds(metric, resolved_thresholds)
                values.extend([warning, critical])
            return tuple(values)
        except Exception:
            return (None,) * 8

    @app.callback(
        Output("drift-threshold-status", "children"),
        Output("reload-token", "data", allow_duplicate=True),
        Input("drift-save-thresholds-btn", "n_clicks"),
        Input("drift-reset-thresholds-btn", "n_clicks"),
        State("global-model-select", "value"),
        State(_threshold_input_id("psi", "warning", prefix="drift"), "value"),
        State(_threshold_input_id("psi", "critical", prefix="drift"), "value"),
        State(_threshold_input_id("js_divergence", "warning", prefix="drift"), "value"),
        State(_threshold_input_id("js_divergence", "critical", prefix="drift"), "value"),
        State(_threshold_input_id("kl_divergence", "warning", prefix="drift"), "value"),
        State(_threshold_input_id("kl_divergence", "critical", prefix="drift"), "value"),
        State(_threshold_input_id("null_rate", "warning", prefix="drift"), "value"),
        State(_threshold_input_id("null_rate", "critical", prefix="drift"), "value"),
        State("session-config-store", "data"),
        prevent_initial_call=True,
    )
    def save_drift_thresholds(
        _save_clicks,
        _reset_clicks,
        model_id,
        psi_warning,
        psi_critical,
        js_warning,
        js_critical,
        kl_warning,
        kl_critical,
        null_warning,
        null_critical,
        session_data,
    ):
        if not model_id:
            return _status_alert("Select a monitor before updating drift thresholds.", "warning"), no_update
        backend = _make_backend(session_data)
        config = backend.get_monitor_config(model_id)
        if not config:
            return _status_alert("Selected monitor no longer exists.", "warning"), no_update
        try:
            triggered_id = ctx.triggered_id
            threshold_overrides = (
                {}
                if triggered_id == "drift-reset-thresholds-btn"
                else _collect_threshold_overrides_from_inputs(
                    {
                        "psi": (psi_warning, psi_critical),
                        "js_divergence": (js_warning, js_critical),
                        "kl_divergence": (kl_warning, kl_critical),
                        "null_rate": (null_warning, null_critical),
                    }
                )
            )
            updated = replace(config, threshold_overrides=threshold_overrides)
            backend.repository.upsert_monitor_config(updated)
        except ValueError as error:
            return _status_alert(str(error), "warning"), no_update
        except Exception as error:
            logger.exception("Failed to update drift thresholds for %s", model_id, exc_info=error)
            return _status_alert(_user_action_error_message("Updating drift thresholds"), "danger"), no_update
        action_text = "Reset drift thresholds to defaults." if ctx.triggered_id == "drift-reset-thresholds-btn" else "Saved drift thresholds."
        return _status_alert(action_text, "success"), datetime.now(timezone.utc).isoformat(timespec="seconds")

    @app.callback(
        Output("drift-heatmap-container", "children"),
        Output("drift-categorical-note", "children"),
        Output("drift-timeline-container", "children"),
        Output("drift-top-drifters-container", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        Input("drift-apply-filters-btn", "n_clicks"),
        State("drift-metric-select", "value"),
        State("drift-granularity-select", "value"),
        State("drift-top-n", "value"),
        State("drift-date-range", "start_date"),
        State("drift-date-range", "end_date"),
        State("drift-class-basis-select", "value"),
        State("drift-class-value-select", "value"),
        State("drift-threshold-toggle", "value"),
    )
    def render_drift(
        pathname,
        model_id,
        _,
        session_data,
        _apply_clicks,
        metric,
        granularity,
        top_n,
        start_date,
        end_date,
        class_basis,
        class_value,
        show_thresholds,
    ):
        if pathname != "/drift":
            return no_update, no_update, no_update, no_update
        try:
            if not model_id:
                empty = make_empty_state("Select a model to inspect drift.", icon="fas fa-wave-square")
                return empty, html.Div(), html.Div(), html.Div()
            backend = _make_backend(session_data)
            config = backend.get_monitor_config(model_id)
            resolved_thresholds = _resolved_monitor_thresholds(config)
            _, _, class_filter_active = normalize_class_filter(class_basis, class_value)
            notes: list[object] = []
            if class_filter_active and not supports_binary_class_filters(config):
                empty = make_empty_state(
                    "Class filters are available only for binary classification monitors with labels.",
                    icon="fas fa-wave-square",
                )
                return empty, html.Div([_status_alert("Class filters are available only for binary classification monitors with labels.", "warning")]), html.Div(), html.Div()
            drift = backend.get_drift_results(
                model_id,
                granularity=granularity or "daily",
                start_date=start_date,
                end_date=end_date,
                class_basis=class_basis,
                class_value=class_value,
            )
            if drift.empty:
                empty_reason = str(drift.attrs.get("_empty_reason") or "").strip().lower()
                if class_filter_active:
                    if empty_reason == "no_filtered_rows":
                        empty_message = "No rows matched the selected class filter in this date range."
                    elif empty_reason == "filtered_source_bounds_unavailable":
                        empty_message = (
                            "Filtered drift history is not available for the full range yet. "
                            "Select a date range or refresh to populate class-aware daily facts."
                        )
                    elif empty_reason == "missing_class_facts":
                        empty_message = (
                            "Filtered drift history is not available yet for this monitor. "
                            "Run a refresh to populate class-aware daily facts or select a date range with published filtered history."
                        )
                    else:
                        empty_message = "Filtered drift history is unavailable until the next refresh populates class-aware daily facts."
                    empty = make_empty_state(empty_message, icon="fas fa-wave-square")
                else:
                    empty = make_empty_state("No drift history available yet. Run a refresh to populate this page.", icon="fas fa-wave-square")
                return empty, html.Div(), html.Div(), html.Div()
            normalized_top_n = _normalize_top_n(top_n, default=10, minimum=1, maximum=50)
            ranked_features = _historical_drift_feature_ranking(
                drift,
                metric=metric or "psi",
                top_n=normalized_top_n,
            )
            period_count = int(drift["period"].nunique()) if "period" in drift.columns else 0
            history_message = _comparison_history_message(period_count, granularity or "daily")
            if history_message:
                notes.append(_status_alert(history_message, "info"))
            filter_summary = _analysis_filter_summary(
                start_date=start_date,
                end_date=end_date,
                class_basis=class_basis,
                class_value=class_value,
            )
            if filter_summary:
                notes.append(_status_alert(f"Active filters: {filter_summary}", "secondary"))
            notes.append(
                _status_alert(
                    f"Top features are ranked by the highest historical {(metric or 'psi').upper()} across comparison windows.",
                    "secondary",
                )
            )
            if config and config.contract.categorical_columns:
                notes.append(
                    _status_alert(
                        "Categorical features are stored in the monitor contract, but the current drift engine renders only numeric feature drift on this page.",
                        "secondary",
                    )
                )
            timeline_features = ranked_features["feature"].tolist() if "feature" in ranked_features.columns else []
            if not timeline_features and "feature" in drift.columns:
                timeline_features = drift["feature"].dropna().astype(str).drop_duplicates().tolist()[:8]
            filtered_drift = drift[drift["feature"].isin(timeline_features)].copy() if timeline_features else drift
            heatmap_scale = charts.describe_drift_heatmap_scale(
                filtered_drift,
                metric=metric or "psi",
                show_thresholds=bool(show_thresholds),
                thresholds=resolved_thresholds,
            )
            clip_note = str(heatmap_scale.get("clip_note") or "").strip()
            if clip_note:
                notes.append(_status_alert(clip_note, "secondary"))
            title_suffix = f" ({filter_summary})" if filter_summary else ""
            heatmap_title = f"{(granularity or 'daily').title()} Feature Drift Heatmap{title_suffix}"
            return (
                make_chart_card(
                    charts.build_drift_heatmap(
                        filtered_drift,
                        metric=metric or "psi",
                        title=heatmap_title,
                        show_thresholds=bool(show_thresholds),
                        thresholds=resolved_thresholds,
                    )
                ),
                html.Div(notes) if notes else html.Div(),
                make_chart_card(
                    charts.build_drift_timeline(
                        filtered_drift,
                        timeline_features,
                        metric=metric or "psi",
                        show_thresholds=bool(show_thresholds),
                        thresholds=resolved_thresholds,
                        title=f"{(metric or 'psi').upper()} Over Time{title_suffix}",
                    )
                ),
                make_chart_card(
                    charts.build_top_drifters_bar(
                        filtered_drift,
                        metric=metric or "psi",
                        top_n=normalized_top_n,
                        title=f"Top {normalized_top_n} Drifting Features (Historical Max){title_suffix}",
                        show_thresholds=bool(show_thresholds),
                        thresholds=resolved_thresholds,
                    )
                ),
            )
        except Exception as error:
            logger.exception("Failed to render drift analysis", exc_info=error)
            empty = make_empty_state("Drift analysis is unavailable right now.", icon="fas fa-wave-square")
            return empty, html.Div([_status_alert(_callback_error_message("drift analysis", error), "danger")]), html.Div(), html.Div()

    @app.callback(
        Output("deepdive-feature-select", "options"),
        Output("deepdive-feature-select", "value"),
        Output("deepdive-dimension-select", "options"),
        Output("deepdive-dimension-select", "value"),
        Input("url", "pathname"),
        Input("url", "search"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("deepdive-feature-select", "value"),
        State("deepdive-dimension-select", "value"),
    )
    def populate_feature_deep_dive(pathname, search, model_id, _, session_data, feature_value, dimension_value):
        if pathname != "/features":
            return no_update, no_update, no_update, no_update
        try:
            backend = _make_backend(session_data)
            feature_options = _option_list(backend.get_feature_options(model_id or ""))
            dimension_options = _option_list(backend.get_dimension_options(model_id or ""), include_blank=True)
        except Exception:
            return [], None, [], ""
        feature_values = {option["value"] for option in feature_options}
        dimension_values = {option["value"] for option in dimension_options}
        requested_feature = _selected_feature_from_search(search)
        selected_feature = requested_feature if requested_feature in feature_values else (feature_value if feature_value in feature_values else None)
        if selected_feature is None and model_id:
            try:
                drift = backend.get_drift_results(model_id)
                ranked_features = _historical_drift_feature_ranking(drift, metric="psi", top_n=max(len(feature_options), 1))
                ranked_values = [
                    str(value)
                    for value in ranked_features.get("feature", pd.Series(dtype=str)).dropna().tolist()
                    if str(value) in feature_values
                ]
                if ranked_values:
                    selected_feature = ranked_values[0]
            except Exception:
                selected_feature = None
        if selected_feature is None:
            selected_feature = feature_options[0]["value"] if feature_options else None
        selected_dimension = dimension_value if dimension_value in dimension_values else ""
        return feature_options, selected_feature, dimension_options, selected_dimension

    @app.callback(
        Output("deepdive-bin-count-input", "disabled"),
        Output("deepdive-custom-edges-input", "disabled"),
        Output("deepdive-outlier-value-input", "disabled"),
        Output("deepdive-outlier-value-label", "children"),
        Output("deepdive-outlier-value-help", "children"),
        Input("deepdive-binning-mode-select", "value"),
        Input("deepdive-outlier-mode-select", "value"),
    )
    def sync_deepdive_controls(binning_mode, outlier_mode):
        normalized_mode = str(binning_mode or "auto").strip().lower()
        normalized_outlier_mode = _normalize_outlier_mode(outlier_mode)
        if normalized_outlier_mode == "percentile_clip":
            return (
                normalized_mode != "fixed",
                normalized_mode != "custom",
                False,
                "Trim Percentile P",
                "Percentile Clip keeps values between P and 100-P.",
            )
        if normalized_outlier_mode == "iqr_fence":
            return (
                normalized_mode != "fixed",
                normalized_mode != "custom",
                False,
                "IQR Multiplier K",
                "IQR Fence keeps values inside Q1 - K*IQR and Q3 + K*IQR.",
            )
        return (
            normalized_mode != "fixed",
            normalized_mode != "custom",
            True,
            "Outlier Parameter",
            "Percentile Clip uses P / 100-P clipping. IQR Fence uses Q1 - K*IQR to Q3 + K*IQR.",
        )

    @app.callback(
        Output("deepdive-distribution-container", "children"),
        Output("deepdive-dimension-container", "children"),
        Output("deepdive-context-container", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("deepdive-feature-select", "value"),
        Input("deepdive-dimension-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        Input("deepdive-apply-controls-btn", "n_clicks"),
        State("deepdive-binning-mode-select", "value"),
        State("deepdive-bin-count-input", "value"),
        State("deepdive-custom-edges-input", "value"),
        State("deepdive-outlier-mode-select", "value"),
        State("deepdive-outlier-value-input", "value"),
        prevent_initial_call=True,
    )
    def render_feature_deep_dive(
        pathname,
        model_id,
        feature,
        dimension,
        _reload_token,
        session_data,
        _apply_clicks,
        binning_mode,
        bin_count,
        custom_edges_text,
        outlier_mode,
        outlier_value_raw,
    ):
        if pathname != "/features":
            return no_update, no_update, no_update
        if not model_id or not feature:
            empty = make_empty_state("Select a model and feature to inspect.", icon="fas fa-search")
            return empty, html.Div(), "Select a model and feature to inspect."
        try:
            backend = _make_backend(session_data)
            normalized_mode = str(binning_mode or "auto").strip().lower()
            custom_edges = _parse_custom_edges(custom_edges_text) if normalized_mode == "custom" else None
            normalized_outlier_mode = _normalize_outlier_mode(outlier_mode)
            outlier_value = _outlier_control_value(normalized_outlier_mode, outlier_value_raw)
            details = backend.get_feature_distribution_details(
                model_id,
                feature,
                require_exact_samples=(normalized_mode == "custom" or normalized_outlier_mode != "off"),
            )
            baseline = details["baseline"]
            current = details["current"]
            distribution_source = str(details.get("distribution_source") or "")
            if distribution_source in {"unavailable_requested_raw", "unavailable_unsafe_bounded_read"}:
                distribution = make_empty_state(
                    _feature_distribution_source_message(distribution_source),
                    icon="fas fa-chart-area",
                )
            else:
                distribution = make_chart_card(
                    charts.build_feature_distribution(
                        baseline,
                        current,
                        feature,
                        binning_mode=normalized_mode,
                        n_bins=_normalize_top_n(bin_count, default=40, minimum=2, maximum=200),
                        custom_edges=custom_edges,
                        outlier_mode=normalized_outlier_mode,
                        outlier_value=outlier_value,
                    )
                )
            dimension_chart = html.Div()
            if dimension:
                breakdown = backend.get_dimension_breakdown(model_id, feature, dimension)
                dimension_chart = html.Div(
                    [
                        html.Small(
                            "This chart compares average vs median and the middle 50% spread (P25 to P75) for each slice.",
                            className="text-muted d-block mb-2",
                        ),
                        make_chart_card(charts.build_dimension_breakdown(breakdown, feature, dimension)),
                    ]
                )
            outlier_text = "Off"
            if normalized_outlier_mode == "percentile_clip":
                outlier_text = f"Percentile Clip (P={outlier_value:.1f})"
            elif normalized_outlier_mode == "iqr_fence":
                outlier_text = f"IQR Fence (K={outlier_value:.2f})"
            context_children = html.Div(
                [
                    html.Div(str(details.get("window_label") or "Latest comparison window unavailable."), className="mb-1"),
                    html.Div(
                        _feature_distribution_source_message(
                            str(details.get("distribution_source") or "unavailable"),
            )
                ),
                html.Div(
                    f"Binning: {normalized_mode.title()}"
                    + (
                            f" | Bin Count: {_normalize_top_n(bin_count, default=40, minimum=2, maximum=200)}"
                            if normalized_mode == "fixed"
                            else ""
                        )
                        + f" | Outlier Mode: {outlier_text}",
                        className="mt-1",
                    ),
                ]
            )
            return distribution, dimension_chart, context_children
        except ValueError as error:
            return _status_alert(str(error), "warning"), html.Div(), str(error)
        except Exception as error:
            logger.exception("Failed to render feature deep dive", exc_info=error)
            return make_empty_state("Could not load feature detail. Check logs and try again.", icon="fas fa-triangle-exclamation"), html.Div(), "Feature detail is unavailable right now."

    @app.callback(
        Output("quality-kpi-cards", "children"),
        Output("quality-volume-container", "children"),
        Output("quality-null-rates-container", "children"),
        Output("quality-prediction-container", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        Input("quality-apply-filters-btn", "n_clicks"),
        State("quality-date-range", "start_date"),
        State("quality-date-range", "end_date"),
        State("quality-class-basis-select", "value"),
        State("quality-class-value-select", "value"),
        State("quality-threshold-toggle", "value"),
    )
    def render_quality(
        pathname,
        model_id,
        _,
        session_data,
        _apply_clicks,
        start_date,
        end_date,
        class_basis,
        class_value,
        show_thresholds,
    ):
        if pathname != "/quality":
            return no_update, no_update, no_update, no_update
        try:
            if not model_id:
                empty = make_empty_state("Select a model to inspect quality.", icon="fas fa-database")
                return empty, html.Div(), html.Div(), html.Div()
            backend = _make_backend(session_data)
            config = backend.get_monitor_config(model_id)
            resolved_thresholds = _resolved_monitor_thresholds(config)
            _, _, class_filter_active = normalize_class_filter(class_basis, class_value)
            if class_filter_active and not supports_binary_class_filters(config):
                empty = make_empty_state(
                    "Class filters are available only for binary classification monitors with labels.",
                    icon="fas fa-database",
                )
                return empty, html.Div(), html.Div(), html.Div()
            quality = backend.get_quality_stats(
                model_id,
                start_date=start_date,
                end_date=end_date,
                class_basis=class_basis,
                class_value=class_value,
            )
            empty_reason = str((quality or {}).get("_empty_reason") or "").strip().lower()
            if empty_reason == "no_filtered_rows":
                empty = make_empty_state(
                    "No rows matched the selected class filter in this date range.",
                    icon="fas fa-database",
                )
                return empty, html.Div(), html.Div(), html.Div()
            if empty_reason == "filtered_source_bounds_unavailable":
                empty = make_empty_state(
                    "Filtered quality history is not available for the full range yet. Select a date range or refresh to populate class-aware daily facts.",
                    icon="fas fa-database",
                )
                return empty, html.Div(), html.Div(), html.Div()
            if not quality:
                empty_message = (
                    "Filtered quality history is unavailable until the next refresh populates class-aware daily facts."
                    if class_filter_active
                    else "No quality snapshot available yet. Run a refresh first."
                )
                empty = make_empty_state(empty_message, icon="fas fa-database")
                return empty, html.Div(), html.Div(), html.Div()
            quality_history = backend.get_quality_history(
                model_id,
                start_date=start_date,
                end_date=end_date,
                class_basis=class_basis,
                class_value=class_value,
            )
            null_rate_history = backend.get_null_rate_history(
                model_id,
                start_date=start_date,
                end_date=end_date,
                class_basis=class_basis,
                class_value=class_value,
            )
            history_note = _comparison_history_message(len(quality_history), "daily")
            filter_summary = _analysis_filter_summary(
                start_date=start_date,
                end_date=end_date,
                class_basis=class_basis,
                class_value=class_value,
            )
            kpis = [
                dbc.Col(
                    make_metric_card(
                        "Monitoring Rows",
                        f"{quality['total_rows']:,}",
                        "Rows covered by persisted monitoring history",
                    ),
                    md=3,
                ),
                dbc.Col(make_metric_card("From", quality["min_date"] or "—", "Earliest data"), md=3),
                dbc.Col(make_metric_card("To", quality["max_date"] or "—", "Latest data"), md=3),
                dbc.Col(make_metric_card("Prediction Average", _format_metric_value(quality["prediction_mean"]), "Latest snapshot"), md=3),
                dbc.Col(make_metric_card("Prediction Std", _format_metric_value(quality["prediction_std"]), "Latest snapshot"), md=3),
            ]
            volume_children_items: list[object] = []
            if history_note:
                volume_children_items.append(_status_alert(history_note, "info"))
            if filter_summary:
                volume_children_items.append(_status_alert(f"Active filters: {filter_summary}", "secondary"))
            volume_children_items.append(
                html.Small(
                    "Daily Monitoring Rows shows daily row volume in persisted monitoring history. Rows Per Comparison Window shows the row volume in each baseline/current comparison window.",
                    className="text-muted d-block mb-2",
                )
            )
            volume_children_items.append(
                html.Small(
                    "A comparison window is the persisted current window paired with its matching baseline window for one refresh cycle.",
                    className="text-muted d-block mb-2",
                )
            )
            volume_children_items.append(
                dbc.Row(
                    [
                        dbc.Col(make_chart_card(charts.build_volume_timeline(quality["daily_volume"])), md=6),
                        dbc.Col(make_chart_card(charts.build_quality_window_timeline(quality_history)), md=6),
                    ],
                    className="g-3",
                )
            )
            volume_children = html.Div(volume_children_items)
            null_children = html.Div(
                [
                    make_chart_card(
                        charts.build_null_rate_chart(
                            quality["null_rates"],
                            show_thresholds=bool(show_thresholds),
                            thresholds=resolved_thresholds,
                        )
                    ),
                    make_chart_card(charts.build_null_rate_timeline(null_rate_history), class_name="mb-0"),
                ]
            )
            latest_window_metrics = backend.get_latest_window_metrics(model_id)
            prediction_children = html.Div(
                [
                    make_chart_card(charts.build_prediction_quality_timeline(quality_history)),
                    make_chart_card(
                        charts.build_latest_window_metric_snapshot(latest_window_metrics),
                        class_name="mb-0",
                    ),
                ]
            )
            return (
                kpis,
                volume_children,
                null_children,
                prediction_children,
            )
        except Exception as error:
            logger.exception("Failed to render data quality", exc_info=error)
            return _callback_error_panel("data quality", error, icon="fas fa-database"), html.Div(), html.Div(), html.Div()

    @app.callback(
        Output("perf-metric-select", "options"),
        Output("perf-metric-select", "value"),
        Input("global-model-select", "value"),
        Input("session-config-store", "data"),
        State("perf-metric-select", "value"),
    )
    def sync_performance_metric_options(model_id, session_data, current_metric):
        try:
            backend = _make_backend(session_data)
            default_metric = _default_performance_metric(model_id, backend)
            config = backend.get_monitor_config(model_id) if model_id else None
            if config:
                options = [
                    {
                        "label": performance_metric_label(metric_name),
                        "value": metric_name,
                    }
                    for metric_name in _configured_performance_metric_names(config)
                ]
            else:
                options = performance_metric_options("classification")
            valid_values = {option["value"] for option in options}
            value = current_metric if current_metric in valid_values else default_metric
            return options, value
        except Exception as error:
            logger.exception("Failed to sync performance metric options", exc_info=error)
            options = performance_metric_options("classification")
            return options, current_metric if current_metric in {option["value"] for option in options} else "f1"

    @app.callback(
        Output("perf-drift-feature-select", "options"),
        Output("perf-drift-feature-select", "value"),
        Input("global-model-select", "value"),
        Input("perf-drift-metric-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("perf-drift-feature-select", "value"),
    )
    def sync_performance_drift_features(model_id, drift_metric, _, session_data, current_values):
        if not model_id:
            return [], []
        try:
            backend = _make_backend(session_data)
            drift = backend.get_drift_results(model_id, granularity="daily")
            if drift.empty or "feature" not in drift.columns:
                return [], []
            metric_key = str(drift_metric or "psi").strip().lower() or "psi"
            if metric_key in drift.columns:
                ranked = _historical_drift_feature_ranking(
                    drift,
                    metric=metric_key,
                    top_n=max(int(drift["feature"].nunique()), 1),
                )
                ordered_features = ranked["feature"].astype(str).tolist()
            else:
                ordered_features = drift["feature"].dropna().astype(str).drop_duplicates().tolist()
            options = _option_list(ordered_features)
            valid_values = {option["value"] for option in options}
            requested_values = current_values if isinstance(current_values, list) else ([current_values] if current_values else [])
            selected = [str(value) for value in requested_values if str(value) in valid_values]
            if not selected:
                selected = ordered_features if len(ordered_features) <= 12 else ordered_features[:12]
            return options, selected
        except Exception as error:
            logger.exception("Failed to sync performance drift features", exc_info=error)
            return [], []

    @app.callback(
        Output("perf-bin-count-input", "disabled"),
        Output("perf-custom-edges-input", "disabled"),
        Output("perf-outlier-value-input", "disabled"),
        Output("perf-outlier-value-label", "children"),
        Output("perf-outlier-value-help", "children"),
        Input("perf-binning-mode-select", "value"),
        Input("perf-outlier-mode-select", "value"),
    )
    def sync_performance_breakdown_controls(binning_mode, outlier_mode):
        normalized_mode = str(binning_mode or "auto").strip().lower()
        normalized_outlier_mode = _normalize_outlier_mode(outlier_mode)
        if normalized_outlier_mode == "percentile_clip":
            return (
                normalized_mode != "fixed",
                normalized_mode != "custom",
                False,
                "Trim Percentile P",
                "Percentile Clip keeps values between P and 100-P.",
            )
        if normalized_outlier_mode == "iqr_fence":
            return (
                normalized_mode != "fixed",
                normalized_mode != "custom",
                False,
                "IQR Multiplier K",
                "IQR Fence keeps values inside Q1 - K*IQR and Q3 + K*IQR.",
            )
        return (
            normalized_mode != "fixed",
            normalized_mode != "custom",
            True,
            "Outlier Parameter",
            "Percentile Clip uses P / 100-P clipping. IQR Fence uses Q1 - K*IQR to Q3 + K*IQR.",
        )

    @app.callback(
        Output("perf-labels-alert", "children"),
        Output("perf-kpi-cards", "children"),
        Output("perf-timeline-container", "children"),
        Output("perf-contributors-container", "children"),
        Output("perf-feature-select", "options"),
        Output("perf-feature-select", "value"),
        Output("perf-date-range-note", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("perf-metric-select", "value"),
        Input("perf-drift-metric-select", "value"),
        Input("perf-drift-threshold-toggle", "value"),
        Input("perf-drift-feature-select", "value"),
        Input("perf-apply-breakdown-controls-btn", "n_clicks"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("perf-binning-mode-select", "value"),
        State("perf-bin-count-input", "value"),
        State("perf-custom-edges-input", "value"),
        State("perf-outlier-mode-select", "value"),
        State("perf-outlier-value-input", "value"),
        State("perf-feature-select", "value"),
    )
    def render_performance(
        pathname,
        model_id,
        metric_name,
        drift_metric,
        perf_show_thresholds,
        current_drift_features,
        _apply_breakdown_clicks,
        _reload_token,
        session_data,
        binning_mode,
        bin_count,
        custom_edges_text,
        outlier_mode,
        outlier_value_raw,
        current_feature,
    ):
        if pathname != "/performance":
            return (no_update,) * 7
        try:
            if not model_id:
                empty = make_empty_state("Select a model to inspect performance.", icon="fas fa-tachometer-alt")
                return empty, html.Div(), html.Div(), html.Div(), [], None, html.Div()
            backend = _make_backend(session_data)
            config = backend.get_monitor_config(model_id)
            resolved_thresholds = _resolved_monitor_thresholds(config)
            if not config or not config.contract.label_col:
                return (
                    _status_alert("This monitor does not have labels configured, so performance degradation analysis is unavailable.", "warning"),
                    html.Div(),
                    html.Div(),
                    html.Div(),
                    [],
                    None,
                    html.Div(),
                )
            resolved_metric = metric_name or _default_performance_metric(model_id, backend)
            normalized_binning_mode = str(binning_mode or "auto").strip().lower()
            custom_edges = _parse_custom_edges(custom_edges_text) if normalized_binning_mode == "custom" else None
            normalized_outlier_mode = _normalize_outlier_mode(outlier_mode)
            outlier_value = _outlier_control_value(normalized_outlier_mode, outlier_value_raw)
            performance = backend.get_performance_summary(model_id, metric_name=resolved_metric)
            latest_bins = performance["latest_bins"]
            all_bins = performance.get("all_bins", pd.DataFrame())
            if all_bins.empty:
                return (
                    _status_alert("No performance metrics available yet. Run a refresh after labels arrive.", "warning"),
                    html.Div(),
                    html.Div(),
                    html.Div(),
                    [],
                    None,
                    html.Div(),
                )
            timeline_metric_names = _configured_performance_metric_names(config)
            classification_timeline_metrics = [
                current_metric
                for current_metric in timeline_metric_names
                if current_metric in {"precision", "recall", "f1"}
            ]
            if classification_timeline_metrics and str(getattr(config, "problem_type", "classification") or "classification").strip().lower() == "classification":
                timeline_metric_names = classification_timeline_metrics
            else:
                timeline_metric_names = [resolved_metric]
            timeline_summaries = {resolved_metric: performance}
            for current_metric in timeline_metric_names:
                if current_metric == resolved_metric:
                    continue
                timeline_summaries[current_metric] = backend.get_performance_summary(model_id, metric_name=current_metric)
            timeline_map: dict[str, dict[str, object]] = {}
            for current_metric, summary in timeline_summaries.items():
                for row in summary.get("timeline", []):
                    raw_period = row.get("period")
                    period_text = str(raw_period or "").strip()
                    if not period_text:
                        continue
                    parsed_period = pd.to_datetime(period_text, errors="coerce")
                    if pd.isna(parsed_period):
                        continue
                    normalized_period = str(pd.Timestamp(parsed_period).date())
                    entry = timeline_map.setdefault(normalized_period, {"period": normalized_period})
                    entry[current_metric] = row.get(current_metric)
            combined_timeline = [
                timeline_map[period]
                for period in sorted(
                    timeline_map,
                    key=lambda value: pd.to_datetime(value, errors="coerce"),
                )
            ]
            contributors = performance["contributors"]
            feature_frame = latest_bins if not latest_bins.empty else all_bins
            exact_breakdown = backend.get_exact_performance_breakdown(
                model_id,
                metric_name=resolved_metric,
                binning_mode=normalized_binning_mode,
                n_bins=_normalize_top_n(bin_count, default=40, minimum=2, maximum=200),
                custom_edges=custom_edges,
                outlier_mode=normalized_outlier_mode,
                outlier_value=outlier_value,
            )
            breakdown_rows = exact_breakdown.get("rows", pd.DataFrame())
            breakdown_contributors = exact_breakdown.get("contributors", pd.DataFrame())
            breakdown_message = str(exact_breakdown.get("message") or "").strip()
            if not breakdown_rows.empty:
                feature_frame = breakdown_rows
                contributors = breakdown_contributors
            features = sorted({str(value) for value in (config.contract.feature_columns or []) if str(value).strip()})
            if not features:
                features = sorted(
                    {
                        str(value)
                        for value in feature_frame.get("feature", pd.Series(dtype=str)).dropna().tolist()
                        if str(value).strip()
                    }
                )
            feature_options = _option_list(features)
            feature_values = {option["value"] for option in feature_options}
            selected_feature = current_feature if current_feature in feature_values else (feature_options[0]["value"] if feature_options else None)
            degradation_detected = bool(performance.get("has_significant_degradation"))
            alert_children: list[object] = [
                html.Small(
                    f"Feature impact metric: {performance_metric_label(resolved_metric)}",
                    className="text-muted d-block mb-2",
                )
            ]
            timeline_reasons: list[str] = []
            for current_metric in timeline_metric_names:
                current_reason = str(timeline_summaries.get(current_metric, {}).get("timeline_unavailable_reason") or "").strip()
                if current_reason and current_reason not in timeline_reasons:
                    timeline_reasons.append(current_reason)
            for current_reason in timeline_reasons:
                alert_children.append(_status_alert(current_reason, "warning"))
            if breakdown_message and breakdown_rows.empty:
                alert_children.append(
                    _status_alert(
                        f"{breakdown_message} Showing the stored latest-window breakdown instead.",
                        "warning",
                    )
                )
            if not degradation_detected:
                alert_children.append(
                    _status_alert(
                        "Performance metrics are populated, but no significant degradation is detected in the latest window.",
                        "info",
                    )
                )
            alert = html.Div(alert_children)
            latest_bin_table = pd.DataFrame()
            if not feature_frame.empty:
                latest_bin_table = feature_frame[
                    [
                        column
                        for column in (
                            "feature",
                            "bin_label",
                            "baseline_metric",
                            "current_metric",
                            "delta",
                            "current_volume_pct",
                            "degradation_contribution",
                        )
                        if column in feature_frame.columns
                    ]
                ].rename(
                    columns={
                        "bin_label": "bin",
                        "baseline_metric": "baseline",
                        "current_metric": "current",
                        "current_volume_pct": "volume_pct",
                        "degradation_contribution": "impact",
                    }
                )
            kpi_cards = [
                dbc.Col(make_metric_card("Tracked Features", str(len(feature_options)), "With labeled performance bins"), md=4),
                dbc.Col(
                    make_metric_card(
                        "Worst Weighted Delta",
                        f"{float(performance.get('worst_weighted_delta', 0.0)):.4f}",
                        "Most degraded feature" if degradation_detected else "Stable latest window",
                    ),
                    md=4,
                ),
                dbc.Col(make_metric_card("Windows", str(len(combined_timeline)), "Historical performance snapshots"), md=4),
            ]
            selected_drift_metric = str(drift_metric or "psi").strip().lower() or "psi"
            drift = backend.get_drift_results(model_id, granularity="daily") if hasattr(backend, "get_drift_results") else pd.DataFrame()
            drift_ranking = (
                drift.assign(_metric_rank=pd.to_numeric(drift.get(selected_drift_metric, pd.Series(dtype=float)), errors="coerce").fillna(0.0))
                .groupby("feature", as_index=False)["_metric_rank"]
                .max()
                .sort_values("_metric_rank", ascending=False)
                if isinstance(drift, pd.DataFrame) and not drift.empty
                else pd.DataFrame(columns=["feature", "_metric_rank"])
            )
            drift_feature_order = drift_ranking["feature"].astype(str).tolist() if not drift_ranking.empty else []
            drift_feature_values = set(drift_feature_order)
            requested_drift_features = current_drift_features if isinstance(current_drift_features, list) else ([current_drift_features] if current_drift_features else [])
            drift_features = [str(value) for value in requested_drift_features if str(value) in drift_feature_values]
            if not drift_features:
                drift_features = drift_feature_order if len(drift_feature_order) <= 12 else drift_feature_order[:12]
            drift_metric_label = _THRESHOLD_LABELS.get(selected_drift_metric, str(selected_drift_metric or "psi").upper())
            drift_scope_label = (
                "All Tracked Features"
                if drift_feature_order and len(drift_features) == len(drift_feature_order)
                else "Selected Drift Features"
            )
            note_source = latest_bins if not latest_bins.empty else all_bins
            note = note_source[[column for column in ("window_start", "window_end") if column in note_source.columns]].drop_duplicates().astype(str)
            note_row = note.to_dict("records")[0] if not note.empty else {}
            note_parts: list[object] = []
            note_text = ""
            if "window_start" in note_row and "window_end" in note_row:
                note_text = f"Latest comparison window: {note_row['window_start']} to {note_row['window_end']}"
            elif "window_end" in note_row:
                note_text = f"Latest comparison window end: {note_row['window_end']}"
            elif "window_start" in note_row:
                note_text = f"Latest comparison window start: {note_row['window_start']}"
            if note_text:
                note_parts.append(html.Small(note_text, className="text-muted d-block"))
            history_message = _comparison_history_message(len(combined_timeline), "daily")
            if history_message:
                note_parts.append(html.Small(history_message, className="text-muted d-block"))
            if any(
                pd.isna(row.get(current_metric))
                for row in combined_timeline
                for current_metric in timeline_metric_names
                if current_metric in row
            ):
                note_parts.append(
                    html.Small(
                        "Chart gaps mean the metric was undefined on those days, not zero.",
                        className="text-muted d-block",
                    )
                )
            impact_help_button = dbc.Button(
                html.I(className="fas fa-circle-question"),
                id="perf-feature-impact-help-btn",
                color="link",
                className="p-0 text-info text-decoration-none",
            )
            impact_help = dbc.Popover(
                [
                    dbc.PopoverHeader("How to Read Feature Impact"),
                    dbc.PopoverBody(
                        [
                            html.Div(
                                [
                                    html.Div(style={"width": "38px", "height": "10px", "backgroundColor": "#e74c3c", "display": "inline-block", "marginRight": "8px"}),
                                    html.Span("Long red bars hurt the selected metric most."),
                                ],
                                className="mb-2",
                            ),
                            html.Div(
                                [
                                    html.Div(style={"width": "38px", "height": "10px", "backgroundColor": "#2ecc71", "display": "inline-block", "marginRight": "8px"}),
                                    html.Span("Long green bars help the selected metric most."),
                                ],
                                className="mb-2",
                            ),
                            html.P("Baseline and Current are the slice-level metric values. Delta is their change. Current Window Share shows how much of the latest traffic sits in that slice. Weighted Contribution is the slice's share-weighted pull on the overall metric.", className="mb-2"),
                            html.P("The color bands come from raw slice delta thresholds. The x-axis is weighted contribution, so use the vertical zero line as the absolute visual guide.", className="mb-2"),
                            html.P("The center line at 0 means no net contribution. Right side is worse. Left side is better.", className="mb-0"),
                        ]
                    ),
                ],
                target="perf-feature-impact-help-btn",
                trigger="click",
                placement="auto",
            )
            latest_bin_table = latest_bin_table.rename(
                columns={
                    "feature": "Feature",
                    "bin": "Bin",
                        "baseline": "Baseline",
                        "current": "Current",
                        "delta": "Delta",
                        "volume_pct": "Current Window Share (%)",
                        "impact": "Weighted Contribution (Delta x Share)",
                    }
                )
            return (
                alert,
                kpi_cards,
                make_chart_card(
                    charts.build_performance_timeline(
                        combined_timeline,
                        metric_name=resolved_metric,
                        metric_names=timeline_metric_names,
                    )
                ),
                html.Div(
                    [
                        html.H6("Drift vs Time", className="text-light mt-3 mb-2"),
                        html.P(
                            f"Compare the {drift_metric_label} trend below with the performance metrics above to spot time-aligned drift and metric shifts. Showing {len(drift_features)} of {len(drift_feature_order) or len(drift_features)} tracked features.",
                            className="text-muted",
                            style={"fontSize": "0.8rem"},
                        ),
                        make_chart_card(
                            charts.build_drift_timeline(
                                drift,
                                drift_features,
                                metric=selected_drift_metric,
                                show_thresholds=bool(perf_show_thresholds),
                                thresholds=resolved_thresholds,
                                title=f"{drift_metric_label} Over Time ({drift_scope_label})",
                            ),
                            class_name="mb-3",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6("Feature Impact on Performance", className="text-light mb-0"),
                                        impact_help_button,
                                    ],
                                    className="d-flex align-items-center gap-2 mb-2",
                                ),
                                impact_help,
                                make_chart_card(charts.build_feature_bin_impact(feature_frame, contributors)),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6("Latest Bin Metrics", className="text-light mb-0"),
                                        dbc.Button(
                                            html.I(className="fas fa-circle-question"),
                                            id="perf-latest-bin-help-btn",
                                            color="link",
                                            className="p-0 text-info text-decoration-none",
                                        ),
                                    ],
                                    className="d-flex align-items-center gap-2 mb-2",
                                ),
                                html.Small(
                                    "Weighted Contribution = Delta x Current Window Share for that slice in the latest comparison window.",
                                    className="text-muted d-block mb-2",
                                ),
                                dbc.Popover(
                                    dbc.PopoverBody(
                                        "Weighted Contribution is the slice's share-weighted pull on the overall metric in the latest comparison window. It combines how much the slice moved (Delta) with how much traffic the slice currently owns (Current Window Share).",
                                    ),
                                    target="perf-latest-bin-help-btn",
                                    trigger="click",
                                    placement="auto",
                                ),
                                _render_frame(latest_bin_table, "No bin-level performance data available."),
                            ]
                        ),
                    ]
                ),
                feature_options,
                selected_feature,
                html.Div(note_parts),
            )
        except Exception as error:
            logger.exception("Failed to render performance analysis", exc_info=error)
            return _status_alert(_callback_error_message("performance analysis", error), "danger"), html.Div(), html.Div(), html.Div(), [], None, html.Div()

    @app.callback(
        Output("perf-bin-detail-container", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("perf-feature-select", "value"),
        Input("perf-metric-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def render_perf_bin_detail(pathname, model_id, feature, metric_name, _, session_data):
        if pathname != "/performance":
            return no_update
        if not model_id or not feature:
            return html.Div("Select a feature to open it in Feature Deep Dive.", className="text-muted")
        try:
            href = f"/features?{urlencode({'model': model_id, 'feature': feature})}"
            return html.Div(
                [
                    html.Small(
                        "Feature Deep Dive lets you inspect this selected feature directly after adjusting the Performance-page breakdown controls above.",
                        className="text-muted d-block mb-2",
                    ),
                    dbc.Button(
                        f"Open {feature} in Feature Deep Dive",
                        href=href,
                        color="secondary",
                    ),
                ]
            )
        except Exception as error:
            return _callback_error_panel("feature deep dive shortcut", error)

    @app.callback(
        Output("reference-page-body", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("reference-monitor-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def render_reference(pathname, global_model_id, reference_model_id, _, session_data):
        if pathname != "/reference":
            return no_update
        try:
            backend = _make_backend(session_data)
            model_id = _resolve_reference_model_id(global_model_id, reference_model_id)
            if not model_id:
                return make_empty_state("Select a model to inspect the contract and runtime state.", icon="fas fa-book")
            data = backend.get_reference_data(model_id)
            config = data["config"]
            if not config:
                return make_empty_state("Selected model is no longer available.", icon="fas fa-book")
        except Exception as error:
            return _callback_error_panel("monitor settings", error, icon="fas fa-book")
        config_status = getattr(config, "status", "active")
        cadence_labels = {
            "hourly": "Hourly",
            "6h": "Every 6 Hours",
            "daily": "Daily",
            "manual": "Manual Only",
            "disabled": "Disabled",
            "6h_3d_repair": "Every 6 Hours (3-Day Repair)",
            "daily_7d_repair": "Daily (7-Day Repair)",
            "daily_14d_repair": "Daily (14-Day Repair)",
        }
        contract_frame = pd.DataFrame(
            [
                {"field": "timestamp_col", "value": config.contract.timestamp_col},
                {"field": "model_id_col", "value": config.contract.model_id_col or ""},
                {"field": "model_id_value", "value": config.model_id_value or ""},
                {"field": "prediction_col", "value": config.contract.prediction_col},
                {"field": "model_version_col", "value": config.contract.model_version_col or ""},
                {"field": "model_version_value", "value": config.model_version_value or ""},
                {"field": "label_col", "value": config.contract.label_col or ""},
                {"field": "labels_table", "value": config.labels_table or ""},
                {"field": "labels_join_col", "value": config.labels_join_col or ""},
                {"field": "labels_order_col", "value": config.labels_order_col or ""},
                {"field": "mlflow_experiment_name", "value": config.mlflow.experiment_name or ""},
                {"field": "mlflow_experiment_id", "value": config.mlflow.experiment_id or ""},
                {"field": "mlflow_run_id", "value": config.mlflow.run_id or ""},
                {"field": "mlflow_registered_model_name", "value": config.mlflow.registered_model_name or ""},
                {"field": "mlflow_model_version", "value": config.mlflow.model_version or ""},
                {"field": "feature_columns", "value": ", ".join(config.contract.feature_columns)},
                {"field": "categorical_columns", "value": ", ".join(config.contract.categorical_columns)},
                {"field": "slice_columns", "value": ", ".join(config.contract.slice_columns)},
                {"field": "baseline_kind", "value": config.baseline.kind},
                {"field": "baseline_days", "value": config.baseline.n_days},
                {"field": "baseline_start", "value": config.baseline.baseline_start or ""},
                {"field": "baseline_end", "value": config.baseline.baseline_end or ""},
                {"field": "problem_type", "value": config.problem_type},
                {
                    "field": "performance_metric_names",
                    "value": ", ".join(
                        performance_metric_label(metric_name)
                        for metric_name in _configured_performance_metric_names(config)
                    ),
                },
                {
                    "field": "default_performance_metric",
                    "value": performance_metric_label(_configured_default_performance_metric(config)),
                },
                {"field": "drift_cadence_preset", "value": config.drift_cadence_preset},
                {"field": "performance_cadence_preset", "value": config.performance_cadence_preset},
                {"field": "schedule_enabled", "value": config.schedule_enabled},
            ]
        )
        summary_frame = pd.DataFrame([data["summary"]]) if data["summary"] else pd.DataFrame()
        runtime_state = data.get("runtime_state") or {}
        runtime_frame = pd.DataFrame(
            [{"field": key, "value": value} for key, value in runtime_state.items()]
        )
        recent_runs = data.get("recent_runs") or []
        refresh_diagnostics = data.get("refresh_diagnostics") or {}
        recent_runs_frame = (
            pd.DataFrame(recent_runs)[
                [
                    column
                    for column in (
                        "scope",
                        "status",
                        "started_at",
                        "completed_at",
                        "total_duration_ms",
                        "error_message",
                        "source_metadata_ms",
                        "daily_profiles_ms",
                        "derivation_ms",
                        "persistence_ms",
                        "rows_scanned",
                        "label_rows_scanned",
                    )
                    if recent_runs and column in recent_runs[0]
                ]
            ]
            if recent_runs
            else pd.DataFrame()
        )
        if not recent_runs_frame.empty and "total_duration_ms" in recent_runs_frame.columns:
            recent_runs_frame = recent_runs_frame.rename(
                columns={
                    "scope": "Scope",
                    "status": "Status",
                    "started_at": "Started At",
                    "completed_at": "Completed At",
                    "total_duration_ms": "Total Duration",
                    "error_message": "Error",
                    "source_metadata_ms": "Source Metadata",
                    "daily_profiles_ms": "Daily Profiles",
                    "derivation_ms": "Derivation",
                    "persistence_ms": "Persistence",
                    "rows_scanned": "Rows Scanned",
                    "label_rows_scanned": "Label Rows",
                }
            )
            for column in ("Total Duration", "Source Metadata", "Daily Profiles", "Derivation", "Persistence"):
                if column in recent_runs_frame.columns:
                    recent_runs_frame[column] = recent_runs_frame[column].apply(_format_duration_ms)
        recent_incident_history = data.get("recent_incident_history") or []
        recent_incident_history_frame = (
            pd.DataFrame(recent_incident_history)[
                [
                    column
                    for column in (
                        "event_type",
                        "feature_name",
                        "metric_name",
                        "severity",
                        "status",
                        "metric_value",
                        "window_end",
                        "observed_at",
                    )
                    if recent_incident_history and column in recent_incident_history[0]
                ]
            ]
            if recent_incident_history
            else pd.DataFrame()
        )
        settings_frame = pd.DataFrame(
            [
                {"field": field, "value": _format_runtime_setting_value(field, value)}
                for field, value in data["settings"].items()
                if field != "shared_schedule"
            ]
        )
        shared_schedule = (data["settings"] or {}).get("shared_schedule") or {}
        resolved_thresholds = _resolved_monitor_thresholds(config)
        configured_refresh_job_id = str(data["settings"].get("refresh_job_id") or "").strip()
        configured_refresh_job_name = str(data["settings"].get("refresh_job_name") or "").strip()
        configured_bootstrap_job_id = str(data["settings"].get("bootstrap_refresh_job_id") or "").strip()
        configured_bootstrap_job_name = str(data["settings"].get("bootstrap_refresh_job_name") or "").strip()
        if configured_refresh_job_id:
            refresh_job_wiring_text = (
                f"This app is configured to trigger shared refresh job ID {configured_refresh_job_id}. "
                "REFRESH_JOB_ID and REFRESH_JOB_NAME are deploy-time app environment variables, not onboarding inputs. "
                "To change them, update app.yaml or the generated manual existing-app app.yaml and redeploy the app."
            )
        else:
            configured_job_name = configured_refresh_job_name or "<unset>"
            refresh_job_wiring_text = (
                f"This app currently resolves the shared refresh workflow by name using REFRESH_JOB_NAME={configured_job_name!r}. "
                "REFRESH_JOB_ID is preferred because it avoids name-matching issues. "
                "To change either value, update app.yaml or the generated manual existing-app app.yaml and redeploy the app."
            )
        if configured_bootstrap_job_id:
            refresh_job_wiring_text += (
                f" Bootstrap and backfill triggers are routed to BOOTSTRAP_REFRESH_JOB_ID={configured_bootstrap_job_id} when that optional override is configured."
            )
        elif configured_bootstrap_job_name:
            refresh_job_wiring_text += (
                f" Bootstrap and backfill triggers are routed to BOOTSTRAP_REFRESH_JOB_NAME={configured_bootstrap_job_name!r} when that optional override is configured."
            )
        else:
            refresh_job_wiring_text += " Bootstrap and backfill triggers use the shared refresh workflow by default."
        show_bootstrap_retry = config_status == "active" and str(runtime_state.get("bootstrap_status") or "pending") != "completed"
        lifecycle_buttons: list[dbc.Col] = []
        if config_status == "active":
            lifecycle_buttons.append(
                dbc.Col(
                    dbc.Button(
                        "Archive Monitor",
                        id="reference-archive-monitor-btn",
                        color="warning",
                        outline=True,
                        className="w-100",
                    ),
                    md=4,
                )
            )
        if config_status == "inactive":
            lifecycle_buttons.append(
                dbc.Col(
                    dbc.Button(
                        "Restore Monitor",
                        id="reference-restore-monitor-btn",
                        color="success",
                        outline=True,
                        className="w-100",
                    ),
                    md=4,
                )
            )
        lifecycle_buttons.append(
            dbc.Col(
                dbc.Button(
                    "Delete Monitor And History",
                    id="reference-delete-monitor-btn",
                    color="danger",
                    outline=True,
                    className="w-100",
                ),
                md=4,
            )
        )
        archive_modal = (
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Archive Monitor")),
                    dbc.ModalBody(
                        [
                            html.P(
                                (
                                    f"Archive the selected monitor {config.display_name} ({config.model_key})? "
                                    "Scheduled refreshes will stop, but the persisted monitoring history stored under this model key will be kept."
                                ),
                                className="mb-0",
                            )
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Cancel", id="reference-archive-cancel-btn", color="secondary", outline=True),
                            dbc.Button("Archive", id="reference-archive-confirm-btn", color="warning"),
                        ]
                    ),
                ],
                id="reference-archive-modal",
                is_open=False,
            )
            if config_status == "active"
            else None
        )
        threshold_controls = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(_THRESHOLD_LABELS[metric], className="text-light mb-3"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Warning", className="text-muted"),
                                                dbc.Input(
                                                    id=_threshold_input_id(metric, "warning"),
                                                    type="number",
                                                    min=0,
                                                    step=0.01 if metric == "null_rate" else 0.001,
                                                    value=get_thresholds(metric, resolved_thresholds)[0],
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Critical", className="text-muted"),
                                                dbc.Input(
                                                    id=_threshold_input_id(metric, "critical"),
                                                    type="number",
                                                    min=0,
                                                    step=0.01 if metric == "null_rate" else 0.001,
                                                    value=get_thresholds(metric, resolved_thresholds)[1],
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ],
                                    className="g-2",
                                ),
                            ]
                        ),
                        className="h-100",
                    ),
                    md=6,
                )
                for metric in THRESHOLD_METRICS
            ],
            className="g-3 mt-1",
        )
        schedule_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Refresh Cadence", className="text-light mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Drift And Quality"),
                                    dbc.Select(
                                        id="reference-drift-cadence-select",
                                        options=[
                                            {"label": cadence_labels[value], "value": value}
                                            for value in DRIFT_CADENCE_PRESETS
                                        ],
                                        value=config.drift_cadence_preset,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Performance And Label Repair"),
                                    dbc.Select(
                                        id="reference-performance-cadence-select",
                                        options=[
                                            {"label": cadence_labels[value], "value": value}
                                            for value in PERFORMANCE_CADENCE_PRESETS
                                        ],
                                        value=config.performance_cadence_preset,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Schedule"),
                                    dbc.Checklist(
                                        id="reference-schedule-enabled-toggle",
                                        options=[{"label": "Enabled", "value": "enabled"}],
                                        value=["enabled"] if config.schedule_enabled else [],
                                        switch=True,
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="g-3",
                    ),
                    dbc.Alert(
                        "Threshold overrides apply immediately to Overview severity, Drift/Data Quality threshold guides, and future incident generation. Historical incident history is preserved as recorded.",
                        color="secondary",
                        className="py-2 mt-3 mb-0",
                    ),
                    html.H6("Thresholds", className="text-light mt-4 mb-3"),
                    threshold_controls,
                    html.H6("Performance Metrics", className="text-light mt-4 mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Tracked Performance Metrics"),
                                    dcc.Dropdown(
                                        id="reference-performance-metrics-select",
                                        options=performance_metric_options(config.problem_type),
                                        value=_configured_performance_metric_names(config),
                                        multi=True,
                                        className="dash-dropdown",
                                    ),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Default Performance Metric"),
                                    dbc.Select(
                                        id="reference-default-performance-metric-select",
                                        options=[
                                            {
                                                "label": performance_metric_label(metric_name),
                                                "value": metric_name,
                                            }
                                            for metric_name in _configured_performance_metric_names(config)
                                        ],
                                        value=_configured_default_performance_metric(config),
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="g-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Performance Degradation Binning"),
                                    dbc.Select(
                                        id="reference-performance-binning-mode-select",
                                        options=[
                                            {"label": "Quantile (Recommended)", "value": "quantile"},
                                            {"label": "Fixed Width", "value": "fixed_width"},
                                        ],
                                        value=getattr(config, "performance_binning_mode", "quantile"),
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Optional Percentile Clip"),
                                    dbc.Input(
                                        id="reference-performance-binning-clip-input",
                                        type="number",
                                        min=0.1,
                                        max=49.9,
                                        step=0.1,
                                        placeholder="Off",
                                        value=getattr(config, "performance_binning_clip_percentile", None),
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="g-3 mt-1",
                    ),
                    dbc.Alert(
                        "Quantile binning is the default for performance degradation because it is more robust to skewed distributions. Optional percentile clipping caps edge generation to P / 100-P while still assigning extreme values into the outer buckets.",
                        color="secondary",
                        className="py-2 mt-3 mb-0",
                    ),
                    dbc.Button("Save Monitor Settings", id="reference-save-schedule-btn", color="primary", className="mt-3"),
                    (
                        html.Div(
                            [
                                html.Hr(className="border-secondary mt-4"),
                                html.H6("Initial Refresh", className="text-light mb-2"),
                                html.P(
                                    "If this monitor is still pending bootstrap, trigger the shared refresh workflow again for this selected monitor only. This uses bootstrap scope and does not change the shared job schedule.",
                                    className="text-muted mb-3",
                                ),
                                dbc.Button(
                                    "Run Initial Refresh Now",
                                    id="reference-run-bootstrap-btn",
                                    color="secondary",
                                    outline=True,
                                ),
                            ]
                        )
                        if show_bootstrap_retry
                        else html.Div()
                    ),
                ]
            ),
            className="mb-4",
        )
        shared_schedule_value = (
            int(shared_schedule.get("current_interval_hours"))
            if shared_schedule.get("current_interval_hours") in SCHEDULE_INTERVAL_OPTIONS
            else 1
        )
        shared_schedule_editable = bool(shared_schedule.get("editable"))
        shared_schedule_alerts: list[object] = []
        for issue in shared_schedule.get("blocking_issues") or []:
            shared_schedule_alerts.append(_status_alert(str(issue), "warning"))
        for warning_text in shared_schedule.get("warnings") or []:
            shared_schedule_alerts.append(_status_alert(str(warning_text), "secondary"))
        if not shared_schedule_editable:
            if shared_schedule.get("management_available") is False:
                shared_schedule_alerts.append(
                    _status_alert(
                        "This app identity can read the shared workflow schedule but cannot edit it. Grant CAN_MANAGE on the shared refresh job to enable in-app changes.",
                        "warning",
                    )
                )
            elif not shared_schedule.get("supported"):
                shared_schedule_alerts.append(
                    _status_alert(
                        "The shared refresh job is using a custom or unsupported scheduler mode. Update it externally if you need a different wake interval.",
                        "secondary",
                    )
                )
            else:
                shared_schedule_alerts.append(
                    _status_alert(
                        "The app could not confirm schedule-edit permission. You can still update the shared refresh job externally if needed.",
                        "secondary",
                    )
                )
        shared_schedule_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Shared Workflow Schedule", className="text-light mb-3"),
                    html.P(
                        "This is the shared refresh job wake-up interval. It controls how often the scheduler checks for due monitors. Per-monitor cadence still decides whether this monitor actually runs.",
                        className="text-muted mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Current Detected Schedule"),
                                    html.Div(
                                        shared_schedule.get("current_label") or "Unavailable",
                                        className="text-light fw-semibold",
                                    ),
                                    html.Small(
                                        (
                                            f"Timezone: {shared_schedule.get('timezone_id') or 'UTC'}"
                                            + (" | Paused" if shared_schedule.get("paused") else "")
                                            + (
                                                f" | Last checked: {shared_schedule.get('checked_at')}"
                                                if shared_schedule.get("checked_at")
                                                else ""
                                            )
                                        ),
                                        className="text-muted d-block mt-1",
                                    ),
                                ],
                                md=5,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Wake Interval"),
                                    dbc.Select(
                                        id="reference-shared-schedule-select",
                                        options=_schedule_interval_options(),
                                        value=shared_schedule_value,
                                        disabled=not shared_schedule_editable,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Apply"),
                                    dbc.Button(
                                        "Save Shared Schedule",
                                        id="reference-save-shared-schedule-btn",
                                        color="secondary",
                                        disabled=not shared_schedule_editable,
                                        className="w-100",
                                    ),
                                ],
                                md=3,
                            ),
                        ],
                        className="g-3",
                    ),
                    html.Div(shared_schedule_alerts, className="mt-3") if shared_schedule_alerts else html.Div(),
                ]
            ),
            className="mb-4",
        )
        lifecycle_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Monitor Lifecycle", className="text-light mb-2"),
                    html.P(
                        "Archive stops scheduled refreshes and hides the monitor from the active app list while keeping its stored history. Restore makes an archived monitor active again. Delete permanently removes the monitor and all persisted monitoring history for this model key.",
                        className="text-muted mb-3",
                    ),
                    dbc.Row(
                        lifecycle_buttons,
                        className="g-3",
                    ),
                    archive_modal,
                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("Delete Monitor And History")),
                            dbc.ModalBody(
                                [
                                    html.P(
                                        "This permanently removes the monitor and all persisted monitoring history. Type the exact model key to confirm.",
                                        className="mb-3",
                                    ),
                                    dbc.Input(
                                        id="reference-delete-confirm-input",
                                        placeholder=config.model_key,
                                        value="",
                                        valid=False,
                                        invalid=False,
                                    ),
                                    dcc.Store(id="reference-delete-model-key-store", data=config.model_key),
                                    html.P(
                                        (
                                            f"Delete the selected monitor {config.display_name} ({config.model_key})? "
                                            "This permanently removes the monitor configuration and all persisted monitoring history stored under this model key."
                                        ),
                                        className="text-muted small mt-3 mb-2",
                                    ),
                                    html.Div(
                                        "Type the exact model key for the selected monitor to enable delete.",
                                        id="reference-delete-confirm-status",
                                        className="text-muted small mt-2",
                                    ),
                                ]
                            ),
                            dbc.ModalFooter(
                                [
                                    dbc.Button("Cancel", id="reference-delete-cancel-btn", color="secondary", outline=True),
                                    dbc.Button("Delete", id="reference-delete-confirm-btn", color="danger", disabled=True),
                                ]
                            ),
                        ],
                        id="reference-delete-modal",
                        is_open=False,
                    ),
                ]
            ),
            className="mb-4",
        )
        contract_tab = html.Div(
            [
                html.P(
                    "Read-only contract details and the latest persisted summary for the selected monitor.",
                    className="text-muted mb-3",
                ),
                html.H6("Monitor Contract", className="text-light mb-2"),
                _render_frame(contract_frame, "No contract data."),
                html.Hr(),
                html.H6("Latest Summary", className="text-light mb-2"),
                _render_frame(summary_frame, "No summary data."),
            ]
        )
        settings_tab = html.Div(
            [
                html.P(
                    "Update cadence and metrics here, then review runtime state and recent refresh performance.",
                    className="text-muted mb-3",
                ),
                schedule_card,
                dbc.Alert(
                    "The shared refresh workflow wakes up on the schedule shown in Admin. The cadence settings above decide whether this monitor is actually due for drift/quality work, while performance cadence is evaluated independently when labels are present.",
                    color="secondary",
                    className="py-2 mb-3",
                ),
                html.H6("Compute Guidance", className="text-light mb-2"),
                _compute_guidance_block(
                    config=config,
                    diagnostics=refresh_diagnostics,
                    shared_schedule=shared_schedule,
                ),
                html.H6("Runtime State", className="text-light mb-2"),
                _render_frame(runtime_frame, "No runtime state yet."),
                html.Hr(),
                html.H6("Refresh Diagnostics", className="text-light mb-2"),
                html.Small(
                    "Recent duration and scanned-row telemetry is the best practical proxy for compute footprint in this deployment model.",
                    className="text-muted d-block mb-2",
                ),
                _render_refresh_diagnostics(refresh_diagnostics),
                html.Hr(),
                html.H6("Recent Refresh Runs", className="text-light mb-2"),
                _render_frame(recent_runs_frame, "No refresh runs recorded yet."),
            ]
        )
        admin_tab = html.Div(
            [
                html.P(
                    "Use this tab for lifecycle actions, refresh retries, and workspace wiring details.",
                    className="text-muted mb-3",
                ),
                lifecycle_card,
                shared_schedule_card,
                html.H6("Recent Incident History", className="text-light mb-2"),
                _render_frame(recent_incident_history_frame, "No incident history recorded yet."),
                html.Hr(),
                html.H6("Runtime Settings", className="text-light mb-2"),
                html.P(
                    refresh_job_wiring_text,
                    className="text-muted mb-2",
                ),
                _render_frame(settings_frame, "No runtime settings."),
            ]
        )
        return html.Div(
            [
                html.P(
                    f"Selected monitor: {config.display_name} ({config.model_key}, {config_status}). Use the tabs below to move between contract details, editable settings, and admin actions.",
                    className="text-muted mb-3",
                ),
                html.Small(
                    " | ".join(
                        [
                            f"Source: {config.source_table}",
                            *(
                                [f"Model ID Value: {config.model_id_value}"]
                                if getattr(config, "model_id_value", None)
                                else []
                            ),
                            *(
                                [f"Model Version Value: {config.model_version_value}"]
                                if getattr(config, "model_version_value", None)
                                else []
                            ),
                        ]
                    ),
                    className="text-muted d-block mb-3",
                ),
                dbc.Tabs(
                    [
                        dbc.Tab(contract_tab, label="Contract", tab_id="reference-contract-tab"),
                        dbc.Tab(settings_tab, label="Settings", tab_id="reference-settings-tab"),
                        dbc.Tab(admin_tab, label="Admin", tab_id="reference-admin-tab"),
                    ],
                    id="reference-sections-tabs",
                    active_tab="reference-contract-tab",
                    className="mb-3",
                ),
            ]
        )

    @app.callback(
        Output("reference-performance-metrics-select", "options"),
        Output("reference-performance-metrics-select", "value"),
        Output("reference-default-performance-metric-select", "options"),
        Output("reference-default-performance-metric-select", "value"),
        Input("global-model-select", "value"),
        Input("reference-monitor-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("reference-performance-metrics-select", "value"),
        State("reference-default-performance-metric-select", "value"),
    )
    def sync_reference_performance_metrics(
        global_model_id,
        reference_model_id,
        _,
        session_data,
        selected_metrics,
        current_default,
    ):
        model_id = _resolve_reference_model_id(global_model_id, reference_model_id)
        if not model_id:
            return [], [], [], None
        try:
            backend = _make_backend(session_data)
            config = _get_monitor_config_for_reference(backend, model_id)
            if not config:
                return [], [], [], None
            seed_metrics = selected_metrics if selected_metrics is not None else _configured_performance_metric_names(config)
            seed_default = current_default if current_default is not None else _configured_default_performance_metric(config)
            return _sync_performance_metric_selection(
                problem_type=config.problem_type,
                selected_metrics=seed_metrics,
                current_default=seed_default,
            )
        except Exception as error:
            logger.exception("Failed to sync reference performance metrics", exc_info=error)
            return [], [], [], None

    @app.callback(
        Output("reference-page-status", "children"),
        Output("reload-token", "data", allow_duplicate=True),
        Input("reference-save-schedule-btn", "n_clicks"),
        State("global-model-select", "value"),
        State("reference-monitor-select", "value"),
        State("reference-drift-cadence-select", "value"),
        State("reference-performance-cadence-select", "value"),
        State("reference-schedule-enabled-toggle", "value"),
        State("reference-performance-metrics-select", "value"),
        State("reference-default-performance-metric-select", "value"),
        State("reference-performance-binning-mode-select", "value"),
        State("reference-performance-binning-clip-input", "value"),
        State("reference-threshold-psi-warning-input", "value"),
        State("reference-threshold-psi-critical-input", "value"),
        State("reference-threshold-js_divergence-warning-input", "value"),
        State("reference-threshold-js_divergence-critical-input", "value"),
        State("reference-threshold-kl_divergence-warning-input", "value"),
        State("reference-threshold-kl_divergence-critical-input", "value"),
        State("reference-threshold-null_rate-warning-input", "value"),
        State("reference-threshold-null_rate-critical-input", "value"),
        State("session-config-store", "data"),
        prevent_initial_call=True,
    )
    def save_reference_schedule(
        _,
        global_model_id,
        reference_model_id,
        drift_cadence,
        performance_cadence,
        schedule_enabled,
        performance_metric_names,
        default_performance_metric,
        performance_binning_mode,
        performance_binning_clip_percentile,
        psi_warning,
        psi_critical,
        js_warning,
        js_critical,
        kl_warning,
        kl_critical,
        null_warning,
        null_critical,
        session_data,
    ):
        model_id = _resolve_reference_model_id(global_model_id, reference_model_id)
        if not model_id:
            return _status_alert("Select a monitor before updating cadence.", "warning"), no_update
        backend = _make_backend(session_data)
        config = _get_monitor_config_for_reference(backend, model_id)
        if not config:
            return _status_alert("Selected monitor no longer exists.", "warning"), no_update
        try:
            threshold_overrides = _collect_threshold_overrides_from_inputs(
                {
                    "psi": (psi_warning, psi_critical),
                    "js_divergence": (js_warning, js_critical),
                    "kl_divergence": (kl_warning, kl_critical),
                    "null_rate": (null_warning, null_critical),
                }
            )
            updated = replace(
                config,
                performance_metric_names=tuple(performance_metric_names or ()),
                default_performance_metric=default_performance_metric or None,
                performance_binning_mode=performance_binning_mode or config.performance_binning_mode,
                performance_binning_clip_percentile=performance_binning_clip_percentile,
                drift_cadence_preset=drift_cadence or config.drift_cadence_preset,
                performance_cadence_preset=(
                    performance_cadence
                    if config.has_labels
                    else "disabled"
                ),
                schedule_enabled="enabled" in (schedule_enabled or []),
                threshold_overrides=threshold_overrides,
            )
            backend.repository.upsert_monitor_config(updated)
            backend.repository.ensure_monitor_runtime_state(updated)
        except ValueError as error:
            return _status_alert(str(error), "warning"), no_update
        except Exception as error:
            logger.exception("Failed to update monitor settings for %s", model_id, exc_info=error)
            return _status_alert(_user_action_error_message("Updating monitor settings"), "danger"), no_update
        return (
            _status_alert(f"Updated monitor settings for {updated.display_name}.", "success"),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @app.callback(
        Output("reference-page-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Input("reference-save-shared-schedule-btn", "n_clicks"),
        State("reference-shared-schedule-select", "value"),
        prevent_initial_call=True,
    )
    def save_reference_shared_schedule(_, interval_hours):
        try:
            status = update_shared_workflow_schedule(int(interval_hours or 1))
        except Exception as error:
            return _status_alert(_user_action_error_message("Updating the shared workflow schedule"), "danger"), no_update
        return (
            _status_alert(
                f"Updated the shared refresh workflow to {status.current_label.lower()} for job {status.job_id}.",
                "success",
            ),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @app.callback(
        Output("reference-page-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Input("reference-run-bootstrap-btn", "n_clicks"),
        State("global-model-select", "value"),
        State("reference-monitor-select", "value"),
        State("session-config-store", "data"),
        prevent_initial_call=True,
    )
    def trigger_reference_bootstrap(_, global_model_id, reference_model_id, session_data):
        model_id = _resolve_reference_model_id(global_model_id, reference_model_id)
        if not model_id:
            return _status_alert("Select a monitor before triggering its initial refresh.", "warning"), no_update
        backend = _make_backend(session_data)
        config = _get_monitor_config_for_reference(backend, model_id)
        if not config:
            return _status_alert("Selected monitor no longer exists.", "warning"), no_update
        try:
            trigger = trigger_refresh_job(
                model_key=config.model_key,
                control_plane_catalog=(session_data or {}).get("control_plane_catalog", settings.control_plane_catalog),
                control_plane_schema=(session_data or {}).get("control_plane_schema", settings.control_plane_schema),
                lakebase_instance_name=(session_data or {}).get("lakebase_instance_name", settings.lakebase_instance_name),
                lakebase_database_name=(session_data or {}).get("lakebase_database_name", settings.lakebase_database_name),
                lakebase_schema=(session_data or {}).get("lakebase_schema", settings.lakebase_schema),
                scope="bootstrap",
            )
        except Exception as error:
            return _status_alert(_manual_refresh_unavailable_message(config.model_key, error), "warning"), no_update
        run_id_text = f", run_id={trigger.run_id}" if trigger.run_id is not None else ""
        return (
            _status_alert(
                f"Triggered the initial refresh for {config.display_name} (job_id={trigger.job_id}{run_id_text}). Check Overview in a minute.",
                "success",
            ),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @app.callback(
        Output("reference-archive-modal", "is_open"),
        Input("reference-archive-monitor-btn", "n_clicks"),
        Input("reference-archive-cancel-btn", "n_clicks"),
        Input("reference-archive-confirm-btn", "n_clicks"),
        State("reference-archive-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_reference_archive_modal(open_clicks, cancel_clicks, confirm_clicks, is_open):
        if any((open_clicks, cancel_clicks, confirm_clicks)):
            return not is_open
        return is_open

    @app.callback(
        Output("reference-page-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Input("reference-archive-confirm-btn", "n_clicks"),
        State("global-model-select", "value"),
        State("reference-monitor-select", "value"),
        State("session-config-store", "data"),
        prevent_initial_call=True,
    )
    def archive_reference_monitor(_, global_model_id, reference_model_id, session_data):
        model_id = _resolve_reference_model_id(global_model_id, reference_model_id)
        if not model_id:
            return _status_alert("Select a monitor before archiving it.", "warning"), no_update
        backend = _make_backend(session_data)
        config, resolution_error = _resolve_monitor_config_by_key(backend, model_id, fail_if_ambiguous=True)
        if resolution_error:
            return _status_alert(resolution_error, "warning"), no_update
        try:
            backend.repository.archive_monitor(model_id)
        except KeyError as error:
            return _status_alert(str(error), "warning"), no_update
        except Exception as error:
            logger.exception("Archiving monitor %s failed", model_id, exc_info=error)
            return _status_alert(_user_action_error_message("Archiving the monitor"), "danger"), no_update
        resolved_key = str(getattr(config, "model_key", "") or model_id).strip() or model_id
        return (
            _status_alert(
                f"Archived {config.display_name} ({resolved_key}). It is no longer active, but its historical rows under this model key were kept.",
                "success",
            ),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @app.callback(
        Output("reference-page-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Input("reference-restore-monitor-btn", "n_clicks"),
        State("global-model-select", "value"),
        State("reference-monitor-select", "value"),
        State("session-config-store", "data"),
        prevent_initial_call=True,
    )
    def restore_reference_monitor(restore_clicks, global_model_id, reference_model_id, session_data):
        if not restore_clicks or ctx.triggered_id != "reference-restore-monitor-btn":
            return no_update, no_update
        model_id = _resolve_reference_model_id(global_model_id, reference_model_id)
        if not model_id:
            return _status_alert("Select a monitor before restoring it.", "warning"), no_update
        backend = _make_backend(session_data)
        config, resolution_error = _resolve_monitor_config_by_key(backend, model_id, fail_if_ambiguous=True)
        if resolution_error:
            return _status_alert(resolution_error, "warning"), no_update
        try:
            backend.repository.restore_monitor(model_id)
        except KeyError as error:
            return _status_alert(str(error), "warning"), no_update
        except Exception as error:
            logger.exception("Restoring monitor %s failed", model_id, exc_info=error)
            return _status_alert(_user_action_error_message("Restoring the monitor"), "danger"), no_update
        resolved_key = str(getattr(config, "model_key", "") or model_id).strip() or model_id
        return (
            _status_alert(
                f"Restored {config.display_name} ({resolved_key}). It is active again and eligible for scheduled refresh.",
                "success",
            ),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @app.callback(
        Output("reference-delete-modal", "is_open"),
        Output("reference-delete-confirm-input", "value"),
        Input("reference-delete-monitor-btn", "n_clicks"),
        Input("reference-delete-cancel-btn", "n_clicks"),
        Input("reference-delete-confirm-btn", "n_clicks"),
        State("reference-delete-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_reference_delete_modal(open_clicks, cancel_clicks, confirm_clicks, is_open):
        if any((open_clicks, cancel_clicks, confirm_clicks)):
            return not is_open, ""
        return is_open, no_update

    @app.callback(
        Output("reference-delete-confirm-btn", "disabled"),
        Output("reference-delete-confirm-input", "valid"),
        Output("reference-delete-confirm-input", "invalid"),
        Output("reference-delete-confirm-status", "children"),
        Input("reference-delete-confirm-input", "value"),
        State("reference-delete-model-key-store", "data"),
        prevent_initial_call=False,
    )
    def validate_reference_delete_confirmation(confirmation_text, expected_model_key):
        expected = str(expected_model_key or "").strip()
        typed = str(confirmation_text or "").strip()
        if not expected:
            return True, False, False, "Select a monitor before deleting it."
        if not typed:
            return (
                True,
                False,
                False,
                html.Span(
                    [
                        "Type ",
                        html.Code(expected),
                        " to enable delete.",
                    ],
                    className="text-muted small",
                ),
            )
        if typed == expected:
            return (
                False,
                True,
                False,
                html.Span(
                    [
                        html.I(className="fas fa-check-circle me-2"),
                        "Model key matched. Delete is enabled.",
                    ],
                    className="text-success small",
                ),
            )
        return (
            True,
            False,
            True,
            html.Span(
                [
                    html.I(className="fas fa-circle-exclamation me-2"),
                    "Typed value does not match ",
                    html.Code(expected),
                    ".",
                ],
                className="text-warning small",
            ),
        )

    @app.callback(
        Output("reference-page-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Input("reference-delete-confirm-btn", "n_clicks"),
        State("global-model-select", "value"),
        State("reference-monitor-select", "value"),
        State("reference-delete-confirm-input", "value"),
        State("session-config-store", "data"),
        prevent_initial_call=True,
    )
    def delete_reference_monitor(_, global_model_id, reference_model_id, confirmation_text, session_data):
        model_id = _resolve_reference_model_id(global_model_id, reference_model_id)
        if not model_id:
            return _status_alert("Select a monitor before deleting it.", "warning"), no_update
        backend = _make_backend(session_data)
        config, resolution_error = _resolve_monitor_config_by_key(backend, model_id, fail_if_ambiguous=True)
        if resolution_error:
            return _status_alert(resolution_error, "warning"), no_update
        if (confirmation_text or "").strip() != config.model_key:
            return _status_alert(
                f"Type the exact model key ({config.model_key}) before deleting this monitor.",
                "warning",
            ), no_update
        try:
            backend.repository.delete_monitor(model_id)
        except KeyError as error:
            return _status_alert(str(error), "warning"), no_update
        except Exception as error:
            logger.exception("Deleting monitor %s failed", model_id, exc_info=error)
            return _status_alert(_user_action_error_message("Deleting the monitor"), "danger"), no_update
        resolved_key = str(getattr(config, "model_key", "") or model_id).strip() or model_id
        return (
            _status_alert(
                f"Deleted {config.display_name} ({resolved_key}) and its persisted monitoring history under that model key.",
                "success",
            ),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
