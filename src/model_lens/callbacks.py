from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import parse_qs

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html, ctx, no_update

from model_lens.backend import DashboardBackend, build_dashboard_backend
from model_lens.config import settings
from model_lens.domain.models import MLflowLineage, MonitorConfig
from model_lens.pages import onboarding
from model_lens.services.contracts import build_contract
from model_lens.services.onboarding import baseline_label, build_default_baseline, build_fixed_baseline
from model_lens.services.refresh_runner import run_refresh_cycle
from model_lens.ui import charts
from model_lens.ui.components import (
    get_thresholds,
    make_chart_card,
    make_empty_state,
    make_metric_card,
    make_model_status_card,
    make_wizard_step,
)


_NUMERIC_TYPE_TOKENS = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal", "numeric", "real")


@lru_cache(maxsize=1)
def _workspace_lakebase_instances() -> tuple[str, ...]:
    if not os.getenv("DATABRICKS_APP_PORT"):
        return ()
    try:
        from databricks.sdk import WorkspaceClient

        names: list[str] = []
        for instance in WorkspaceClient().database.list_database_instances(page_size=20):
            name = str(getattr(instance, "name", "") or "").strip()
            if name:
                names.append(name)
            if len(names) >= 5:
                break
        return tuple(names)
    except Exception:
        return ()


def _status_alert(message: str, color: str = "info") -> dbc.Alert:
    return dbc.Alert(message, color=color, className="py-2 mb-3")


def _setup_retry_message(error: object) -> str:
    return f"Setup failed. Fix the issue and click Setup Control Plane again to retry. Details: {error}"


def _status_block(items: list[tuple[str, str]]) -> html.Div:
    return html.Div([_status_alert(message, color) for message, color in items])


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
    model_id_col = _guess_column(columns, ("model_id", "model", "model_name"))
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
    recorded_catalog = str(ready_state.get("control_plane_catalog") or "").strip()
    recorded_schema = str(ready_state.get("control_plane_schema") or "").strip()
    return bool(recorded_catalog and recorded_schema)


def _selected_model_from_search(search: str | None) -> str | None:
    if not search:
        return None
    parsed = parse_qs(search.lstrip("?"))
    values = parsed.get("model", [])
    if not values:
        return None
    selected = str(values[0]).strip()
    return selected or None


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
    return bool(
        recorded_catalog
        and recorded_schema
        and recorded_catalog == session["control_plane_catalog"]
        and recorded_schema == session["control_plane_schema"]
    )


def _step_style(is_active: bool) -> dict:
    return {} if is_active else {"display": "none"}


def _monitor_contract_ready(
    *,
    scan_data: dict | None,
    display_name: str | None,
    model_key: str | None,
    timestamp_col: str | None,
    model_id_col: str | None,
    prediction_col: str | None,
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
            model_id_col,
            prediction_col,
            feature_columns,
        ]
    ):
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
        return bool(entity_id_col and (labels_join_col or "").strip() and ((external_label_col or "").strip() or source_label_col))
    return True


def _review_summary(
    *,
    control_plane_catalog: str | None,
    control_plane_schema: str | None,
    source_table: str | None,
    display_name: str | None,
    model_key: str | None,
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
    lakebase_instance_name: str | None,
    lakebase_database_name: str | None,
    mlflow_experiment_name: str | None,
    mlflow_registered_model_name: str | None,
) -> html.Div:
    label_source = (labels_table or "").strip() if (labels_table or "").strip() else ((source_label_col or "").strip() or "none")
    baseline_policy = (
        build_fixed_baseline((baseline_start or "").strip(), (baseline_end or "").strip())
        if (baseline_kind or "rolling") == "fixed" and (baseline_start or "").strip() and (baseline_end or "").strip()
        else build_default_baseline(int(baseline_days or 7))
    )
    rows = [
        ("Control Plane", f"{(control_plane_catalog or '').strip()}.{(control_plane_schema or '').strip()}"),
        ("Source Table", (source_table or "").strip() or "Not scanned yet"),
        ("Display Name", (display_name or "").strip() or "Not set"),
        ("Model Key", (model_key or "").strip() or "Not set"),
        ("Problem Type", (problem_type or "classification").title()),
        ("Baseline Policy", baseline_label(baseline_policy)),
        ("Feature Columns", str(len(feature_columns or []))),
        ("Categorical Columns", str(len(categorical_columns or []))),
        ("Slice Columns", str(len(slice_columns or []))),
        ("Model Scope", (model_id_value or "").strip() or "All model_id values"),
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
        step = int(current_step or 1)
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
        Input("mlflow-experiment-input", "value"),
        Input("mlflow-registered-model-input", "value"),
        Input("lakebase-instance-input", "value"),
        Input("lakebase-database-input", "value"),
        Input("lakebase-schema-input", "value"),
    )
    def render_onboarding_wizard(
        current_step,
        control_plane_ready_state,
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
        mlflow_experiment_name,
        mlflow_registered_model_name,
        lakebase_instance_name,
        lakebase_database_name,
        lakebase_schema,
    ):
        step = max(1, min(int(current_step or 1), len(onboarding.STEP_LABELS)))
        workspace_ready = _control_plane_ready(
            control_plane_ready_state,
            control_plane_catalog=control_plane_catalog,
            control_plane_schema=control_plane_schema,
            lakebase_instance_name=lakebase_instance_name,
            lakebase_database_name=lakebase_database_name,
            lakebase_schema=lakebase_schema,
        )
        source_ready = bool(scan_data and scan_data.get("columns"))
        contract_ready = _monitor_contract_ready(
            scan_data=scan_data,
            display_name=display_name,
            model_key=model_key,
            timestamp_col=timestamp_col,
            model_id_col=model_id_col,
            prediction_col=prediction_col,
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
            4: True,
        }.get(step, False)
        guidance = {
            1: (
                "Confirm the control-plane namespace and run setup. If setup fails, fix the issue and click Setup Control Plane again to retry. If you change the catalog or schema later, run setup again before saving the monitor.",
                "info" if workspace_ready else "secondary",
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
                "Activate the monitor. Model Lens saves the config and runs the initial refresh for you.",
                "primary",
            ),
        }
        next_labels = {
            1: "Continue to Source",
            2: "Continue to Contract",
            3: "Continue to Review",
            4: "Continue",
        }
        review = _review_summary(
            control_plane_catalog=control_plane_catalog,
            control_plane_schema=control_plane_schema,
            source_table=(scan_data or {}).get("table_name") or source_table,
            display_name=display_name,
            model_key=model_key,
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
            next_labels[step],
            not (workspace_ready and contract_ready),
            review,
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
        backend = _make_backend(session_data)
        models = backend.list_models()
        options = [{"label": model["name"], "value": model["id"]} for model in models]
        if not options:
            return [], None
        values = {option["value"] for option in options}
        requested = _selected_model_from_search(search)
        if requested in values:
            return options, requested
        return options, current_value if current_value in values else options[0]["value"]

    @app.callback(
        Output("sidebar-status", "children"),
        Output("sidebar-alert-badge", "children"),
        Output("sidebar-mode-banner", "children"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def update_sidebar(model_id, _, session_data):
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
                html.Small(model["description"], className="text-muted d-block"),
                html.Small(
                    f"Features: {model['feature_count']} | Baseline: {model['baseline_label']}",
                    className="text-muted d-block",
                ),
                html.Small(f"Rows observed: {model['total_rows']}", className="text-muted d-block"),
            ]
        )
        return status, badge, _deployment_mode_prompt(session_data)

    @app.callback(
        Output("drift-model-banner", "children"),
        Output("deepdive-model-banner", "children"),
        Output("perf-model-banner", "children"),
        Output("quality-model-banner", "children"),
        Input("global-model-select", "value"),
        Input("session-config-store", "data"),
    )
    def update_analysis_banners(model_id, session_data):
        backend = _make_backend(session_data)
        banner = _model_banner(model_id, backend)
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
            return no_update, _status_alert(f"Scan failed: {error}", "danger"), html.Div()
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
                "model_id_col": discovery.config.contract.model_id_col,
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
                status_items.append(
                    (
                        f"Join validation: matched={int(label_validation.get('matched_rows', 0) or 0)}, "
                        f"unmatched={int(label_validation.get('unmatched_rows', 0) or 0)}, "
                        f"duplicate_keys={int(label_validation.get('duplicate_join_keys', 0) or 0)}.",
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
    )
    def populate_monitor_form(scan_data):
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
            required_options,
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
            return _status_alert(_setup_retry_message(error), "danger"), no_update, no_update, no_update
        return (
            _status_alert(
                f"Control plane ready at {backend.repository.table_names.catalog}.{backend.repository.table_names.schema}.",
                "success",
            ),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            session,
            session,
        )

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
        State("control-plane-catalog-input", "value"),
        State("control-plane-schema-input", "value"),
        State("lakebase-instance-input", "value"),
        State("lakebase-database-input", "value"),
        State("lakebase-schema-input", "value"),
        State("control-plane-ready-store", "data"),
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
        control_plane_catalog,
        control_plane_schema,
        lakebase_instance_name,
        lakebase_database_name,
        lakebase_schema,
        ready_state,
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
        label_col = external_label_col or source_label_col or None
        if labels_table and (not entity_id_col or not labels_join_col or not label_col):
            return _status_alert(
                "External labels require Entity ID Column, External Labels Join Column, and External Label Column.",
                "warning",
            ), no_update, no_update
        if model_version_value and not model_version_col:
            return _status_alert("Monitored Model Version Value requires a mapped Model Version Column.", "warning"), no_update, no_update
        try:
            contract = build_contract(
                columns=scan_data["columns"],
                timestamp_col=timestamp_col,
                model_id_col=model_id_col,
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
            backend.repository.validate_monitor_source(config)
            backend.repository.upsert_monitor_config(config)
            counts = run_refresh_cycle(backend.repository, model_key=config.model_key)
        except Exception as error:
            return _status_alert(f"Save failed: {error}", "danger"), no_update, no_update
        messages = [(
            f"Saved monitor {config.model_key} and ran initial refresh: drift_rows={counts.drift_rows}, quality_rows={counts.quality_rows}, performance_rows={counts.performance_rows}, incidents={counts.incident_rows}.",
            "success" if counts.models else "warning",
        )]
        if not counts.models:
            messages.append(("The monitor was saved, but the current source data did not produce a comparable baseline/current window yet.", "warning"))
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
        backend = _make_backend(session_data)
        overview_data = backend.get_overview_rows(metric="psi")
        if not overview_data:
            return make_empty_state("No monitors configured. Use Onboarding to add a model.", icon="fas fa-plus-circle")

        psi_warning, psi_critical = get_thresholds("psi")
        healthy = sum(1 for row in overview_data if row["max_psi"] <= psi_warning)
        warning = sum(1 for row in overview_data if psi_warning < row["max_psi"] <= psi_critical)
        critical = sum(1 for row in overview_data if row["max_psi"] > psi_critical)
        summary_row = dbc.Row(
            [
                dbc.Col(make_metric_card("Models Monitored", str(len(overview_data)), "Active in production"), md=3),
                dbc.Col(make_metric_card("Healthy", str(healthy), f"PSI < {psi_warning}", "success"), md=3),
                dbc.Col(make_metric_card("Warning", str(warning), f"{psi_warning} < PSI < {psi_critical}", "warning"), md=3),
                dbc.Col(make_metric_card("Critical", str(critical), f"PSI > {psi_critical}", "danger"), md=3),
            ],
            className="mb-4 g-3",
        )
        sorted_data = sorted(overview_data, key=lambda item: item["max_psi"], reverse=True)
        model_cards = [
            dbc.Col(
                dcc.Link(
                    make_model_status_card(
                        model_name=row["model_name"],
                        model_id=row["model_id"],
                        description=row["description"],
                        max_psi=row["max_psi"],
                        avg_psi=row["avg_psi"],
                        drifting_count=row["drifting_features"],
                        total_features=row["total_features"],
                        max_null_rate=row["max_null_rate"],
                        has_labels=row["has_labels"],
                        computing=row["computing"],
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
                {"model": row["model_name"], "max_psi": row["max_psi"], "avg_psi": row["avg_psi"], "drifting_features": row["drifting_features"]}
                for row in sorted_data
            ]
        )
        details = pd.DataFrame(
            [
                {
                    "model": row["model_name"],
                    "versions": ", ".join(row.get("versions", [])),
                    "max_psi": round(row["max_psi"], 4),
                    "avg_psi": round(row["avg_psi"], 4),
                    "avg_js": round(row["avg_js"], 4),
                    "drifting_features": f"{row['drifting_features']} / {row['total_features']}",
                    "top_drifter": row["top_drifter"],
                    "max_null_rate": round(row["max_null_rate"], 2),
                }
                for row in sorted_data
            ]
        )
        return html.Div([summary_row, dbc.Row(model_cards, className="mb-4"), make_chart_card(summary_chart), _render_frame(details, "No overview detail available.")])

    @app.callback(
        Output("drift-heatmap-container", "children"),
        Output("drift-categorical-note", "children"),
        Output("drift-timeline-container", "children"),
        Output("drift-top-drifters-container", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("drift-metric-select", "value"),
        Input("drift-granularity-select", "value"),
        Input("drift-top-n", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def render_drift(pathname, model_id, metric, granularity, top_n, _, session_data):
        if pathname != "/drift":
            return no_update, no_update, no_update, no_update
        if not model_id:
            empty = make_empty_state("Select a model to inspect drift.", icon="fas fa-wave-square")
            return empty, html.Div(), html.Div(), html.Div()
        backend = _make_backend(session_data)
        config = backend.get_monitor_config(model_id)
        drift = backend.get_drift_results(model_id, granularity=granularity or "daily")
        if drift.empty:
            empty = make_empty_state("No drift history available yet. Run a refresh to populate this page.", icon="fas fa-wave-square")
            return empty, html.Div(), html.Div(), html.Div()
        latest = drift[drift["period"] == drift["period"].max()].nlargest(int(top_n or 10), metric or "psi")
        thresholds = dict(zip(("warning", "critical"), get_thresholds(metric or "psi")))
        note = html.Div()
        if config and config.contract.categorical_columns:
            note = _status_alert(
                "Categorical features are stored in the monitor contract, but the current drift engine renders only numeric feature drift on this page.",
                "secondary",
            )
        return (
            make_chart_card(charts.build_drift_heatmap(drift, metric=metric or "psi")),
            note,
            make_chart_card(charts.build_drift_timeline(drift, latest["feature"].tolist(), metric=metric or "psi", thresholds=thresholds)),
            make_chart_card(charts.build_top_drifters_bar(drift, metric=metric or "psi", top_n=int(top_n or 10))),
        )

    @app.callback(
        Output("deepdive-feature-select", "options"),
        Output("deepdive-feature-select", "value"),
        Output("deepdive-dimension-select", "options"),
        Output("deepdive-dimension-select", "value"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("deepdive-feature-select", "value"),
        State("deepdive-dimension-select", "value"),
    )
    def populate_feature_deep_dive(pathname, model_id, _, session_data, feature_value, dimension_value):
        if pathname != "/features":
            return no_update, no_update, no_update, no_update
        backend = _make_backend(session_data)
        feature_options = _option_list(backend.get_feature_options(model_id or ""))
        dimension_options = _option_list(backend.get_dimension_options(model_id or ""), include_blank=True)
        feature_values = {option["value"] for option in feature_options}
        dimension_values = {option["value"] for option in dimension_options}
        selected_feature = feature_value if feature_value in feature_values else (feature_options[0]["value"] if feature_options else None)
        selected_dimension = dimension_value if dimension_value in dimension_values else ""
        return feature_options, selected_feature, dimension_options, selected_dimension

    @app.callback(
        Output("deepdive-distribution-container", "children"),
        Output("deepdive-dimension-container", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("deepdive-feature-select", "value"),
        Input("deepdive-dimension-select", "value"),
        Input("session-config-store", "data"),
    )
    def render_feature_deep_dive(pathname, model_id, feature, dimension, session_data):
        if pathname != "/features":
            return no_update, no_update
        if not model_id or not feature:
            empty = make_empty_state("Select a model and feature to inspect.", icon="fas fa-search")
            return empty, html.Div()
        backend = _make_backend(session_data)
        baseline, current = backend.get_feature_distribution(model_id, feature)
        distribution = make_chart_card(charts.build_feature_distribution(baseline, current, feature))
        dimension_chart = html.Div()
        if dimension:
            breakdown = backend.get_dimension_breakdown(model_id, feature, dimension)
            dimension_chart = make_chart_card(charts.build_dimension_breakdown(breakdown, feature, dimension))
        return distribution, dimension_chart

    @app.callback(
        Output("quality-kpi-cards", "children"),
        Output("quality-volume-container", "children"),
        Output("quality-null-rates-container", "children"),
        Output("quality-prediction-container", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def render_quality(pathname, model_id, _, session_data):
        if pathname != "/quality":
            return no_update, no_update, no_update, no_update
        if not model_id:
            empty = make_empty_state("Select a model to inspect quality.", icon="fas fa-database")
            return empty, html.Div(), html.Div(), html.Div()
        backend = _make_backend(session_data)
        quality = backend.get_quality_stats(model_id)
        if not quality:
            empty = make_empty_state("No quality snapshot available yet. Run a refresh first.", icon="fas fa-database")
            return empty, html.Div(), html.Div(), html.Div()
        kpis = [
            dbc.Col(make_metric_card("Rows", f"{quality['total_rows']:,}", "Observed rows"), md=3),
            dbc.Col(make_metric_card("From", quality["min_date"] or "—", "Earliest data"), md=3),
            dbc.Col(make_metric_card("To", quality["max_date"] or "—", "Latest data"), md=3),
            dbc.Col(make_metric_card("Prediction Mean", f"{quality['prediction_mean']:.4f}", "Latest snapshot"), md=3),
        ]
        return (
            kpis,
            make_chart_card(charts.build_volume_timeline(quality["daily_volume"])),
            make_chart_card(charts.build_null_rate_chart(quality["null_rates"])),
            make_chart_card(charts.build_prediction_distribution(backend.get_prediction_distribution(model_id))),
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
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
        State("perf-feature-select", "value"),
    )
    def render_performance(pathname, model_id, metric_name, _, session_data, current_feature):
        if pathname != "/performance":
            return (no_update,) * 7
        if not model_id:
            empty = make_empty_state("Select a model to inspect performance.", icon="fas fa-tachometer-alt")
            return empty, html.Div(), html.Div(), html.Div(), [], None, html.Div()
        backend = _make_backend(session_data)
        config = backend.get_monitor_config(model_id)
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
        performance = backend.get_performance_summary(model_id, metric_name=metric_name or "f1")
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
        contributors = performance["contributors"]
        feature_frame = latest_bins if not latest_bins.empty else all_bins
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
        alert = html.Div()
        if not degradation_detected:
            alert = _status_alert(
                "Performance metrics are populated, but no significant degradation is detected in the latest window.",
                "info",
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
            dbc.Col(make_metric_card("Windows", str(len(performance["timeline"])), "Historical performance snapshots"), md=4),
        ]
        note_source = latest_bins if not latest_bins.empty else all_bins
        note = note_source[[column for column in ("window_start", "window_end") if column in note_source.columns]].drop_duplicates().astype(str)
        note_text = html.Small(
            f"Latest comparison window: {note.iloc[0]['window_start']} to {note.iloc[0]['window_end']}" if not note.empty else "",
            className="text-muted",
        )
        return (
            alert,
            kpi_cards,
            make_chart_card(charts.build_performance_timeline(performance["timeline"], metric_name=metric_name or "f1")),
            make_chart_card(charts.build_feature_bin_impact(feature_frame, contributors)),
            feature_options,
            selected_feature,
            note_text,
        )

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
            return html.Div()
        backend = _make_backend(session_data)
        performance = backend.get_performance_summary(model_id, metric_name=metric_name or "f1")
        latest_bins = performance["latest_bins"] if not performance["latest_bins"].empty else performance.get("all_bins", pd.DataFrame())
        feature_bins = latest_bins[latest_bins["feature"] == feature]
        return make_chart_card(charts.build_bin_detail(feature_bins, feature, metric_name=metric_name or "f1"))

    @app.callback(
        Output("reference-page-body", "children"),
        Input("url", "pathname"),
        Input("global-model-select", "value"),
        Input("reload-token", "data"),
        Input("session-config-store", "data"),
    )
    def render_reference(pathname, model_id, _, session_data):
        if pathname != "/reference":
            return no_update
        backend = _make_backend(session_data)
        if not model_id:
            return make_empty_state("Select a model to inspect the contract and runtime state.", icon="fas fa-book")
        data = backend.get_reference_data(model_id)
        config = data["config"]
        if not config:
            return make_empty_state("Selected model is no longer available.", icon="fas fa-book")
        contract_frame = pd.DataFrame(
            [
                {"field": "timestamp_col", "value": config.contract.timestamp_col},
                {"field": "model_id_col", "value": config.contract.model_id_col},
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
            ]
        )
        summary_frame = pd.DataFrame([data["summary"]]) if data["summary"] else pd.DataFrame()
        settings_frame = pd.DataFrame(
            [
                {"field": field, "value": _format_runtime_setting_value(field, value)}
                for field, value in data["settings"].items()
            ]
        )
        return html.Div(
            [
                html.P(
                    f"Showing contract and latest summary for the currently selected monitor: {config.display_name} ({config.model_key}). Runtime settings are global to the app.",
                    className="text-muted mb-3",
                ),
                html.H6("Monitor Contract", className="text-light mb-2"),
                _render_frame(contract_frame, "No contract data."),
                html.Hr(),
                html.H6("Latest Summary", className="text-light mb-2"),
                _render_frame(summary_frame, "No summary data."),
                html.Hr(),
                html.H6("Runtime Settings", className="text-light mb-2"),
                _render_frame(settings_frame, "No runtime settings."),
            ]
        )
