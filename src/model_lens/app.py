from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, ctx, dcc, html

from model_lens.config import settings
from model_lens.domain.models import MonitorConfig
from model_lens.services.contracts import build_contract
from model_lens.services.control_plane import get_default_repository
from model_lens.services.onboarding import build_default_baseline
from model_lens.services.refresh_runner import run_refresh_cycle


def _repo():
    return get_default_repository()


_NUMERIC_TYPE_TOKENS = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal", "numeric", "real")


@lru_cache(maxsize=1)
def _workspace_lakebase_instances() -> tuple[str, ...]:
    # Only probe the workspace when running inside a Databricks App.
    # Local dev and tests should not block on network-bound workspace discovery.
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
    prediction_score_col = _guess_column(
        columns,
        ("prediction_proba", "prediction_score", "probability", "score"),
        fallback_first=False,
    )
    entity_id_col = _guess_column(
        columns,
        ("entity_id", "request_id", "user_id", "account_id", "id"),
        fallback_first=False,
    )
    source_label_col = _guess_column(columns, ("label", "target", "actual"), fallback_first=False)
    reserved = {timestamp_col, model_id_col, prediction_col, model_version_col, prediction_score_col, source_label_col}
    features = [column for column in columns if column not in reserved and column != entity_id_col]
    categorical = [
        column for column in features
        if any(token in column.lower() for token in ("country", "segment", "region", "category", "type"))
    ]
    slices = categorical[:]
    table_leaf = table_name.split(".")[-1] if table_name else "monitor"
    display_name = table_leaf.replace("_", " ").title()
    model_key = re.sub(r"[^a-zA-Z0-9_]", "_", table_leaf).lower()
    return {
        "display_name": display_name,
        "model_key": model_key,
        "timestamp_col": timestamp_col,
        "model_id_col": model_id_col,
        "prediction_col": prediction_col,
        "model_version_col": model_version_col,
        "prediction_score_col": prediction_score_col if prediction_score_col != prediction_col else "",
        "entity_id_col": entity_id_col,
        "source_label_col": source_label_col if source_label_col not in {timestamp_col, model_id_col, prediction_col} else "",
        "features": features,
        "categorical": [column for column in categorical if column in features],
        "slices": [column for column in slices if column in features],
    }


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


def _status_alert(message: str, color: str = "info") -> dbc.Alert:
    return dbc.Alert(message, color=color, className="py-2 mb-3")


def _status_block(items: list[tuple[str, str]]) -> html.Div:
    return html.Div([_status_alert(message, color) for message, color in items])


def _deployment_mode_prompt() -> dbc.Alert:
    instances = _workspace_lakebase_instances()
    if settings.use_lakebase_read_model:
        detail = settings.lakebase_database_name or settings.lakebase_instance_name or "managed resource"
        return _status_alert(
            f"Lakebase mode is active. Model Lens will prefer the Lakebase read model for monitor summaries and incidents ({detail}).",
            "success",
        )
    if instances:
        preview = ", ".join(instances[:3])
        return _status_alert(
            "Warehouse-only mode is active, but Lakebase appears to be available in this workspace "
            f"({preview}). Redeploy with the Lakebase-enabled target to accelerate the UI.",
            "info",
        )
    return _status_alert(
        "Warehouse-only mode is active. If Lakebase is added later, redeploy with the Lakebase-enabled target for faster monitor and incident views.",
        "secondary",
    )


def _schema_frame(scan_data: dict | None) -> pd.DataFrame:
    if not scan_data or not scan_data.get("schema"):
        return pd.DataFrame(columns=["col_name", "data_type"])
    return pd.DataFrame(scan_data["schema"])


def _schema_types(scan_data: dict | None) -> dict[str, str]:
    frame = _schema_frame(scan_data)
    if frame.empty or "col_name" not in frame.columns:
        return {}
    return {
        str(row["col_name"]): str(row.get("data_type") or "")
        for _, row in frame.iterrows()
    }


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


def _build_layout() -> html.Div:
    form_style = {"marginBottom": "0.75rem"}
    return html.Div([
        dcc.Store(id="scan-data"),
        dcc.Store(id="reload-token", data=0),
        html.H2(settings.app_title, className="mb-2"),
        html.P(
            "Set up the control plane, scan a source table, create a monitor, and run a refresh from one place.",
            className="text-muted mb-4",
        ),
        _deployment_mode_prompt(),
        html.Div(id="action-status"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Workspace Setup", className="mb-3"),
                html.P(
                    "Create the control-plane catalog/schema/tables before onboarding the first model.",
                    className="text-muted",
                ),
                dbc.Button("Setup Control Plane", id="setup-control-plane-btn", color="primary", className="me-2"),
                dbc.Button("Refresh All Monitors", id="refresh-all-btn", color="secondary"),
                html.Hr(),
                dbc.Label("Refresh One Monitor"),
                dcc.Dropdown(id="refresh-monitor-select", options=[], placeholder="Select monitor..."),
                dbc.Button("Refresh Selected Monitor", id="refresh-selected-btn", color="secondary", className="mt-2"),
            ])), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Deployment Inputs", className="mb-3"),
                _render_frame(pd.DataFrame([
                    {"name": "DEPLOYMENT_MODE", "value": "lakebase" if settings.use_lakebase_read_model else "warehouse_only"},
                    {"name": "CONTROL_PLANE_CATALOG", "value": settings.control_plane_catalog},
                    {"name": "CONTROL_PLANE_SCHEMA", "value": settings.control_plane_schema},
                    {"name": "SQL_WAREHOUSE_ID", "value": settings.sql_warehouse_id or "(missing)"},
                    {"name": "USE_LAKEBASE_READ_MODEL", "value": str(settings.use_lakebase_read_model).lower()},
                    {"name": "LAKEBASE_DATABASE_NAME", "value": settings.lakebase_database_name or "(managed resource)"},
                    {"name": "LAKEBASE_SCHEMA", "value": settings.lakebase_schema},
                    {"name": "REFRESH_JOB_ID", "value": settings.refresh_job_id or "(optional)"},
                ]), empty_message=""),
            ])), md=8),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("1. Scan Source Table", className="mb-3"),
                dbc.InputGroup([
                    dbc.Input(id="source-table-input", placeholder="catalog.schema.inference_logs"),
                    dbc.Button("Scan", id="scan-source-btn", color="primary"),
                ], className="mb-3"),
                html.Div(id="scan-status"),
                html.Div(id="scan-preview"),
            ])), md=12),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("2. Create Or Update Monitor", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Display Name"),
                        dbc.Input(id="display-name-input"),
                    ], md=6, style=form_style),
                    dbc.Col([
                        dbc.Label("Model Key"),
                        dbc.Input(id="model-key-input"),
                    ], md=6, style=form_style),
                ]),
                dbc.Row([
                    dbc.Col([dbc.Label("Timestamp Column"), dcc.Dropdown(id="timestamp-col-dropdown")], md=4, style=form_style),
                    dbc.Col([dbc.Label("Model ID Column"), dcc.Dropdown(id="model-id-col-dropdown")], md=4, style=form_style),
                    dbc.Col([dbc.Label("Prediction Column"), dcc.Dropdown(id="prediction-col-dropdown")], md=4, style=form_style),
                ]),
                dbc.Row([
                    dbc.Col([dbc.Label("Model Version Column"), dcc.Dropdown(id="model-version-col-dropdown")], md=4, style=form_style),
                    dbc.Col([dbc.Label("Prediction Score Column"), dcc.Dropdown(id="prediction-score-col-dropdown")], md=4, style=form_style),
                    dbc.Col([dbc.Label("Entity ID Column"), dcc.Dropdown(id="entity-id-col-dropdown")], md=4, style=form_style),
                ]),
                dbc.Row([
                    dbc.Col([dbc.Label("Label Column In Source"), dcc.Dropdown(id="source-label-col-dropdown")], md=4, style=form_style),
                    dbc.Col([dbc.Label("External Labels Table"), dbc.Input(id="labels-table-input", placeholder="catalog.schema.labels_table")], md=4, style=form_style),
                    dbc.Col([dbc.Label("External Labels Join Column"), dbc.Input(id="labels-join-col-input", placeholder="entity_id")], md=4, style=form_style),
                ]),
                dbc.Row([
                    dbc.Col([dbc.Label("External Label Column"), dbc.Input(id="external-label-col-input", placeholder="label")], md=4, style=form_style),
                    dbc.Col([dbc.Label("Problem Type"), dcc.Dropdown(id="problem-type-dropdown", options=[
                        {"label": "Classification", "value": "classification"},
                        {"label": "Regression", "value": "regression"},
                    ], value="classification")], md=4, style=form_style),
                    dbc.Col([dbc.Label("Baseline Days"), dbc.Input(id="baseline-days-input", type="number", min=1, value=7)], md=4, style=form_style),
                ]),
                dbc.Label("Feature Columns"),
                dcc.Dropdown(id="feature-cols-dropdown", multi=True, className="mb-3"),
                dbc.Label("Categorical Columns"),
                dcc.Dropdown(id="categorical-cols-dropdown", multi=True, className="mb-3"),
                dbc.Label("Slice Columns"),
                dcc.Dropdown(id="slice-cols-dropdown", multi=True, className="mb-3"),
                dbc.Button("Save Monitor And Run Initial Refresh", id="save-monitor-btn", color="success"),
            ])), md=12),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Monitors", className="mb-3"),
                html.Div(id="monitor-summary"),
            ])), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Open Incidents", className="mb-3"),
                html.Div(id="incident-summary"),
            ])), md=5),
        ], className="g-3"),
    ], style={"padding": "24px"})


def create_app() -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
    app.layout = _build_layout()

    @app.callback(
        Output("scan-data", "data"),
        Output("scan-status", "children"),
        Output("scan-preview", "children"),
        Input("scan-source-btn", "n_clicks"),
        State("source-table-input", "value"),
        prevent_initial_call=True,
    )
    def scan_source_table(_, source_table):
        if not source_table:
            return dash.no_update, _status_alert("Enter a fully qualified source table name.", "warning"), dash.no_update
        try:
            columns, preview, schema = _repo().scan_source_table(source_table.strip())
        except Exception as error:
            return dash.no_update, _status_alert(f"Scan failed: {error}", "danger"), html.Div()
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
        }
        status = _status_block([
            (
                f"Scanned {source_table.strip()} with {len(columns)} columns. "
                f"Detected {numeric_count} numeric columns that can participate in the current drift engine.",
                "success",
            ),
        ])
        preview_div = html.Div([
            html.H6("Column Types", className="mb-2"),
            _render_frame(schema[[column for column in ("col_name", "data_type") if column in schema.columns]], "No schema metadata returned."),
            html.Hr(),
            html.H6("Sample Rows", className="mb-2"),
            _render_frame(preview, "No preview rows returned.", max_rows=5),
        ])
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
        Output("feature-cols-dropdown", "options"),
        Output("feature-cols-dropdown", "value"),
        Output("categorical-cols-dropdown", "options"),
        Output("categorical-cols-dropdown", "value"),
        Output("slice-cols-dropdown", "options"),
        Output("slice-cols-dropdown", "value"),
        Input("scan-data", "data"),
    )
    def populate_monitor_form(scan_data):
        if not scan_data:
            empty = [], ""
            return "", "", *empty, *empty, *empty, [{"label": "(none)", "value": ""}], "", [{"label": "(none)", "value": ""}], "", [{"label": "(none)", "value": ""}], "", [{"label": "(none)", "value": ""}], "", [], [], [], [], [], []
        columns = scan_data.get("columns", [])
        defaults = _guess_defaults(scan_data.get("table_name", ""), columns)
        required_options = _option_list(columns)
        optional_options = _option_list(columns, include_blank=True)
        feature_options = _option_list(defaults["features"])
        categorical_options = _option_list(defaults["features"])
        slice_options = _option_list(defaults["features"])
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
            feature_options,
            defaults["features"],
            categorical_options,
            defaults["categorical"],
            slice_options,
            defaults["slices"],
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
        Input("feature-cols-dropdown", "value"),
        State("categorical-cols-dropdown", "value"),
        State("slice-cols-dropdown", "value"),
        prevent_initial_call=True,
    )
    def sync_feature_dependent_options(feature_columns, categorical_columns, slice_columns):
        options = _option_list(feature_columns or [])
        allowed = set(feature_columns or [])
        categorical_values = [column for column in (categorical_columns or []) if column in allowed]
        slice_values = [column for column in (slice_columns or []) if column in allowed]
        return options, categorical_values, options, slice_values

    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Input("setup-control-plane-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def setup_control_plane(_):
        try:
            _repo().ensure_control_plane()
        except Exception as error:
            return _status_alert(f"Setup failed: {error}", "danger"), dash.no_update
        return _status_alert(
            f"Control plane ready at {settings.control_plane_catalog}.{settings.control_plane_schema}.",
            "success",
        ), datetime.now(timezone.utc).isoformat(timespec="seconds")

    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Input("refresh-all-btn", "n_clicks"),
        Input("refresh-selected-btn", "n_clicks"),
        State("refresh-monitor-select", "value"),
        prevent_initial_call=True,
    )
    def refresh_monitors(_, __, selected_model):
        trigger = ctx.triggered_id
        model_key = selected_model if trigger == "refresh-selected-btn" else ""
        if trigger == "refresh-selected-btn" and not model_key:
            return _status_alert("Select a monitor before running a targeted refresh.", "warning"), dash.no_update
        try:
            counts = run_refresh_cycle(_repo(), model_key=model_key or "")
        except Exception as error:
            return _status_alert(f"Refresh failed: {error}", "danger"), dash.no_update
        scope = model_key or "all active monitors"
        color = "success" if counts.models else "warning"
        message = (
            f"Refresh complete for {scope}: models={counts.models}, drift_rows={counts.drift_rows}, "
            f"quality_rows={counts.quality_rows}, performance_rows={counts.performance_rows}, incidents={counts.incident_rows}."
        )
        if not counts.models:
            message += " No monitor produced a comparable baseline/current window."
        return _status_alert(message, color), datetime.now(timezone.utc).isoformat(timespec="seconds")

    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Output("reload-token", "data", allow_duplicate=True),
        Input("save-monitor-btn", "n_clicks"),
        State("scan-data", "data"),
        State("display-name-input", "value"),
        State("model-key-input", "value"),
        State("timestamp-col-dropdown", "value"),
        State("model-id-col-dropdown", "value"),
        State("prediction-col-dropdown", "value"),
        State("model-version-col-dropdown", "value"),
        State("prediction-score-col-dropdown", "value"),
        State("entity-id-col-dropdown", "value"),
        State("source-label-col-dropdown", "value"),
        State("labels-table-input", "value"),
        State("labels-join-col-input", "value"),
        State("external-label-col-input", "value"),
        State("feature-cols-dropdown", "value"),
        State("categorical-cols-dropdown", "value"),
        State("slice-cols-dropdown", "value"),
        State("problem-type-dropdown", "value"),
        State("baseline-days-input", "value"),
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
        model_version_col,
        prediction_score_col,
        entity_id_col,
        source_label_col,
        labels_table,
        labels_join_col,
        external_label_col,
        feature_columns,
        categorical_columns,
        slice_columns,
        problem_type,
        baseline_days,
    ):
        if not scan_data:
            return _status_alert("Scan a source table before saving a monitor.", "warning"), dash.no_update
        if not feature_columns:
            return _status_alert("Select at least one feature column.", "warning"), dash.no_update
        labels_table = (labels_table or "").strip()
        external_label_col = (external_label_col or "").strip()
        labels_join_col = (labels_join_col or "").strip()
        label_col = external_label_col or source_label_col or None
        if labels_table and (not entity_id_col or not labels_join_col or not label_col):
            return _status_alert(
                "External labels require Entity ID Column, External Labels Join Column, and External Label Column.",
                "warning",
            ), dash.no_update
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
                baseline=build_default_baseline(int(baseline_days or 7)),
                problem_type=problem_type or "classification",
                labels_table=labels_table or None,
                labels_join_col=labels_join_col or None,
                created_by="app",
            )
            repository = _repo()
            repository.upsert_monitor_config(config)
            counts = run_refresh_cycle(repository, model_key=config.model_key)
        except Exception as error:
            return _status_alert(f"Save failed: {error}", "danger"), dash.no_update
        messages = [(
            f"Saved monitor {config.model_key} and ran initial refresh: "
            f"drift_rows={counts.drift_rows}, quality_rows={counts.quality_rows}, "
            f"performance_rows={counts.performance_rows}, incidents={counts.incident_rows}.",
            "success" if counts.models else "warning",
        )]
        if not counts.models:
            messages.append((
                "The monitor was saved, but the current source data did not produce a comparable baseline/current window yet.",
                "warning",
            ))
        non_numeric = _non_numeric_features(feature_columns or [], scan_data)
        if non_numeric:
            messages.append((
                "These selected feature columns are not numeric and will be stored in the contract "
                f"but skipped by the current drift engine: {', '.join(non_numeric)}.",
                "warning",
            ))
        return _status_block(messages), datetime.now(timezone.utc).isoformat(timespec="seconds")

    @app.callback(
        Output("monitor-summary", "children"),
        Output("incident-summary", "children"),
        Output("refresh-monitor-select", "options"),
        Input("reload-token", "data"),
    )
    def render_dashboard(_):
        try:
            repository = _repo()
            configs = repository.list_monitor_configs(status="active")
            config_frame = pd.DataFrame([
                {
                    "model_key": config.model_key,
                    "display_name": config.display_name,
                    "source_table": config.source_table,
                    "features": len(config.contract.feature_columns),
                    "labels_table": config.labels_table or "",
                }
                for config in configs
            ])
            summary = repository.get_monitor_summary()
            incidents = repository.get_open_incidents()
        except Exception as error:
            alert = _status_alert(f"Unable to load control-plane state: {error}", "warning")
            return alert, html.Div(), []

        if not config_frame.empty and not summary.empty:
            monitors = config_frame.merge(summary, on=["model_key", "display_name"], how="left")
        else:
            monitors = config_frame
        preferred_columns = [
            "display_name",
            "model_key",
            "source_table",
            "features",
            "feature_count",
            "max_psi",
            "latest_window_end",
            "latest_data_date",
            "total_rows",
            "last_refresh_at",
            "open_incident_count",
            "labels_table",
        ]
        monitors = monitors[[column for column in preferred_columns if column in monitors.columns]]
        options = [{"label": config.display_name, "value": config.model_key} for config in configs]
        return (
            _render_frame(monitors, "No active monitors yet."),
            _render_frame(incidents, "No open incidents."),
            options,
        )

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=settings.app_port, debug=False)


if __name__ == "__main__":
    main()
