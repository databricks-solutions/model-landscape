from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_lens.config import settings
from model_lens.ui.components import make_wizard_step


STEP_LABELS = ["Setup", "Discover", "Confirm", "Activate"]


def _workspace_step(form_style: dict) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Workspace Setup", className="mb-3"),
                            html.P(
                                "Set the control-plane namespace once for this workspace. "
                                "Most teams only need the catalog, schema, and setup button.",
                                className="text-muted",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Control Plane Catalog"),
                                            dbc.Input(id="control-plane-catalog-input", value=settings.control_plane_catalog),
                                        ],
                                        md=6,
                                        style=form_style,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Control Plane Schema"),
                                            dbc.Input(id="control-plane-schema-input", value=settings.control_plane_schema),
                                        ],
                                        md=6,
                                        style=form_style,
                                    ),
                                ]
                            ),
                            html.Details(
                                [
                                    html.Summary("Permission checklist", className="fw-semibold"),
                                    html.Ul(
                                        [
                                            html.Li(
                                                "App service principal: CAN_USE on the SQL warehouse, "
                                                "source-data USE CATALOG / USE SCHEMA / SELECT, and "
                                                "control-plane USE CATALOG / USE SCHEMA / SELECT / MODIFY."
                                            ),
                                            html.Li(
                                                "If setup should create objects: CREATE TABLE in the control-plane schema, "
                                                "CREATE SCHEMA if the schema is missing, and CREATE CATALOG only when you enable the toggle."
                                            ),
                                            html.Li(
                                                "Refresh workflow identity: the same warehouse, source-data, and control-plane access as the app."
                                            ),
                                            html.Li(
                                                "Optional MLflow discovery: read access to the target experiment or registered model."
                                            ),
                                            html.Li(
                                                "Optional Lakebase reads or sync: permission to resolve the Lakebase instance and connect to the target database; "
                                                "workflow sync also needs write access to the Lakebase schema."
                                            ),
                                        ],
                                        className="text-muted small mt-2 mb-0",
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Button("Setup Control Plane", id="setup-control-plane-btn", color="primary", className="mt-2"),
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Only expand these fields if you want Lakebase-backed reads or need setup to create the catalog.",
                                                className="text-muted",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Lakebase Instance Name"),
                                                            dbc.Input(
                                                                id="lakebase-instance-input",
                                                                value=settings.lakebase_instance_name,
                                                                placeholder="customer-lakebase-instance",
                                                            ),
                                                        ],
                                                        md=4,
                                                        style=form_style,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Lakebase Database Name"),
                                                            dbc.Input(
                                                                id="lakebase-database-input",
                                                                value=settings.lakebase_database_name,
                                                                placeholder="model_lens_ui",
                                                            ),
                                                        ],
                                                        md=4,
                                                        style=form_style,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Lakebase Schema"),
                                                            dbc.Input(id="lakebase-schema-input", value=settings.lakebase_schema),
                                                        ],
                                                        md=4,
                                                        style=form_style,
                                                    ),
                                                ]
                                            ),
                                            dbc.Checklist(
                                                id="create-catalog-toggle",
                                                options=[
                                                    {
                                                        "label": "Create catalog if missing (requires elevated privileges)",
                                                        "value": "create_catalog",
                                                    }
                                                ],
                                                value=[],
                                                switch=True,
                                                className="mb-0",
                                            ),
                                        ],
                                        title="Advanced workspace options",
                                    )
                                ],
                                start_collapsed=True,
                                className="mt-4",
                            ),
                        ]
                    )
                ),
                lg=12,
            ),
        ],
        className="g-3",
    )


def _source_step() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Discover Monitor Draft", className="mb-3"),
                            html.P(
                                "Paste the inference table you want to monitor. You can optionally add a labels table "
                                "and MLflow experiment or registered model, then let Model Lens infer the draft for you.",
                                className="text-muted",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="source-table-input", placeholder="catalog.schema.inference_logs"),
                                    dbc.Button("Discover", id="scan-source-btn", color="primary"),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Optional Labels Table"),
                                            dbc.Input(id="labels-table-input", placeholder="catalog.schema.labels"),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Optional MLflow Experiment"),
                                            dbc.Input(id="mlflow-experiment-input", placeholder="/Users/name/fraud-monitoring"),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Optional Registered Model"),
                                            dbc.Input(id="mlflow-registered-model-input", placeholder="catalog.schema.fraud_model"),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="g-3 mb-3",
                            ),
                            html.Div(id="scan-status"),
                            html.Div(id="scan-preview"),
                        ]
                    )
                ),
                width=12,
            )
        ],
        className="g-3",
    )


def _contract_step(form_style: dict) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Confirm Monitor Draft", className="mb-3"),
                            html.P(
                                "Review the inferred monitor name, problem type, and baseline. "
                                "Only open Advanced if the draft needs manual correction.",
                                className="text-muted",
                            ),
                            dbc.Alert(
                                "Discovery now fills the stable monitoring contract automatically. "
                                "Most customers should be able to confirm the draft and continue.",
                                color="secondary",
                                className="py-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Display Name"), dbc.Input(id="display-name-input")], md=6, style=form_style),
                                    dbc.Col([dbc.Label("Model Key"), dbc.Input(id="model-key-input")], md=6, style=form_style),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Problem Type"),
                                            dcc.Dropdown(
                                                id="problem-type-dropdown",
                                                options=[
                                                    {"label": "Classification", "value": "classification"},
                                                    {"label": "Regression", "value": "regression"},
                                                ],
                                                value="classification",
                                            ),
                                        ],
                                        md=6,
                                        style=form_style,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Baseline Policy"),
                                            dbc.RadioItems(
                                                id="baseline-kind-input",
                                                options=[
                                                    {"label": "Rolling", "value": "rolling"},
                                                    {"label": "Fixed", "value": "fixed"},
                                                ],
                                                value="rolling",
                                                inline=True,
                                                className="pt-2",
                                            ),
                                        ],
                                        md=6,
                                        style=form_style,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    dbc.Label("Baseline Days"),
                                    dbc.Input(id="baseline-days-input", type="number", min=1, value=7),
                                ],
                                id="baseline-days-wrapper",
                                style=form_style,
                            ),
                            html.Div(
                                [
                                    dbc.Label("Fixed Baseline Date Range"),
                                    dcc.DatePickerRange(
                                        id="baseline-fixed-range-input",
                                        display_format="YYYY-MM-DD",
                                        minimum_nights=0,
                                        clearable=True,
                                    ),
                                    html.Small(
                                        "Model Lens compares this fixed known-good window to the latest window of the same length.",
                                        className="text-muted d-block mt-2",
                                    ),
                                ],
                                id="baseline-fixed-range-wrapper",
                                style={**form_style, "display": "none"},
                            ),
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col([dbc.Label("Timestamp Column"), dcc.Dropdown(id="timestamp-col-dropdown")], md=4, style=form_style),
                                                    dbc.Col([dbc.Label("Model ID Column"), dcc.Dropdown(id="model-id-col-dropdown")], md=4, style=form_style),
                                                    dbc.Col([dbc.Label("Prediction Column"), dcc.Dropdown(id="prediction-col-dropdown")], md=4, style=form_style),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col([dbc.Label("Monitored Model ID Value"), dbc.Input(id="model-id-value-input", placeholder="fraud_model_v1")], md=6, style=form_style),
                                                    dbc.Col([dbc.Label("Monitored Model Version Value"), dbc.Input(id="model-version-value-input", placeholder="2026-03-01")], md=6, style=form_style),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col([dbc.Label("Model Version Column"), dcc.Dropdown(id="model-version-col-dropdown")], md=4, style=form_style),
                                                    dbc.Col([dbc.Label("Prediction Score Column"), dcc.Dropdown(id="prediction-score-col-dropdown")], md=4, style=form_style),
                                                    dbc.Col([dbc.Label("Entity ID Column"), dcc.Dropdown(id="entity-id-col-dropdown")], md=4, style=form_style),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col([dbc.Label("Label Column In Source"), dcc.Dropdown(id="source-label-col-dropdown")], md=4, style=form_style),
                                                    dbc.Col([dbc.Label("External Labels Join Column"), dbc.Input(id="labels-join-col-input", value="entity_id", placeholder="entity_id")], md=4, style=form_style),
                                                    dbc.Col([dbc.Label("External Label Column"), dbc.Input(id="external-label-col-input", value="label", placeholder="label")], md=4, style=form_style),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col([dbc.Label("External Labels Order Column"), dbc.Input(id="labels-order-col-input", value="label_timestamp", placeholder="label_timestamp")], md=4, style=form_style),
                                                    dbc.Col([dbc.Label("Feature Columns"), dcc.Dropdown(id="feature-cols-dropdown", multi=True)], md=8, style=form_style),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col([dbc.Label("Categorical Columns"), dcc.Dropdown(id="categorical-cols-dropdown", multi=True)], md=6, style=form_style),
                                                    dbc.Col([dbc.Label("Slice Columns"), dcc.Dropdown(id="slice-cols-dropdown", multi=True)], md=6, style=form_style),
                                                ]
                                            ),
                                        ],
                                        title="Advanced mappings and overrides",
                                    )
                                ],
                                className="mt-3",
                                start_collapsed=True,
                            ),
                        ]
                    )
                ),
                width=12,
            )
        ],
        className="g-3",
    )


def _review_step() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Activate Monitoring", className="mb-3"),
                            html.P(
                                "Review the final draft and activate monitoring for this model.",
                                className="text-muted",
                            ),
                            html.Div(id="onboarding-review-summary"),
                            dbc.Button(
                                "Save Monitor And Run Initial Refresh",
                                id="save-monitor-btn",
                                color="success",
                                className="mt-3",
                            ),
                        ]
                    )
                ),
                lg=7,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("What Happens Next", className="mb-3"),
                            html.Ul(
                                [
                                    html.Li("The monitor config is written into the control-plane namespace."),
                                    html.Li("The initial refresh computes drift, quality, and performance summaries."),
                                    html.Li("If Lakebase is configured, the UI projection is synchronized after refresh."),
                                    html.Li("After activation, use Overview and the analysis pages to inspect the monitor."),
                                ],
                                className="text-muted mb-0",
                            ),
                        ]
                    )
                ),
                lg=5,
            ),
        ],
        className="g-3",
    )


def layout():
    form_style = {"marginBottom": "0.75rem"}
    return html.Div(
        [
            dcc.Store(id="scan-data"),
            dcc.Store(id="onboarding-current-step", data=1),
            dcc.Store(id="control-plane-ready-store", data={}),
            html.H4("Add Monitor", className="text-light mb-1"),
            html.P(
                "Set up the workspace once, discover a monitor draft, confirm it, and activate monitoring.",
                className="text-muted mb-4",
            ),
            html.Div(
                id="wizard-steps-indicator",
                children=[make_wizard_step(index + 1, label, 1) for index, label in enumerate(STEP_LABELS)],
                className="d-flex justify-content-center flex-wrap gap-2 mb-4",
            ),
            html.Div(id="wizard-step-guidance", className="mb-3"),
            html.Div(id="action-status"),
            html.Div(id="wizard-step-workspace", children=_workspace_step(form_style), className="mb-4", style={}),
            html.Div(id="wizard-step-source", children=_source_step(), className="mb-4", style={"display": "none"}),
            html.Div(id="wizard-step-contract", children=_contract_step(form_style), className="mb-4", style={"display": "none"}),
            html.Div(id="wizard-step-review", children=_review_step(), className="mb-4", style={"display": "none"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Back", id="wizard-back-btn", color="secondary", outline=True),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("Continue", id="wizard-next-btn", color="primary"),
                        width="auto",
                    ),
                ],
                className="g-2 justify-content-center",
            ),
        ]
    )
