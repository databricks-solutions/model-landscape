from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_lens.config import settings
from model_lens.ui.components import make_wizard_step


STEP_LABELS = ["Workspace", "Source", "Contract", "Review"]


def _workspace_step(form_style: dict) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Workspace Setup", className="mb-3"),
                            html.P(
                                "Point Model Lens at the control-plane namespace for this workspace. "
                                "If Lakebase is available, you can also configure the instance and database here "
                                "so the app can use the faster UI read model.",
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
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Lakebase Instance Name (Optional)"),
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
                                            dbc.Label("Lakebase Database Name (Optional)"),
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
                                className="mb-3",
                            ),
                            dbc.Button("Setup Control Plane", id="setup-control-plane-btn", color="primary", className="me-2"),
                            dbc.Button("Refresh All Monitors", id="refresh-all-btn", color="secondary"),
                            html.Hr(),
                            dbc.Label("Refresh One Monitor"),
                            dcc.Dropdown(id="refresh-monitor-select", options=[], placeholder="Select monitor..."),
                            dbc.Button("Refresh Selected Monitor", id="refresh-selected-btn", color="secondary", className="mt-2"),
                        ]
                    )
                ),
                lg=7,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Runtime Defaults", className="mb-3"),
                            html.Div(id="onboarding-runtime-defaults"),
                        ]
                    )
                ),
                lg=5,
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
                                "and MLflow experiment or registered model, then let Model Lens infer the contract and scope.",
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
                            html.H5("Create Or Update Monitor", className="mb-3"),
                            html.P(
                                "Map source columns into the stable monitoring contract. "
                                "This is also where you scope shared inference tables to one model or version.",
                                className="text-muted",
                            ),
                            dbc.Alert(
                                "Model Lens auto-detects the core contract from the scanned table. "
                                "Most customers should only need the fields below. Use Advanced only if the inferred mapping is wrong.",
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
                                    dbc.Col([dbc.Label("Baseline Days"), dbc.Input(id="baseline-days-input", type="number", min=1, value=7)], md=6, style=form_style),
                                ]
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
                            html.H5("Review Monitor Configuration", className="mb-3"),
                            html.P(
                                "Review the namespace, source table, contract, and label strategy before activating the monitor.",
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
                            html.H5("Activation Notes", className="mb-3"),
                            html.Ul(
                                [
                                    html.Li("The monitor config is written into the control-plane namespace."),
                                    html.Li("The initial refresh computes drift, quality, and performance summaries."),
                                    html.Li("If Lakebase is configured, the UI projection is synchronized after refresh."),
                                    html.Li("If recent data does not yet produce a comparable baseline/current window, the monitor is still saved."),
                                ],
                                className="text-muted mb-0",
                            ),
                        ]
                    )
                ),
                lg=5,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody([html.H5("Monitors", className="mb-3"), html.Div(id="monitor-summary")])),
                lg=7,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody([html.H5("Open Incidents", className="mb-3"), html.Div(id="incident-summary")])),
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
            dcc.Store(id="onboarding-current-step", storage_type="session", data=1),
            dcc.Store(id="control-plane-ready-store", storage_type="session", data={}),
            html.H4("Model Onboarding", className="text-light mb-1"),
            html.P(
                "Create the control plane, scan a source table, map the contract, and run the first refresh.",
                className="text-muted mb-4",
            ),
            html.Div(
                id="wizard-steps-indicator",
                children=[make_wizard_step(index + 1, label, 1) for index, label in enumerate(STEP_LABELS)],
                className="d-flex justify-content-center flex-wrap gap-2 mb-4",
            ),
            html.Div(id="wizard-step-guidance", className="mb-3"),
            html.Div(id="action-status"),
            html.Div(id="wizard-step-workspace", children=_workspace_step(form_style), className="mb-4"),
            html.Div(id="wizard-step-source", children=_source_step(), className="mb-4"),
            html.Div(id="wizard-step-contract", children=_contract_step(form_style), className="mb-4"),
            html.Div(id="wizard-step-review", children=_review_step(), className="mb-4"),
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
