from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_landscape.ui.styles import DROPDOWN_STYLE


def layout():
    return html.Div(
        [
            html.H4("Drift Analysis", className="text-light mb-1"),
            html.Div(id="drift-model-banner", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Metric", className="text-muted"),
                            dbc.Select(
                                id="drift-metric-select",
                                options=[
                                    {"label": "PSI", "value": "psi"},
                                    {"label": "Jensen-Shannon", "value": "js_divergence"},
                                    {"label": "KL Divergence", "value": "kl_divergence"},
                                ],
                                value="psi",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Granularity", className="text-muted"),
                            dbc.Select(
                                id="drift-granularity-select",
                                options=[
                                    {"label": "Daily", "value": "daily"},
                                    {"label": "Weekly", "value": "weekly"},
                                    {"label": "Monthly", "value": "monthly"},
                                ],
                                value="daily",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Top N Features", className="text-muted"),
                            dcc.Slider(
                                id="drift-top-n",
                                min=1,
                                max=50,
                                step=1,
                                value=10,
                                marks={
                                    1: "1",
                                    5: "5",
                                    10: "10",
                                    20: "20",
                                    30: "30",
                                    40: "40",
                                    50: "50",
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Date Range", className="text-muted"),
                            dcc.DatePickerRange(
                                id="drift-date-range",
                                display_format="YYYY-MM-DD",
                                minimum_nights=0,
                                className="w-100",
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Class Basis", className="text-muted"),
                            dbc.Select(
                                id="drift-class-basis-select",
                                options=[
                                    {"label": "All Rows", "value": "all"},
                                    {"label": "Actual Label", "value": "actual"},
                                    {"label": "Predicted Label", "value": "predicted"},
                                ],
                                value="all",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Class Value", className="text-muted"),
                            dbc.Select(
                                id="drift-class-value-select",
                                options=[
                                    {"label": "All Rows", "value": "all"},
                                    {"label": "Positive", "value": "positive"},
                                    {"label": "Negative", "value": "negative"},
                                ],
                                value="all",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Threshold Guides", className="text-muted"),
                            dbc.Switch(
                                id="drift-threshold-toggle",
                                value=False,
                                label="Show Threshold Guides",
                            ),
                            html.Small(
                                "Guide visibility is optional here, but thresholds still drive status and incident semantics.",
                                className="text-muted d-block mt-1",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Apply Filters", className="text-muted"),
                            dbc.Button(
                                "Apply Drift Filters",
                                id="drift-apply-filters-btn",
                                color="primary",
                                className="w-100",
                            ),
                            html.Small(
                                "Metric, date, class, granularity, and guide changes apply when you click this button.",
                                className="text-muted d-block mt-1",
                            ),
                        ],
                        md=2,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.Small(
                                "Edit warning and critical thresholds for drift and null-rate guides on this page. Saved values also drive Overview severity and future incidents.",
                                className="text-muted d-block mb-3",
                            ),
                            html.Div(id="drift-threshold-status", className="mb-2"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6("PSI", className="text-light mb-3"),
                                            dbc.Label("Warning", className="text-muted"),
                                            dbc.Input(
                                                id="drift-threshold-psi-warning-input",
                                                type="number",
                                                min=0,
                                                step=0.0001,
                                            ),
                                            dbc.Label("Critical", className="text-muted mt-2"),
                                            dbc.Input(
                                                id="drift-threshold-psi-critical-input",
                                                type="number",
                                                min=0,
                                                step=0.0001,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6("Jensen-Shannon", className="text-light mb-3"),
                                            dbc.Label("Warning", className="text-muted"),
                                            dbc.Input(
                                                id="drift-threshold-js_divergence-warning-input",
                                                type="number",
                                                min=0,
                                                step=0.0001,
                                            ),
                                            dbc.Label("Critical", className="text-muted mt-2"),
                                            dbc.Input(
                                                id="drift-threshold-js_divergence-critical-input",
                                                type="number",
                                                min=0,
                                                step=0.0001,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6("KL Divergence", className="text-light mb-3"),
                                            dbc.Label("Warning", className="text-muted"),
                                            dbc.Input(
                                                id="drift-threshold-kl_divergence-warning-input",
                                                type="number",
                                                min=0,
                                                step=0.0001,
                                            ),
                                            dbc.Label("Critical", className="text-muted mt-2"),
                                            dbc.Input(
                                                id="drift-threshold-kl_divergence-critical-input",
                                                type="number",
                                                min=0,
                                                step=0.0001,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6("Null Rate (%)", className="text-light mb-3"),
                                            dbc.Label("Warning", className="text-muted"),
                                            dbc.Input(
                                                id="drift-threshold-null_rate-warning-input",
                                                type="number",
                                                min=0,
                                                step=0.01,
                                            ),
                                            dbc.Label("Critical", className="text-muted mt-2"),
                                            dbc.Input(
                                                id="drift-threshold-null_rate-critical-input",
                                                type="number",
                                                min=0,
                                                step=0.01,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                ],
                                className="g-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Save Thresholds",
                                            id="drift-save-thresholds-btn",
                                            color="primary",
                                            className="w-100",
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Reset to Defaults",
                                            id="drift-reset-thresholds-btn",
                                            color="secondary",
                                            outline=True,
                                            className="w-100",
                                        ),
                                        md=3,
                                    ),
                                ],
                                className="g-3 mt-3",
                            ),
                        ],
                        title="Thresholds",
                    )
                ],
                start_collapsed=True,
                className="mb-4",
            ),
            dcc.Loading(
                type="default",
                children=html.Div(
                    [
                        html.Div(id="drift-heatmap-container", className="mb-3"),
                        html.Div(id="drift-categorical-note", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="drift-timeline-container"), md=7),
                                dbc.Col(html.Div(id="drift-top-drifters-container"), md=5),
                            ]
                        ),
                    ]
                ),
            ),
        ]
    )
