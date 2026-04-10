from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_lens.ui.styles import DROPDOWN_STYLE


def layout():
    return html.Div(
        [
            html.H4("Data Quality", className="text-light mb-1"),
            html.Div(id="quality-model-banner", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Date Range", className="text-muted"),
                            dcc.DatePickerRange(
                                id="quality-date-range",
                                display_format="YYYY-MM-DD",
                                minimum_nights=0,
                                className="w-100",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Class Basis", className="text-muted"),
                            dbc.Select(
                                id="quality-class-basis-select",
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
                                id="quality-class-value-select",
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
                                id="quality-threshold-toggle",
                                value=False,
                                label="Show Threshold Guides",
                            ),
                            html.Small(
                                "Guide visibility is optional here, but thresholds still drive status and incident semantics.",
                                className="text-muted d-block mt-1",
                            ),
                        ],
                        md=2,
                    ),
                ],
                className="mb-4",
            ),
            dcc.Loading(
                type="default",
                children=html.Div(
                    [
                        dbc.Row(id="quality-kpi-cards", className="mb-3"),
                        html.Div(id="quality-volume-container", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="quality-null-rates-container"), md=6),
                                dbc.Col(html.Div(id="quality-prediction-container"), md=6),
                            ]
                        ),
                    ]
                ),
            ),
        ]
    )
