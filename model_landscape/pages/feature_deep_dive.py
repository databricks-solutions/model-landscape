from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_landscape.ui.styles import DROPDOWN_STYLE


def layout():
    return html.Div(
        [
            html.H4("Feature Deep Dive", className="text-light mb-1"),
            html.Div(id="deepdive-model-banner", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Select Feature", className="text-muted"),
                            dcc.Dropdown(
                                id="deepdive-feature-select",
                                placeholder="Choose a feature to investigate...",
                                className="dash-dropdown",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Break down by dimension", className="text-muted"),
                            dcc.Dropdown(
                                id="deepdive-dimension-select",
                                placeholder="Select dimension...",
                                className="dash-dropdown",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Binning Mode", className="text-muted"),
                            dbc.Select(
                                id="deepdive-binning-mode-select",
                                options=[
                                    {"label": "Auto", "value": "auto"},
                                    {"label": "Fixed Bin Count", "value": "fixed"},
                                    {"label": "Custom Edges", "value": "custom"},
                                ],
                                value="auto",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Bin Count", className="text-muted"),
                            dbc.Input(id="deepdive-bin-count-input", type="number", min=2, max=200, step=1, value=40),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Custom Bin Edges", className="text-muted"),
                            dbc.Input(
                                id="deepdive-custom-edges-input",
                                type="text",
                                placeholder="e.g. 0, 10, 20, 50",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Outlier Mode", className="text-muted"),
                            dbc.Select(
                                id="deepdive-outlier-mode-select",
                                options=[
                                    {"label": "Off", "value": "off"},
                                    {"label": "Percentile Clip", "value": "percentile_clip"},
                                    {"label": "IQR Fence", "value": "iqr_fence"},
                                ],
                                value="off",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Outlier Parameter", id="deepdive-outlier-value-label", className="text-muted"),
                            dbc.Input(id="deepdive-outlier-value-input", type="number", min=0, step=0.1, value=1.0, disabled=True),
                            html.Small(
                                "Percentile Clip uses P / 100-P clipping. IQR Fence uses Q1 - K*IQR to Q3 + K*IQR.",
                                id="deepdive-outlier-value-help",
                                className="text-muted",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Apply Controls", className="text-muted"),
                            dbc.Button(
                                "Apply Distribution Controls",
                                id="deepdive-apply-controls-btn",
                                color="primary",
                                className="w-100",
                            ),
                            html.Small(
                                "Feature and dimension changes update immediately. Binning and outlier settings apply when you click this button.",
                                className="text-muted d-block mt-2",
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-4",
            ),
            dcc.Loading(
                type="default",
                children=html.Div(
                    [
                        html.Div(id="deepdive-distribution-container", className="mb-3"),
                        html.Div(id="deepdive-dimension-container", className="mb-3"),
                    ]
                ),
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6([html.I(className="fas fa-circle-info me-2"), "Feature Context"], className="text-light mb-2"),
                        html.Div(
                            id="deepdive-context-container",
                            children="Dimension breakdown uses the latest current window and the selected slice column from the monitor contract.",
                            className="text-muted",
                            style={"fontSize": "0.85rem"},
                        ),
                    ]
                ),
                className="mb-3",
            ),
        ]
    )
