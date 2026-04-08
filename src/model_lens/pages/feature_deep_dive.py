from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_lens.ui.styles import DROPDOWN_STYLE


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
