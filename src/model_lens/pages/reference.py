from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout():
    return html.Div(
        [
            html.H4("Reference", className="text-light mb-1"),
            html.P(
                "Current monitor contract and latest summary for the selected model, plus global runtime settings.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Reference Filter"),
                            dbc.Select(
                                id="reference-monitor-status-filter",
                                options=[
                                    {"label": "Active", "value": "active"},
                                    {"label": "Archived", "value": "inactive"},
                                    {"label": "All", "value": "all"},
                                ],
                                value="active",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Reference Monitor"),
                            dcc.Dropdown(id="reference-monitor-select", placeholder="Select monitor..."),
                        ],
                        md=9,
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="reference-page-status", className="mb-3"),
            html.Div(id="reference-page-body"),
        ]
    )
