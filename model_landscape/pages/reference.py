from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout():
    return html.Div(
        [
            html.H4("Monitor Settings", className="text-light mb-1"),
            html.P(
                "Review the selected model contract, update monitoring settings, and manage lifecycle actions.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Status"),
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
                            dbc.Label("Monitor"),
                            dcc.Dropdown(
                                id="reference-monitor-select", placeholder="Select monitor..."
                            ),
                        ],
                        md=9,
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="reference-page-status", className="mb-3"),
            dcc.Loading(type="default", children=html.Div(id="reference-page-body")),
        ]
    )
