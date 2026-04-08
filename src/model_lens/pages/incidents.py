from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_lens.ui.styles import DROPDOWN_STYLE


def layout():
    return html.Div(
        [
            html.H4("Incidents", className="text-light mb-1"),
            html.P(
                "Open incidents and recent incident lifecycle history across monitors.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Monitor", className="text-muted"),
                            dcc.Dropdown(id="incidents-monitor-filter", placeholder="All monitors", className="dash-dropdown"),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Severity", className="text-muted"),
                            dbc.Select(
                                id="incidents-severity-filter",
                                options=[
                                    {"label": "All", "value": "all"},
                                    {"label": "Critical", "value": "critical"},
                                    {"label": "Warning", "value": "warning"},
                                ],
                                value="all",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Status", className="text-muted"),
                            dbc.Select(
                                id="incidents-status-filter",
                                options=[
                                    {"label": "All", "value": "all"},
                                    {"label": "Open", "value": "open"},
                                    {"label": "Recovered", "value": "recovered"},
                                ],
                                value="all",
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Metric", className="text-muted"),
                            dcc.Dropdown(id="incidents-metric-filter", placeholder="All metrics", className="dash-dropdown"),
                        ],
                        md=3,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dcc.Loading(type="default", children=html.Div(id="incidents-page-body")),
        ]
    )
