from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout():
    return html.Div(
        [
            html.H4("Data Quality", className="text-light mb-1"),
            html.Div(id="quality-model-banner", className="mb-3"),
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
