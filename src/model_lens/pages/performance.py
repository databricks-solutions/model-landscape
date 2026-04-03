from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_lens.domain.performance_metrics import default_primary_performance_metric, performance_metric_options
from model_lens.ui.styles import DROPDOWN_STYLE


def layout():
    return html.Div(
        [
            html.H4("Performance Degradation Analysis", className="text-light mb-1"),
            html.Div(id="perf-model-banner", className="mb-3"),
            html.Div(id="perf-labels-alert"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Primary Metric", className="text-muted"),
                            dbc.Select(
                                id="perf-metric-select",
                                options=performance_metric_options("classification"),
                                value=default_primary_performance_metric("classification"),
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(id="perf-kpi-cards", className="mb-3"),
            html.Div(id="perf-timeline-container", className="mb-3"),
            html.Div(id="perf-contributors-container", className="mb-4"),
            html.H6("Bin-Level Drill Down", className="text-light mb-2"),
            html.P(
                "Select a feature to see baseline vs current performance for each value range.",
                className="text-muted",
                style={"fontSize": "0.8rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="perf-feature-select",
                                placeholder="Choose a feature...",
                                className="dash-dropdown",
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-2",
            ),
            html.Div(id="perf-bin-detail-container", className="mb-3"),
            html.Div(id="perf-date-range-note", className="mt-3"),
        ]
    )
