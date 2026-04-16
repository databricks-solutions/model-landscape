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
            dcc.Loading(
                type="default",
                children=html.Div(
                    [
                        html.Div(id="perf-labels-alert"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Primary Metric (Feature Impact)", className="text-muted"),
                                        dbc.Select(
                                            id="perf-metric-select",
                                            options=performance_metric_options("classification"),
                                            value=default_primary_performance_metric("classification"),
                                            style=DROPDOWN_STYLE,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Drift Metric (Trend Chart)", className="text-muted"),
                                        dbc.Select(
                                            id="perf-drift-metric-select",
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
                                        dbc.Label("Tracked Drift Features", className="text-muted"),
                                        dcc.Dropdown(
                                            id="perf-drift-feature-select",
                                            multi=True,
                                            placeholder="Select drift features...",
                                            className="dash-dropdown",
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Drift Threshold Guides", className="text-muted"),
                                        dbc.Switch(
                                            id="perf-drift-threshold-toggle",
                                            value=False,
                                            label="Show Threshold Guides",
                                        ),
                                        html.Small(
                                            "Guide visibility follows the selected drift metric and uses the monitor's saved thresholds.",
                                            className="text-muted d-block mt-1",
                                        ),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="mb-4",
                        ),
                        dbc.Row(id="perf-kpi-cards", className="mb-3"),
                        html.Div(id="perf-timeline-container", className="mb-3"),
                        html.Div(id="perf-contributors-container", className="mb-4"),
                        html.H6("Feature Deep Dive Shortcut", className="text-light mb-2"),
                        html.P(
                            "Use this shortcut to inspect the selected feature with configurable binning and outlier handling in Feature Deep Dive.",
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
                ),
            ),
        ]
    )
