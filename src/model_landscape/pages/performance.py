from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_landscape.domain.performance_metrics import default_primary_performance_metric, performance_metric_options
from model_landscape.ui.styles import DROPDOWN_STYLE


def layout():
    return html.Div(
        [
            dcc.Store(id="perf-breakdown-state"),
            html.H4("Performance Degradation Analysis", className="text-light mb-1"),
            html.Div(id="perf-model-banner", className="mb-3"),
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
            dcc.Loading(
                type="default",
                children=html.Div(
                    [
                        dbc.Row(id="perf-kpi-cards", className="mb-3"),
                        html.Div(id="perf-timeline-container", className="mb-3"),
                        html.Div(id="perf-drift-container", className="mb-4"),
                    ]
                ),
            ),
            html.H6("Per-Bin Breakdown Controls", className="text-light mb-2"),
            html.P(
                "Use the same binning and outlier settings here as in Feature Deep Dive. These controls update the all-feature per-bin impact chart when you click Apply.",
                className="text-muted",
                style={"fontSize": "0.8rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Binning Mode", className="text-muted"),
                            dbc.Select(
                                id="perf-binning-mode-select",
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
                            dbc.Input(id="perf-bin-count-input", type="number", min=2, max=200, step=1, value=40),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Custom Bin Edges", className="text-muted"),
                            dbc.Input(
                                id="perf-custom-edges-input",
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
                                id="perf-outlier-mode-select",
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
                            dbc.Label("Outlier Parameter", id="perf-outlier-value-label", className="text-muted"),
                            dbc.Input(id="perf-outlier-value-input", type="number", min=0, step=0.1, value=1.0, disabled=True),
                            html.Small(
                                "Percentile Clip uses P / 100-P clipping. IQR Fence uses Q1 - K*IQR to Q3 + K*IQR.",
                                id="perf-outlier-value-help",
                                className="text-muted",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Apply Controls", className="text-muted"),
                            dbc.Button(
                                "Apply Breakdown Controls",
                                id="perf-apply-breakdown-controls-btn",
                                color="primary",
                                className="w-100",
                            ),
                            html.Small(
                                "Metric and drift controls update immediately. Binning and outlier settings apply when you click this button.",
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
                children=html.Div(id="perf-contributors-container", className="mb-4"),
            ),
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
    )
