from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_lens.config import settings
from model_lens.ui.styles import COLORS


NAV_ITEMS = [
    {"label": "Overview", "icon": "fas fa-chart-line", "href": "/"},
    {"label": "Onboarding", "icon": "fas fa-plus-circle", "href": "/onboarding"},
    {"label": "Drift Analysis", "icon": "fas fa-wave-square", "href": "/drift"},
    {"label": "Feature Deep Dive", "icon": "fas fa-search", "href": "/features"},
    {"label": "Performance", "icon": "fas fa-tachometer-alt", "href": "/performance"},
    {"label": "Data Quality", "icon": "fas fa-database", "href": "/quality"},
    {"label": "Reference", "icon": "fas fa-book", "href": "/reference"},
]


def build_sidebar():
    return html.Div(
        [
            html.Div(
                [
                    html.H5(settings.app_title, className="text-light mb-1"),
                    html.Small("Databricks-native model observability", className="text-muted d-block mb-2"),
                    html.Div(id="sidebar-mode-banner", className="mb-2"),
                    html.Hr(style={"borderColor": COLORS["grid"]}),
                    html.Small("Active Model", className="text-muted d-block mb-1"),
                    dcc.Dropdown(id="global-model-select", placeholder="Select model...", className="dash-dropdown mb-3"),
                    html.Div(id="sidebar-alert-badge", className="mt-1"),
                ]
            ),
            html.Div(
                [
                    dbc.Nav(
                        [
                            dbc.NavLink([html.I(className=f"{item['icon']} me-2"), item["label"]], href=item["href"], active="exact")
                            for item in NAV_ITEMS
                        ],
                        vertical=True,
                        pills=True,
                    ),
                ],
                className="model-lens-sidebar-nav",
            ),
            html.Div(
                [
                    html.Hr(style={"borderColor": COLORS["grid"]}),
                    html.Div(id="sidebar-status", className="mb-2"),
                    html.Small("Model Lens workspace control plane", className="text-muted"),
                ],
                className="model-lens-sidebar-footer",
            ),
        ],
        className="model-lens-sidebar",
    )
