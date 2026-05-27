from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_landscape.config import settings
from model_landscape.ui.styles import COLORS

PRIMARY_NAV_ITEMS = [
    {"label": "Overview", "icon": "fas fa-chart-line", "href": "/"},
    {"label": "Incidents", "icon": "fas fa-triangle-exclamation", "href": "/incidents"},
    {"label": "Drift Analysis", "icon": "fas fa-wave-square", "href": "/drift"},
    {"label": "Feature Deep Dive", "icon": "fas fa-search", "href": "/features"},
    {"label": "Performance", "icon": "fas fa-tachometer-alt", "href": "/performance"},
    {"label": "Data Quality", "icon": "fas fa-database", "href": "/quality"},
    {"label": "Monitor Settings", "icon": "fas fa-book", "href": "/reference"},
]


def build_sidebar():
    return html.Div(
        [
            html.Div(
                [
                    html.H5(settings.app_title, className="text-light mb-1"),
                    html.Small(
                        "Databricks-native model observability", className="text-muted d-block mb-2"
                    ),
                    html.Div(id="sidebar-mode-banner", className="mb-2"),
                    html.Hr(style={"borderColor": COLORS["grid"]}),
                    html.Small("Active Model", className="text-muted d-block mb-1"),
                    dcc.Dropdown(
                        id="global-model-select",
                        placeholder="Select model...",
                        className="dash-dropdown model-landscape-sidebar-dropdown mb-3",
                    ),
                    html.Div(id="sidebar-alert-badge", className="mt-1"),
                ]
            ),
            html.Div(
                [
                    dbc.Nav(
                        [
                            dbc.NavLink(
                                [html.I(className=f"{item['icon']} me-2"), item["label"]],
                                href=item["href"],
                                active="exact",
                            )
                            for item in PRIMARY_NAV_ITEMS
                        ],
                        vertical=True,
                        pills=True,
                        id="sidebar-primary-nav",
                    ),
                ],
                className="model-landscape-sidebar-nav",
            ),
            html.Div(
                [
                    dbc.NavLink(
                        [html.I(className="fas fa-plus-circle me-2"), "Add Monitor"],
                        href="/onboarding",
                        active="exact",
                        id="sidebar-onboarding-link",
                        className="model-landscape-sidebar-cta",
                    ),
                ],
                className="model-landscape-sidebar-bottom-action",
            ),
            html.Div(
                [
                    html.Hr(style={"borderColor": COLORS["grid"]}),
                    html.Div(id="sidebar-status", className="mb-2"),
                    html.Small("Model Landscape workspace control plane", className="text-muted"),
                ],
                className="model-landscape-sidebar-footer",
            ),
        ],
        className="model-landscape-sidebar",
    )
