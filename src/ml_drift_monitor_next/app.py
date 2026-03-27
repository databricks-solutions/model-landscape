from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import html

from ml_drift_monitor_next.config import settings


def build_layout() -> html.Div:
    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Bundle-managed"),
            html.P("App and refresh workflow are deployed via Databricks Asset Bundles."),
        ])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("One control plane"),
            html.P("No per-model job sprawl. One workflow refreshes active monitors."),
        ])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Canonical contract"),
            html.P("Customer tables map into one inference-log contract before monitoring."),
        ])), md=4),
    ], className="g-3")

    return html.Div([
        html.H2(settings.app_title, className="mb-3"),
        html.P(
            "This app is the control plane for customer-deployed model observability on Databricks.",
            className="text-muted",
        ),
        cards,
        html.Hr(),
        html.H4("Deployment Inputs"),
        dbc.Table.from_dataframe(
            __import__("pandas").DataFrame([
                {"name": "CONTROL_PLANE_CATALOG", "value": settings.control_plane_catalog},
                {"name": "CONTROL_PLANE_SCHEMA", "value": settings.control_plane_schema},
                {"name": "SQL_WAREHOUSE_ID", "value": settings.sql_warehouse_id or "(bind via valueFrom)"},
                {"name": "REFRESH_JOB_ID", "value": settings.refresh_job_id or "(bind via valueFrom)"},
            ]),
            striped=True,
            bordered=True,
            hover=True,
        ),
    ], style={"padding": "24px"})


def create_app() -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    app.layout = build_layout()
    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=settings.app_port, debug=False)


if __name__ == "__main__":
    main()

