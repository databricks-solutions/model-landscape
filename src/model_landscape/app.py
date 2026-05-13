from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from model_landscape.callbacks import (
    _control_plane_ready,
    _feature_candidates,
    _non_numeric_features,
    _selected_model_from_search,
    _session_config,
    _workspace_lakebase_instances,
    register_callbacks,
)
from model_landscape.config import settings
from model_landscape.pages import data_quality, drift_analysis, feature_deep_dive, incidents, onboarding, overview, performance, reference
from model_landscape.ui.sidebar import build_sidebar
from model_landscape.ui.styles import INDEX_STRING


def _page_layout(pathname: str):
    if pathname == "/onboarding":
        return onboarding.layout()
    if pathname == "/drift":
        return drift_analysis.layout()
    if pathname == "/incidents":
        return incidents.layout()
    if pathname == "/features":
        return feature_deep_dive.layout()
    if pathname == "/performance":
        return performance.layout()
    if pathname == "/quality":
        return data_quality.layout()
    if pathname == "/reference":
        return reference.layout()
    return overview.layout()


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
        title=settings.app_title,
        suppress_callback_exceptions=True,
    )
    app.index_string = INDEX_STRING

    session_defaults = _session_config(None)

    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="session-config-store", data=session_defaults),
            dcc.Store(id="reload-token", data=0),
            build_sidebar(),
            html.Div(id="page-content", className="model-landscape-content"),
        ],
        className="model-landscape-shell",
    )

    app.validation_layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="session-config-store", data=session_defaults),
            dcc.Store(id="reload-token", data=0),
            build_sidebar(),
            html.Div(
                [
                    overview.layout(),
                    onboarding.layout(),
                    incidents.layout(),
                    drift_analysis.layout(),
                    feature_deep_dive.layout(),
                    performance.layout(),
                    data_quality.layout(),
                    reference.layout(),
                ],
                className="model-landscape-content",
            ),
        ],
        className="model-landscape-shell",
    )

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname):
        return _page_layout(pathname or "/")

    register_callbacks(app)
    return app


app = create_app()
server = app.server


def main() -> None:
    app.run(host="0.0.0.0", port=settings.app_port, debug=False)


if __name__ == "__main__":
    main()
