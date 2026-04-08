from __future__ import annotations

from dash import dcc, html

from model_lens.ui.components import make_empty_state


def layout():
    return html.Div(
        [
            html.H4("Model Overview", className="text-light mb-3"),
            dcc.Loading(
                type="default",
                children=html.Div(
                    id="overview-page-body",
                    children=make_empty_state("Loading model overview...", icon="fas fa-chart-line"),
                ),
            ),
        ]
    )
