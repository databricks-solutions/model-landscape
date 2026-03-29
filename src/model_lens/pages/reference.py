from __future__ import annotations

from dash import html


def layout():
    return html.Div(
        [
            html.H4("Reference", className="text-light mb-1"),
            html.P("Current monitor contract, runtime settings, and refresh state.", className="text-muted"),
            html.Div(id="reference-page-body"),
        ]
    )

