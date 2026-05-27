"""Shared Plotly layout for mlflow-lens panels.

Every quick function calls :func:`lens_layout` so the gallery looks coherent.
The theme is plotly-only; mlflow-lens does not depend on matplotlib.
"""

from __future__ import annotations

from typing import Any

LENS_COLORWAY: tuple[str, ...] = (
    "#FF3621",  # Databricks orange
    "#1B3139",  # ink
    "#00A972",  # green
    "#0073E6",  # blue
    "#FFAB00",  # amber
    "#8A2BE2",  # violet
    "#E91E63",  # magenta
    "#00BCD4",  # cyan
)

LENS_DIAGONAL_COLOR = "#9CA3AF"
LENS_GRID_COLOR = "#E5E7EB"
LENS_PAPER_BG = "white"


def lens_layout(
    title: str | None = None,
    *,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return a layout dict suitable for ``fig.update_layout(**lens_layout(...))``.

    Args:
        title: Optional figure title (rendered bold).
        xaxis_title: Optional x-axis label.
        yaxis_title: Optional y-axis label.
        **overrides: Additional layout keys to merge on top of the defaults.

    Returns:
        A dict that callers pass straight to ``fig.update_layout``.
    """
    layout: dict[str, Any] = {
        "template": "plotly_white",
        "colorway": list(LENS_COLORWAY),
        "margin": dict(l=60, r=20, t=60, b=60),
        "font": dict(family="Inter, system-ui, sans-serif", size=13, color="#1B3139"),
        "hovermode": "closest",
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0)",
        ),
        "paper_bgcolor": LENS_PAPER_BG,
        "plot_bgcolor": LENS_PAPER_BG,
        "xaxis": dict(gridcolor=LENS_GRID_COLOR, zerolinecolor=LENS_GRID_COLOR),
        "yaxis": dict(gridcolor=LENS_GRID_COLOR, zerolinecolor=LENS_GRID_COLOR),
    }

    if title is not None:
        layout["title"] = dict(text=f"<b>{title}</b>", x=0.02, xanchor="left")
    if xaxis_title is not None:
        layout["xaxis"] = {**layout["xaxis"], "title": xaxis_title}
    if yaxis_title is not None:
        layout["yaxis"] = {**layout["yaxis"], "title": yaxis_title}

    for key, value in overrides.items():
        if key in {"xaxis", "yaxis", "legend", "margin", "font"} and isinstance(value, dict):
            layout[key] = {**layout.get(key, {}), **value}
        else:
            layout[key] = value

    return layout
