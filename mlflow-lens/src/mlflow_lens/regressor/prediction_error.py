"""Prediction error: y_true vs y_pred scatter with identity + best-fit lines."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

from mlflow_lens._plotly_theme import LENS_DIAGONAL_COLOR, lens_layout
from mlflow_lens._quickfn import quickfn


def _build(y_true: Any, y_pred: Any, **layout: Any) -> tuple:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred))

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    # Best-fit line via least squares
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    line_x = np.array([lo, hi])
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name=f"predictions (R²={r2:.3f})",
            marker=dict(size=6, opacity=0.6),
            hovertemplate="y_true=%{x:.3f}<br>y_pred=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name=f"best fit (slope={slope:.2f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color=LENS_DIAGONAL_COLOR, dash="dash"),
            name="identity",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Prediction Error"),
            xaxis_title="y_true",
            yaxis_title="y_pred",
            **layout,
        )
    )

    data = {
        "points": [{"y_true": float(t), "y_pred": float(p)} for t, p in zip(y_true, y_pred)],
        "r2": r2,
        "slope": float(slope),
        "intercept": float(intercept),
    }
    return fig, data


@quickfn(panel_type="prediction_error")
def prediction_error(
    model: Any,
    X: Any,
    y: Any,
    **layout: Any,
) -> tuple:
    """Scatter of true vs predicted with identity and best-fit lines."""
    y_pred = model.predict(X)
    return _build(y, y_pred, **layout)


@quickfn(panel_type="prediction_error")
def _from_predictions(y_true: Any, y_pred: Any, **layout: Any) -> tuple:
    """Prediction error from pre-computed predictions."""
    return _build(y_true, y_pred, **layout)


prediction_error.from_predictions = _from_predictions
