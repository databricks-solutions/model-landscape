"""Residual plot: y_pred vs (y_pred - y_true) with optional train overlay."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn
from mlflow_lens.regressor._utils import predict_values


def _residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y_true


def _build(
    y_true: Any,
    y_pred: Any,
    *,
    y_true_train: Any = None,
    y_pred_train: Any = None,
    **layout: Any,
) -> tuple:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res = _residuals(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    fig = go.Figure()
    if y_true_train is not None and y_pred_train is not None:
        y_true_train = np.asarray(y_true_train, dtype=float)
        y_pred_train = np.asarray(y_pred_train, dtype=float)
        res_train = _residuals(y_true_train, y_pred_train)
        fig.add_trace(
            go.Scatter(
                x=y_pred_train,
                y=res_train,
                mode="markers",
                name=f"train (R²={r2_score(y_true_train, y_pred_train):.3f})",
                marker=dict(size=5, opacity=0.4),
                hovertemplate="y_pred=%{x:.3f}<br>residual=%{y:.3f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=res,
            mode="markers",
            name=f"test (R²={r2:.3f})",
            marker=dict(size=6, opacity=0.7),
            hovertemplate="y_pred=%{x:.3f}<br>residual=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line=dict(dash="dash"))
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Residuals"),
            xaxis_title="Predicted",
            yaxis_title="Residual (y_pred − y_true)",
            **layout,
        )
    )

    data = {
        "test": [{"y_pred": float(p), "residual": float(r)} for p, r in zip(y_pred, res)],
        "r2": r2,
    }
    if y_true_train is not None:
        data["train"] = [
            {"y_pred": float(p), "residual": float(r)}
            for p, r in zip(
                y_pred_train, _residuals(np.asarray(y_true_train, dtype=float), y_pred_train)
            )
        ]
    return fig, data


@quickfn(panel_type="residuals")
def residuals(
    model: Any,
    X: Any,
    y: Any,
    *,
    X_train: Any = None,
    y_train: Any = None,
    **layout: Any,
) -> tuple:
    """Residual plot for a fitted regressor.

    Args:
        model: Fitted regressor.
        X: Test feature matrix.
        y: Test targets.
        X_train: Optional training feature matrix; combined with ``y_train``,
            plots train residuals as a lighter overlay.
        y_train: Optional training targets (see ``X_train``).
    """
    y_pred = predict_values(model, X)
    y_pred_train = predict_values(model, X_train) if X_train is not None else None
    return _build(y, y_pred, y_true_train=y_train, y_pred_train=y_pred_train, **layout)


@quickfn(panel_type="residuals")
def _from_predictions(
    y_true: Any,
    y_pred: Any,
    *,
    y_true_train: Any = None,
    y_pred_train: Any = None,
    **layout: Any,
) -> tuple:
    """Residual plot from pre-computed predictions."""
    return _build(y_true, y_pred, y_true_train=y_true_train, y_pred_train=y_pred_train, **layout)


residuals.from_predictions = _from_predictions
