"""Feature importances bar chart.

Reads :attr:`feature_importances_` (tree-based) or :attr:`coef_` (linear models).
For SHAP-derived importance, use :mod:`mlflow_lens.feature_importance` instead
(it produces the same panel type).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn


def _extract(model: Any) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        # For multi-output linear models, average |coef| over outputs.
        if coef.ndim > 1:
            return np.abs(coef).mean(axis=0)
        return np.abs(coef)
    raise AttributeError(
        f"{type(model).__name__} has neither feature_importances_ nor coef_; "
        "pass importances directly via .from_values(...)"
    )


def _build(
    names: list[str],
    values: np.ndarray,
    *,
    top_n: int | None,
    **layout: Any,
) -> tuple:
    order = np.argsort(-values)
    names_sorted = [names[i] for i in order]
    values_sorted = values[order]
    if top_n is not None:
        names_sorted = names_sorted[:top_n]
        values_sorted = values_sorted[:top_n]

    fig = go.Figure(
        data=go.Bar(
            x=values_sorted.tolist(),
            y=names_sorted,
            orientation="h",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Feature Importances"),
            xaxis_title="Importance",
            yaxis_title="Feature",
            yaxis=dict(autorange="reversed"),
            **layout,
        )
    )

    rows = [
        {"feature": n, "importance": float(v)}
        for n, v in zip(names_sorted, values_sorted)
    ]
    data = rows
    return fig, data


@quickfn(panel_type="feature_importance")
def feature_importances(
    model: Any,
    feature_names: list[str],
    *,
    top_n: int | None = None,
    **layout: Any,
) -> tuple:
    """Horizontal bar of feature importances from a fitted model.

    Reads ``model.feature_importances_`` (tree-based) or ``|model.coef_|``
    (linear models). For SHAP-based importance, use
    :func:`mlflow_lens.feature_importance.shap_importance`.
    """
    values = _extract(model)
    if len(values) != len(feature_names):
        raise ValueError(
            f"got {len(values)} importances but {len(feature_names)} names"
        )
    return _build(list(feature_names), values, top_n=top_n, **layout)


@quickfn(panel_type="feature_importance")
def _from_values(
    feature_names: list[str],
    importances: Any,
    *,
    top_n: int | None = None,
    **layout: Any,
) -> tuple:
    """Feature importance bar chart from a pre-computed value array."""
    values = np.asarray(importances, dtype=float)
    return _build(list(feature_names), values, top_n=top_n, **layout)


feature_importances.from_values = _from_values
