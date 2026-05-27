"""Alpha selection for regularized linear regressors (Ridge/Lasso/ElasticNet)."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn


def _build(alphas: np.ndarray, mean_scores: np.ndarray, std_scores: np.ndarray, **layout: Any) -> tuple:
    best_idx = int(np.argmax(mean_scores))
    best_alpha = float(alphas[best_idx])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=alphas,
            y=mean_scores,
            mode="lines+markers",
            name="CV score",
            error_y=dict(type="data", array=std_scores, visible=True),
            hovertemplate="alpha=%{x:.4g}<br>score=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=best_alpha,
        line=dict(dash="dash"),
        annotation_text=f"best α = {best_alpha:.3g}",
        annotation_position="top right",
    )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Alpha Selection"),
            xaxis_title="α",
            yaxis_title="CV score",
            xaxis=dict(type="log"),
            **layout,
        )
    )

    rows = [
        {"alpha": float(a), "score_mean": float(m), "score_std": float(s)}
        for a, m, s in zip(alphas, mean_scores, std_scores)
    ]
    data = {"sweeps": rows, "best_alpha": best_alpha, "best_score": float(mean_scores[best_idx])}
    return fig, data


@quickfn(panel_type="alpha_selection")
def alpha_selection(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    alphas: Any = None,
    cv: int = 5,
    scoring: str | None = None,
    **layout: Any,
) -> tuple:
    """Sweep ``alpha`` for a regularized linear regressor and plot CV scores.

    Args:
        estimator: An sklearn estimator with an ``alpha`` hyperparameter
            (e.g. ``Ridge``, ``Lasso``, ``ElasticNet``).
        X: Feature matrix.
        y: Targets.
        alphas: Iterable of alpha values to try; defaults to ``np.logspace(-4, 4, 25)``.
        cv: Number of CV folds.
        scoring: Optional sklearn scoring string (default: ``r2`` for regressors).
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 25)
    alphas = np.asarray(alphas, dtype=float)

    mean_scores = np.zeros(alphas.size)
    std_scores = np.zeros(alphas.size)
    for i, a in enumerate(alphas):
        est = clone(estimator).set_params(alpha=a)
        scores = cross_val_score(est, X, y, cv=cv, scoring=scoring)
        mean_scores[i] = float(np.mean(scores))
        std_scores[i] = float(np.std(scores))

    return _build(alphas, mean_scores, std_scores, **layout)
