"""Learning curve: model score vs. training set size."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import learning_curve as sk_learning_curve

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn


def _band(fig: go.Figure, x: np.ndarray, mean: np.ndarray, std: np.ndarray, name: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            name=name,
            hovertemplate=f"{name}: %{{y:.3f}} @ size=%{{x}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            opacity=0.15,
            name=f"{name} ± σ",
            showlegend=False,
            hoverinfo="skip",
        )
    )


@quickfn(panel_type="learning_curve")
def learning_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    cv: int = 5,
    train_sizes: Any = None,
    scoring: str | None = None,
    n_jobs: int | None = None,
    **layout: Any,
) -> tuple:
    """Train/validation score vs. training-set size, with ±σ bands."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    sizes, train_scores, val_scores = sk_learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig = go.Figure()
    _band(fig, sizes, train_mean, train_std, "train")
    _band(fig, sizes, val_mean, val_std, "validation")
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Learning Curve"),
            xaxis_title="Training set size",
            yaxis_title="Score",
            **layout,
        )
    )

    rows = [
        {
            "train_size": int(s),
            "train_score_mean": float(tm),
            "train_score_std": float(ts),
            "val_score_mean": float(vm),
            "val_score_std": float(vs),
        }
        for s, tm, ts, vm, vs in zip(sizes, train_mean, train_std, val_mean, val_std)
    ]
    data = rows
    return fig, data
