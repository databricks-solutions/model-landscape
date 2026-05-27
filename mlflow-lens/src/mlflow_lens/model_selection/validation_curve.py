"""Validation curve: model score vs. a single hyperparameter."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import validation_curve as sk_validation_curve

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn


def _band(fig: go.Figure, x: np.ndarray, mean: np.ndarray, std: np.ndarray, name: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            name=name,
            hovertemplate=f"{name}: %{{y:.3f}} @ %{{x}}<extra></extra>",
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
            showlegend=False,
            hoverinfo="skip",
        )
    )


@quickfn(panel_type="validation_curve")
def validation_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    param_name: str,
    param_range: Any,
    cv: int = 5,
    scoring: str | None = None,
    n_jobs: int | None = None,
    log_x: bool = False,
    **layout: Any,
) -> tuple:
    """Train/validation score across values of a single hyperparameter."""
    param_range = np.asarray(param_range)
    train_scores, val_scores = sk_validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig = go.Figure()
    _band(fig, param_range, train_mean, train_std, "train")
    _band(fig, param_range, val_mean, val_std, "validation")
    if log_x:
        layout.setdefault("xaxis", {})["type"] = "log"
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", f"Validation Curve ({param_name})"),
            xaxis_title=param_name,
            yaxis_title="Score",
            **layout,
        )
    )

    rows = [
        {
            "param_value": (
                float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
            ),
            "train_score_mean": float(tm),
            "train_score_std": float(ts),
            "val_score_mean": float(vm),
            "val_score_std": float(vs),
        }
        for v, tm, ts, vm, vs in zip(param_range, train_mean, train_std, val_mean, val_std)
    ]
    data = {"param_name": param_name, "values": rows}
    return fig, data
