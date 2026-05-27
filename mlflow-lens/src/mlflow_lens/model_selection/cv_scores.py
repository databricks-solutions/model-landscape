"""Cross-validation fold scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn


def _build(scores: np.ndarray, **layout: Any) -> tuple:
    folds = list(range(1, scores.size + 1))
    mean = float(np.mean(scores))
    std = float(np.std(scores))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=folds,
            y=scores.tolist(),
            name="fold score",
            hovertemplate="fold %{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=mean,
        line=dict(dash="dash"),
        annotation_text=f"mean = {mean:.3f} ± {std:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "CV Scores"),
            xaxis_title="Fold",
            yaxis_title="Score",
            **layout,
        )
    )

    rows = [{"fold": f, "score": float(s)} for f, s in zip(folds, scores)]
    data = {"folds": rows, "mean": mean, "std": std}
    return fig, data


@quickfn(panel_type="cv_scores")
def cv_scores(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    cv: int = 5,
    scoring: str | None = None,
    n_jobs: int | None = None,
    **layout: Any,
) -> tuple:
    """Bar chart of cross-validation fold scores with a mean reference line."""
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return _build(np.asarray(scores), **layout)


@quickfn(panel_type="cv_scores")
def _from_scores(scores: Any, **layout: Any) -> tuple:
    """CV-score bar chart from a pre-computed array of fold scores."""
    return _build(np.asarray(scores, dtype=float), **layout)


cv_scores.from_scores = _from_scores
