"""Discrimination threshold — precision/recall/F1/queue-rate over thresholds (binary only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn
from mlflow_lens.classifier._utils import class_labels, is_binary, predict_scores


def _sweep(
    y_true: np.ndarray, scores_pos: np.ndarray, pos_label: Any, n_thresholds: int
) -> list[dict]:
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    rows = []
    y_pos = (y_true == pos_label).astype(int)
    for t in thresholds:
        y_pred = (scores_pos >= t).astype(int)
        tp = int(((y_pred == 1) & (y_pos == 1)).sum())
        fp = int(((y_pred == 1) & (y_pos == 0)).sum())
        fn = int(((y_pred == 0) & (y_pos == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        queue_rate = float(y_pred.mean())
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "queue_rate": queue_rate,
            }
        )
    return rows


def _build(
    y_true: Any,
    scores_pos: np.ndarray,
    pos_label: Any,
    *,
    n_thresholds: int,
    **layout: Any,
) -> tuple:
    y_true = np.asarray(y_true)
    rows = _sweep(y_true, np.asarray(scores_pos), pos_label, n_thresholds)
    thresholds = [r["threshold"] for r in rows]
    f1_values = [r["f1"] for r in rows]
    best_idx = int(np.argmax(f1_values))
    best_threshold = thresholds[best_idx]

    fig = go.Figure()
    for metric in ("precision", "recall", "f1", "queue_rate"):
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=[r[metric] for r in rows],
                mode="lines",
                name=metric,
                hovertemplate=f"threshold=%{{x:.2f}}<br>{metric}=%{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_vline(
        x=best_threshold,
        line=dict(dash="dash"),
        annotation_text=f"best F1 @ {best_threshold:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Discrimination Threshold"),
            xaxis_title="Decision Threshold",
            yaxis_title="Score",
            **layout,
        )
    )

    data = {
        "thresholds": rows,
        "best_threshold": best_threshold,
        "best_f1": f1_values[best_idx],
    }
    return fig, data


@quickfn(panel_type="discrimination_threshold")
def discrimination_threshold(
    model: Any,
    X: Any,
    y: Any,
    *,
    n_thresholds: int = 50,
    **layout: Any,
) -> tuple:
    """Sweep decision thresholds and plot precision/recall/F1/queue-rate.

    Binary classifiers only.
    """
    classes = class_labels(model, y)
    if not is_binary(classes):
        raise ValueError(
            "discrimination_threshold supports binary classifiers only "
            f"(got {len(classes)} classes)"
        )
    scores = predict_scores(model, X)
    scores_pos = scores[:, 1]
    return _build(y, scores_pos, classes[1], n_thresholds=n_thresholds, **layout)


@quickfn(panel_type="discrimination_threshold")
def _from_scores(
    y_true: Any,
    y_score: Any,
    *,
    pos_label: Any = 1,
    n_thresholds: int = 50,
    **layout: Any,
) -> tuple:
    """Discrimination threshold sweep from pre-computed positive-class scores."""
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    return _build(y_true, y_score, pos_label, n_thresholds=n_thresholds, **layout)


discrimination_threshold.from_scores = _from_scores
