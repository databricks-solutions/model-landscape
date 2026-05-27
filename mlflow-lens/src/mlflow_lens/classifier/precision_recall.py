"""Precision-recall curve."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn
from mlflow_lens.classifier._utils import class_labels, is_binary, predict_scores


def _build(y_true: Any, y_score: np.ndarray, classes: list[Any], **layout: Any) -> tuple:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    curves: list[dict] = []
    if is_binary(classes):
        pos_idx = 1 if y_score.ndim == 2 else 0
        scores_pos = y_score[:, pos_idx] if y_score.ndim == 2 else y_score
        precision, recall, _ = precision_recall_curve(
            y_true, scores_pos, pos_label=classes[1]
        )
        ap = float(average_precision_score((y_true == classes[1]).astype(int), scores_pos))
        curves.append(
            {
                "label": str(classes[1]),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "average_precision": ap,
            }
        )
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ap = float(average_precision_score(y_bin[:, i], y_score[:, i]))
            curves.append(
                {
                    "label": str(cls),
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "average_precision": ap,
                }
            )

    fig = go.Figure()
    for c in curves:
        fig.add_trace(
            go.Scatter(
                x=c["recall"],
                y=c["precision"],
                mode="lines",
                name=f"{c['label']} (AP={c['average_precision']:.3f})",
                hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Precision-Recall Curve"),
            xaxis_title="Recall",
            yaxis_title="Precision",
            **layout,
        )
    )

    data = {"classes": curves, "n_classes": len(classes)}
    return fig, data


@quickfn(panel_type="precision_recall_curve")
def precision_recall(
    model: Any,
    X: Any,
    y: Any,
    **layout: Any,
) -> tuple:
    """Plot precision-recall curve(s) for a fitted classifier."""
    classes = class_labels(model, y)
    scores = predict_scores(model, X)
    return _build(y, scores, classes, **layout)


@quickfn(panel_type="precision_recall_curve")
def _from_scores(
    y_true: Any,
    y_score: Any,
    *,
    classes: list[Any] | None = None,
    **layout: Any,
) -> tuple:
    """Precision-recall from pre-computed scores."""
    y_score = np.asarray(y_score)
    if classes is None:
        if y_score.ndim == 1 or y_score.shape[1] == 1:
            classes = sorted(set(np.asarray(y_true).tolist()))
        else:
            classes = list(range(y_score.shape[1]))
    return _build(y_true, y_score, classes, **layout)


precision_recall.from_scores = _from_scores
