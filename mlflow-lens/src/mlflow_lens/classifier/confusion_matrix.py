"""Confusion matrix heatmap."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn
from mlflow_lens.classifier._utils import class_labels, predict_labels


def _build(
    y_true: Any,
    y_pred: Any,
    classes: list[Any],
    *,
    normalize: str | None = None,
    **layout: Any,
) -> tuple:
    cm = sk_confusion_matrix(y_true, y_pred, labels=classes, normalize=normalize)
    text_fmt = ".2f" if normalize else "d"
    text = [[format(v, text_fmt) for v in row] for row in cm]
    label_strs = [str(c) for c in classes]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=label_strs,
            y=label_strs,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar=dict(title="count" if not normalize else "rate"),
            hovertemplate=(
                "true=%{y}<br>predicted=%{x}<br>"
                + ("rate" if normalize else "count")
                + "=%{z}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Confusion Matrix"),
            xaxis_title="Predicted",
            yaxis_title="True",
            **layout,
        )
    )
    fig.update_yaxes(autorange="reversed")

    data = {"matrix": cm.tolist(), "normalize": normalize}
    return fig, data


@quickfn(panel_type="confusion_matrix")
def confusion_matrix(
    model: Any,
    X: Any,
    y: Any,
    *,
    normalize: str | None = None,
    labels: list[Any] | None = None,
    **layout: Any,
) -> tuple:
    """Confusion matrix heatmap for a fitted classifier.

    Args:
        model: Fitted classifier.
        X: Feature matrix.
        y: True labels.
        normalize: Pass through to :func:`sklearn.metrics.confusion_matrix`
            (``None``, ``"true"``, ``"pred"``, or ``"all"``).
        labels: Explicit class label ordering (defaults to ``model.classes_``).
    """
    classes = labels if labels is not None else class_labels(model, y)
    y_pred = predict_labels(model, X, classes)
    return _build(y, y_pred, classes, normalize=normalize, **layout)


@quickfn(panel_type="confusion_matrix")
def _from_predictions(
    y_true: Any,
    y_pred: Any,
    *,
    labels: list[Any] | None = None,
    normalize: str | None = None,
    **layout: Any,
) -> tuple:
    """Confusion matrix from pre-computed predictions."""
    classes = (
        labels
        if labels is not None
        else sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    )
    return _build(y_true, y_pred, classes, normalize=normalize, **layout)


confusion_matrix.from_predictions = _from_predictions
