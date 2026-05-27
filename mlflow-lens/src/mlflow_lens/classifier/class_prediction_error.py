"""Class prediction error — stacked bar of predicted classes per true class."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn
from mlflow_lens.classifier._utils import class_labels


def _build(y_true: Any, y_pred: Any, classes: list[Any], **layout: Any) -> tuple:
    cm = sk_confusion_matrix(y_true, y_pred, labels=classes)
    label_strs = [str(c) for c in classes]

    fig = go.Figure()
    for i, predicted_cls in enumerate(label_strs):
        fig.add_trace(
            go.Bar(
                x=label_strs,
                y=cm[:, i].tolist(),
                name=f"predicted = {predicted_cls}",
                hovertemplate="true=%{x}<br>count=%{y}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        **lens_layout(
            title=layout.pop("title", "Class Prediction Error"),
            xaxis_title="Actual Class",
            yaxis_title="Number of Predicted Class",
            **layout,
        ),
    )

    rows = []
    for i, true_cls in enumerate(label_strs):
        for j, pred_cls in enumerate(label_strs):
            rows.append(
                {"true": true_cls, "predicted": pred_cls, "count": int(cm[i, j])}
            )
    data = rows
    return fig, data


@quickfn(panel_type="class_prediction_error")
def class_prediction_error(
    model: Any,
    X: Any,
    y: Any,
    **layout: Any,
) -> tuple:
    """Stacked bar showing the predicted-class breakdown per true class."""
    classes = class_labels(model, y)
    y_pred = model.predict(X)
    return _build(y, y_pred, classes, **layout)


@quickfn(panel_type="class_prediction_error")
def _from_predictions(
    y_true: Any,
    y_pred: Any,
    *,
    classes: list[Any] | None = None,
    **layout: Any,
) -> tuple:
    """Class prediction error from pre-computed predictions."""
    if classes is None:
        classes = sorted(
            set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())
        )
    return _build(y_true, y_pred, classes, **layout)


class_prediction_error.from_predictions = _from_predictions
