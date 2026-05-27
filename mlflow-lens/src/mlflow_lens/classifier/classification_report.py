"""Classification report — precision/recall/F1/support per class."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report as sk_classification_report

from mlflow_lens._plotly_theme import lens_layout
from mlflow_lens._quickfn import quickfn
from mlflow_lens.classifier._utils import class_labels


_METRICS = ("precision", "recall", "f1-score")


def _build(y_true: Any, y_pred: Any, classes: list[Any], **layout: Any) -> tuple:
    label_strs = [str(c) for c in classes]
    report = sk_classification_report(
        y_true, y_pred, labels=classes, target_names=label_strs, output_dict=True, zero_division=0
    )

    rows = []
    z = []
    for cls in label_strs:
        entry = report[cls]
        rows.append(
            {
                "class": cls,
                "precision": entry["precision"],
                "recall": entry["recall"],
                "f1_score": entry["f1-score"],
                "support": int(entry["support"]),
            }
        )
        z.append([entry[m] for m in _METRICS])

    z = np.array(z)
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(_METRICS),
            y=label_strs,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="score"),
            hovertemplate="class=%{y}<br>metric=%{x}<br>score=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "Classification Report"),
            xaxis_title="Metric",
            yaxis_title="Class",
            **layout,
        )
    )

    data = rows
    return fig, data


@quickfn(panel_type="classification_report")
def classification_report(
    model: Any,
    X: Any,
    y: Any,
    **layout: Any,
) -> tuple:
    """Per-class precision/recall/F1 heatmap."""
    classes = class_labels(model, y)
    y_pred = model.predict(X)
    return _build(y, y_pred, classes, **layout)


@quickfn(panel_type="classification_report")
def _from_predictions(
    y_true: Any,
    y_pred: Any,
    *,
    classes: list[Any] | None = None,
    **layout: Any,
) -> tuple:
    """Classification report from pre-computed predictions."""
    if classes is None:
        classes = sorted(
            set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())
        )
    return _build(y_true, y_pred, classes, **layout)


classification_report.from_predictions = _from_predictions
