"""ROC curve + AUC for binary and multi-class classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from mlflow_lens._plotly_theme import LENS_DIAGONAL_COLOR, lens_layout
from mlflow_lens._quickfn import quickfn
from mlflow_lens.classifier._utils import class_labels, is_binary, predict_scores


def _build(y_true: Any, y_score: np.ndarray, classes: list[Any], **layout: Any) -> tuple:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    curves: list[dict] = []
    if is_binary(classes):
        pos_idx = 1 if y_score.ndim == 2 else 0
        scores_pos = y_score[:, pos_idx] if y_score.ndim == 2 else y_score
        fpr, tpr, _ = roc_curve(y_true, scores_pos, pos_label=classes[1])
        curves.append(
            {
                "label": str(classes[1]),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc(fpr, tpr)),
            }
        )
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            curves.append(
                {
                    "label": str(cls),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc(fpr, tpr)),
                }
            )
        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        curves.append(
            {
                "label": "micro-average",
                "fpr": fpr_micro.tolist(),
                "tpr": tpr_micro.tolist(),
                "auc": float(auc(fpr_micro, tpr_micro)),
            }
        )

    fig = go.Figure()
    for c in curves:
        fig.add_trace(
            go.Scatter(
                x=c["fpr"],
                y=c["tpr"],
                mode="lines",
                name=f"{c['label']} (AUC={c['auc']:.3f})",
                hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=LENS_DIAGONAL_COLOR, dash="dash"),
            name="chance",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        **lens_layout(
            title=layout.pop("title", "ROC Curve"),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            **layout,
        )
    )

    data = {
        "classes": curves,
        "n_classes": len(classes),
        "macro_auc": float(np.mean([c["auc"] for c in curves if c["label"] != "micro-average"])),
    }
    return fig, data


@quickfn(panel_type="roc_curve")
def roc_auc(
    model: Any,
    X: Any,
    y: Any,
    **layout: Any,
) -> tuple:
    """Plot ROC curve(s) and AUC for a fitted classifier.

    Args:
        model: A fitted classifier with ``predict_proba`` or ``decision_function``.
        X: Feature matrix used for scoring.
        y: True labels.
        **layout: Forwarded to :func:`lens_layout` (e.g. ``title``).
    """
    classes = class_labels(model, y)
    scores = predict_scores(model, X)
    return _build(y, scores, classes, **layout)


@quickfn(panel_type="roc_curve")
def _from_scores(
    y_true: Any,
    y_score: Any,
    *,
    classes: list[Any] | None = None,
    **layout: Any,
) -> tuple:
    """ROC from pre-computed scores. Pass per-class probabilities/scores."""
    y_score = np.asarray(y_score)
    if classes is None:
        if y_score.ndim == 1 or y_score.shape[1] == 1:
            classes = sorted(set(np.asarray(y_true).tolist()))
        else:
            classes = list(range(y_score.shape[1]))
    return _build(y_true, y_score, classes, **layout)


roc_auc.from_scores = _from_scores
