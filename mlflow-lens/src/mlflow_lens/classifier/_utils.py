"""Shared helpers for classifier panels."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlflow_lens._models import booster_class_scores, is_booster


def predict_scores(model: Any, X: Any) -> np.ndarray:
    """Return per-class scores from a fitted classifier.

    Accepted shapes (in priority order):

    1. sklearn-style ``predict_proba`` → returns its output directly.
    2. sklearn-style ``decision_function`` → for binary problems with 1-D
       output, returns ``(n, 2)`` with columns ``[-score, score]`` so
       binary and multi-class callers can share code.
    3. ``xgboost.Booster`` (from ``xgb.train``) → wraps ``X`` in
       :class:`xgboost.DMatrix`, calls ``predict``, reshapes binary
       output to ``(n, 2)``.
    4. ``lightgbm.Booster`` (from ``lgb.train``) → calls ``predict``,
       reshapes binary output to ``(n, 2)``.

    For anything else, callers should pre-compute scores and use the
    panel's ``.from_scores`` entry point.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        dec = np.asarray(model.decision_function(X))
        if dec.ndim == 1:
            return np.column_stack([-dec, dec])
        return dec
    if is_booster(model):
        return booster_class_scores(model, X)
    raise AttributeError(
        f"{type(model).__name__} has no predict_proba, decision_function, "
        "or recognized Booster API. "
        "Compute scores yourself and call <panel>.from_scores(y_true, y_score)."
    )


def predict_labels(model: Any, X: Any, classes: list[Any]) -> np.ndarray:
    """Return predicted class labels from a fitted classifier.

    For sklearn-compatible models calls ``model.predict(X)`` directly. For
    Booster instances, derives labels from :func:`predict_scores` — either
    a 0.5 threshold for binary or ``argmax`` for multi-class — and maps the
    resulting indices back through ``classes``.
    """
    if is_booster(model):
        scores = booster_class_scores(model, X)
        if len(classes) == 2:
            idx = (scores[:, 1] >= 0.5).astype(int)
        else:
            idx = np.argmax(scores, axis=1)
        return np.asarray([classes[i] for i in idx])
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    raise AttributeError(
        f"{type(model).__name__} has no predict() method. "
        "Compute predictions yourself and call <panel>.from_predictions(y_true, y_pred)."
    )


def class_labels(model: Any, y: Any) -> list[Any]:
    """Get the class label list from the model or fall back to y.

    Booster instances don't carry classes_ (they're low-level and operate
    on numeric arrays), so we derive from y in that case.
    """
    if not is_booster(model) and hasattr(model, "classes_"):
        return list(model.classes_)
    return sorted(set(np.asarray(y).tolist()))


def is_binary(classes: list[Any]) -> bool:
    return len(classes) == 2
