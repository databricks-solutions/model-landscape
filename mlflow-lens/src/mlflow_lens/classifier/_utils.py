"""Shared helpers for classifier panels."""

from __future__ import annotations

from typing import Any

import numpy as np


def predict_scores(model: Any, X: Any) -> np.ndarray:
    """Return per-class scores from a fitted classifier.

    Prefers :meth:`predict_proba`; falls back to :meth:`decision_function`.
    For binary problems with a 1-D decision function, returns shape ``(n, 2)``
    where column 0 is ``-score`` and column 1 is ``score`` so callers can
    treat binary and multi-class uniformly.
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        return proba
    if hasattr(model, "decision_function"):
        dec = np.asarray(model.decision_function(X))
        if dec.ndim == 1:
            return np.column_stack([-dec, dec])
        return dec
    raise AttributeError(
        f"{type(model).__name__} has neither predict_proba nor decision_function; "
        "pass pre-computed scores via .from_scores(...)"
    )


def class_labels(model: Any, y: Any) -> list[Any]:
    """Get the class label list from the model or fall back to y."""
    if hasattr(model, "classes_"):
        return list(model.classes_)
    return sorted(set(np.asarray(y).tolist()))


def is_binary(classes: list[Any]) -> bool:
    return len(classes) == 2
