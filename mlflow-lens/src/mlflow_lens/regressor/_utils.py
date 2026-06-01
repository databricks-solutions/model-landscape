"""Shared helpers for regressor panels."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlflow_lens._models import booster_predict_raw, is_booster


def predict_values(model: Any, X: Any) -> np.ndarray:
    """Return numeric predictions from a fitted regressor.

    Handles sklearn-compatible ``model.predict(X)`` plus low-level
    ``xgboost.Booster`` (wraps ``X`` in :class:`xgboost.DMatrix`
    automatically) and ``lightgbm.Booster`` (passes ``X`` directly).
    """
    if is_booster(model):
        return booster_predict_raw(model, X)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    raise AttributeError(
        f"{type(model).__name__} has no predict() method. "
        "Compute predictions yourself and call <panel>.from_predictions(y_true, y_pred)."
    )
