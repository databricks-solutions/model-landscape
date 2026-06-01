"""Cross-framework model introspection.

The mlflow-lens panels work primarily against sklearn-compatible estimators
(``predict_proba`` / ``decision_function`` / ``predict``), but the two big
gradient-boosting libraries — XGBoost and LightGBM — also expose a
*low-level* ``Booster`` API (``xgb.train(...)`` / ``lgb.train(...)``) that
doesn't conform to that contract. This module detects those Booster
instances and translates their predictions into the shape the panels
expect, so a Booster can be passed straight to ``roc_auc(booster, X, y)``
just like an ``XGBClassifier``.

For any model outside this list, callers should drop down to the
``.from_scores`` / ``.from_predictions`` / ``.from_values`` entry points
on the panel quick functions and pass pre-computed outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _xgboost_booster_module():
    """Return the xgboost module if available, else None."""
    try:
        import xgboost

        return xgboost
    except ImportError:
        return None


def _lightgbm_booster_module():
    try:
        import lightgbm

        return lightgbm
    except ImportError:
        return None


def is_xgboost_booster(model: Any) -> bool:
    xgb = _xgboost_booster_module()
    return xgb is not None and isinstance(model, xgb.Booster)


def is_lightgbm_booster(model: Any) -> bool:
    lgb = _lightgbm_booster_module()
    return lgb is not None and isinstance(model, lgb.Booster)


def is_booster(model: Any) -> bool:
    """True if `model` is a low-level Booster from xgboost or lightgbm."""
    return is_xgboost_booster(model) or is_lightgbm_booster(model)


def booster_predict_raw(model: Any, X: Any) -> np.ndarray:
    """Call ``predict`` on a Booster, wrapping ``X`` in a DMatrix for XGBoost.

    Returns the raw output of ``booster.predict`` — shape ``(n,)`` for binary
    classification with the standard objectives (``binary:logistic``,
    LightGBM ``binary``), or ``(n, k)`` for multi-class
    (``multi:softprob`` / LightGBM ``multiclass``).
    """
    if is_xgboost_booster(model):
        xgb = _xgboost_booster_module()
        return np.asarray(model.predict(xgb.DMatrix(X)))
    if is_lightgbm_booster(model):
        return np.asarray(model.predict(X))
    raise TypeError(f"not a supported Booster: {type(model).__name__}")


def booster_class_scores(model: Any, X: Any) -> np.ndarray:
    """Booster output as per-class scores, shape ``(n, k)``.

    Binary output of shape ``(n,)`` is expanded to ``(n, 2)`` with
    column 0 = ``1 - p`` and column 1 = ``p`` so downstream code can
    treat binary and multi-class uniformly.
    """
    raw = booster_predict_raw(model, X)
    if raw.ndim == 1:
        return np.column_stack([1.0 - raw, raw])
    return raw
