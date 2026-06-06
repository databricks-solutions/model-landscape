"""SHAP-based global feature importance panel.

Computes global feature importance as the mean absolute SHAP value per feature.
The explainer is auto-selected from the model type:

* tree-based models (scikit-learn ensembles, XGBoost, LightGBM, CatBoost) use
  :class:`shap.TreeExplainer` — fast and exact;
* everything else falls back to :class:`shap.KernelExplainer`, which is
  model-agnostic but slow, so it runs against a summarized background sample
  and a bounded number of explained rows.

Produces the same ``feature_importance`` panel type as
:func:`mlflow_lens.model_selection.feature_importances`, so the two are
interchangeable in the Lens UI.

Requires the ``[shap]`` extra::

    pip install "mlflow-lens[shap]"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlflow_lens._quickfn import quickfn
from mlflow_lens.model_selection.feature_importances import _build

# Class-name fragments that identify tree ensembles across frameworks. Module
# names are unreliable (sklearn trees live under ``sklearn.ensemble``/``.tree``),
# so we match on the estimator class name instead.
_TREE_CLASS_HINTS = (
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "DecisionTree",
    "XGB",
    "LGBM",
    "CatBoost",
    "Booster",  # native xgboost / lightgbm boosters
)


def _is_tree_model(model: Any) -> bool:
    """Best-effort detection of tree-based estimators for explainer selection."""
    if type(model).__module__.split(".")[0] in {"xgboost", "lightgbm", "catboost"}:
        return True
    name = type(model).__name__
    return any(hint in name for hint in _TREE_CLASS_HINTS)


def _resolve_names(X: Any, feature_names: list[str] | None) -> list[str]:
    if feature_names is not None:
        return list(feature_names)
    if hasattr(X, "columns"):
        return list(X.columns)
    raise ValueError(
        "feature_names is required when X is not a pandas DataFrame"
    )


def _head(X: Any, n: int) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[:n]
    return np.asarray(X)[:n]


def _mean_abs_shap(shap_values: Any) -> np.ndarray:
    """Reduce SHAP output of any shape to one mean-|value| per feature.

    Handles the three shapes SHAP returns:

    * a list of ``(n_samples, n_features)`` arrays (one per class),
    * a single ``(n_samples, n_features)`` array,
    * a ``(n_samples, n_features, n_classes)`` array (newer multi-class trees).
    """
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(np.asarray(v)) for v in shap_values], axis=0)
        return stacked.mean(axis=(0, 1))
    arr = np.abs(np.asarray(shap_values, dtype=float))
    if arr.ndim == 3:
        return arr.mean(axis=(0, 2))
    return arr.mean(axis=0)


@quickfn(panel_type="feature_importance")
def shap_importance(
    model: Any,
    X: Any,
    *,
    feature_names: list[str] | None = None,
    explainer: str | None = None,
    max_samples: int = 100,
    background_samples: int = 100,
    nsamples: int | str = "auto",
    top_n: int | None = None,
    **layout: Any,
) -> tuple:
    """Global feature importance from SHAP values for a fitted model.

    Args:
        model: Fitted estimator.
        X: Feature matrix to explain. A pandas DataFrame supplies feature names
            automatically; otherwise pass ``feature_names``.
        feature_names: Column names (required when ``X`` is not a DataFrame).
        explainer: Force ``"tree"`` or ``"kernel"``. Defaults to auto-detection
            (tree models -> :class:`shap.TreeExplainer`, else
            :class:`shap.KernelExplainer`).
        max_samples: Upper bound on rows explained. KernelExplainer cost scales
            with this, so it is capped rather than explaining all of ``X``.
        background_samples: Rows sampled as the KernelExplainer background
            (ignored for the tree explainer).
        nsamples: KernelExplainer coalition budget per row (its "number of
            trials"); ``"auto"`` uses ``2 * n_features + 2048``.
        top_n: Keep only the top-N features by importance.
        **layout: Forwarded to :func:`lens_layout` (e.g. ``title``).
    """
    import shap

    names = _resolve_names(X, feature_names)
    X_explain = _head(X, max_samples)

    kind = explainer or ("tree" if _is_tree_model(model) else "kernel")
    if kind == "tree":
        expl = shap.TreeExplainer(model)
        shap_values = expl.shap_values(X_explain)
    elif kind == "kernel":
        predict = getattr(model, "predict_proba", None) or model.predict
        n_bg = min(background_samples, _len(X))
        background = shap.sample(X, n_bg, random_state=0)
        expl = shap.KernelExplainer(predict, background)
        shap_values = expl.shap_values(X_explain, nsamples=nsamples, silent=True)
    else:
        raise ValueError(
            f"explainer must be 'tree', 'kernel', or None; got {explainer!r}"
        )

    importances = _mean_abs_shap(shap_values)
    if len(importances) != len(names):
        raise ValueError(
            f"got {len(importances)} SHAP importances but {len(names)} feature names"
        )
    return _build(names, importances, top_n=top_n, **layout)


def _len(X: Any) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)


@quickfn(panel_type="feature_importance")
def _from_shap_values(
    feature_names: list[str],
    shap_values: Any,
    *,
    top_n: int | None = None,
    **layout: Any,
) -> tuple:
    """Feature importance from pre-computed SHAP values.

    Use this for explainers not covered by auto-detection (e.g.
    :class:`shap.DeepExplainer` / :class:`shap.GradientExplainer` for PyTorch).
    ``shap_values`` may be a single array, a per-class list, or a 3-D
    ``(samples, features, classes)`` array; it is reduced to mean-|value| per
    feature.
    """
    importances = _mean_abs_shap(shap_values)
    return _build(list(feature_names), importances, top_n=top_n, **layout)


shap_importance.from_shap_values = _from_shap_values
