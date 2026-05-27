"""Declarative visualization panels stored as MLflow artifacts.

Panels store structured data (not rendered HTML) so the Lens UI can render
and compare them across runs. Each panel type has a known schema; the UI
knows how to interpret it.
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd

from mlflow_lens._artifacts import log_json_artifact
from mlflow_lens._version import __version__

PANEL_TYPES = frozenset(
    {
        # Original 4 (Dash app contract — JSON shape must stay stable):
        "confusion_matrix",
        "roc_curve",
        "feature_importance",
        "learning_curve",
        # Classification:
        "classification_report",
        "precision_recall_curve",
        "class_prediction_error",
        "discrimination_threshold",
        # Regression:
        "prediction_error",
        "residuals",
        "alpha_selection",
        # Model selection:
        "validation_curve",
        "cv_scores",
    }
)


def log_panel(
    panel_type: str,
    data: dict | pd.DataFrame | list,
    *,
    top_n: int | None = None,
    labels: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Log a structured panel artifact to the active MLflow run.

    Args:
        panel_type: One of the supported panel types.
        data: The panel data. DataFrames are converted to records.
        top_n: For feature_importance, limit to top N features.
        labels: For confusion_matrix, the class labels.
        extra: Arbitrary extra metadata to include.

    Returns:
        The panel payload that was logged.
    """
    if panel_type not in PANEL_TYPES:
        raise ValueError(f"Unknown panel type {panel_type!r}. Supported: {sorted(PANEL_TYPES)}")

    if isinstance(data, pd.DataFrame):
        serialized = data.to_dict(orient="records")
    elif isinstance(data, list):
        serialized = data
    else:
        serialized = data

    payload: dict[str, Any] = {
        "lens_version": __version__,
        "schema_version": "1",
        "type": panel_type,
        "data": serialized,
    }

    if top_n is not None:
        payload["top_n"] = top_n
    if labels is not None:
        payload["labels"] = labels
    if extra:
        payload.update(extra)

    mlflow.set_tag("lens.version", __version__)
    mlflow.set_tag(f"lens.panel.{panel_type}", "true")

    filename = f"{panel_type}.json"
    log_json_artifact(payload, artifact_path="lens/panels", filename=filename)
    return payload
