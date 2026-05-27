"""MLflow version detection and compatibility helpers."""

from __future__ import annotations

import mlflow

_mlflow_version = tuple(int(x) for x in mlflow.__version__.split(".")[:2])


def mlflow_major() -> int:
    return _mlflow_version[0]


def has_logged_models() -> bool:
    """True if the MLflow server supports LoggedModel (3.x+)."""
    return _mlflow_version[0] >= 3 and hasattr(mlflow, "search_logged_models")
