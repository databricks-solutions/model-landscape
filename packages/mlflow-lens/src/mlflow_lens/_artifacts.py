"""Shared artifact helpers for writing structured JSON to MLflow runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow


def log_json_artifact(payload: dict, artifact_path: str, filename: str) -> None:
    """Write a dict as a JSON artifact on the active MLflow run.

    Uses a proper temp directory that gets cleaned up, instead of
    hardcoding /tmp paths.

    Args:
        payload: The data to serialize.
        artifact_path: The artifact subdirectory (e.g. "lens").
        filename: The file name within that directory (e.g. "summary.json").
    """
    with tempfile.TemporaryDirectory(prefix="mlflow_lens_") as tmp_dir:
        tmp_file = Path(tmp_dir) / filename
        tmp_file.write_text(json.dumps(payload, indent=2))
        mlflow.log_artifact(str(tmp_file), artifact_path=artifact_path)
