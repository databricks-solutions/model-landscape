"""Shared artifact helpers for writing structured JSON to MLflow runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars/arrays as native Python types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def log_json_artifact(payload: dict, artifact_path: str, filename: str) -> None:
    """Write a dict as a JSON artifact on the active MLflow run.

    Numpy scalars and arrays are serialized as their native Python equivalents.

    Args:
        payload: The data to serialize.
        artifact_path: The artifact subdirectory (e.g. "lens").
        filename: The file name within that directory (e.g. "summary.json").
    """
    with tempfile.TemporaryDirectory(prefix="mlflow_lens_") as tmp_dir:
        tmp_file = Path(tmp_dir) / filename
        tmp_file.write_text(json.dumps(payload, indent=2, cls=_NumpyEncoder))
        mlflow.log_artifact(str(tmp_file), artifact_path=artifact_path)
