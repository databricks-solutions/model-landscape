"""Composes with mlflow.start_run() to log structured context and metadata."""

import subprocess

import mlflow

from mlflow_lens._config import (
    get_cluster_id,
    get_notebook_path,
    get_runtime_version,
    get_workspace_url,
    is_on_databricks,
)
from mlflow_lens._version import __version__


def auto_log_context() -> dict[str, str]:
    """Log contextual metadata to the active MLflow run as lens-prefixed tags.

    Captures environment info that MLflow does not log by default:
    workspace URL, cluster ID, runtime version, notebook path, and git SHA.

    Returns the dict of tags that were logged.
    """
    tags: dict[str, str] = {"lens.version": __version__}

    if is_on_databricks():
        if url := get_workspace_url():
            tags["lens.workspace_url"] = url
        if cid := get_cluster_id():
            tags["lens.cluster_id"] = cid
        if rtv := get_runtime_version():
            tags["lens.runtime_version"] = rtv
        if nbp := get_notebook_path():
            tags["lens.notebook_path"] = nbp
    else:
        tags["lens.environment"] = "local"

    git_sha = _get_git_sha()
    if git_sha:
        tags["lens.git_sha"] = git_sha

    mlflow.set_tags(tags)
    return tags


def log(data: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict into dot-notation keys and log as MLflow params.

    Example::

        log({"data": {"source": "v3", "rows": 50000}})
        # logs params: data.source="v3", data.rows="50000"

    Returns the flattened dict that was logged.
    """
    flat = _flatten(data, prefix)
    mlflow.log_params(flat)
    return flat


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = str(v)
    return items


def _get_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
