"""Cost context tagging for system table joins.

The SDK side is intentionally thin: it tags the active run with enough
metadata (cluster ID, start time) so the app backend can join against
Databricks system tables to compute actual DBU/dollar costs.
"""

from __future__ import annotations

import mlflow

from mlflow_lens._config import get_cluster_id, is_on_databricks
from mlflow_lens._version import __version__


def log_cost_context() -> dict[str, str]:
    """Tag the active run with metadata needed for cost attribution.

    On Databricks, this captures the cluster ID so the app can join
    against ``system.billing.usage``. Locally, this is a no-op that
    returns an empty dict.

    Returns:
        The dict of tags that were set.
    """
    tags: dict[str, str] = {"lens.version": __version__}

    if not is_on_databricks():
        mlflow.set_tags(tags)
        return tags

    cluster_id = get_cluster_id()
    if cluster_id:
        tags["lens.cluster_id"] = cluster_id
        tags["lens.cost_eligible"] = "true"

    mlflow.set_tags(tags)
    return tags
