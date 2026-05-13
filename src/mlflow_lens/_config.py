import os


def is_on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_workspace_url() -> str | None:
    for key in ("DATABRICKS_HOST", "SPARK_DATABRICKS_WORKSPACE_URL"):
        val = os.environ.get(key)
        if val:
            return val.rstrip("/")
    return None


def get_cluster_id() -> str | None:
    return os.environ.get("DB_CLUSTER_ID") or os.environ.get("SPARK_DATABRICKS_CLUSTERID")


def get_runtime_version() -> str | None:
    return os.environ.get("DATABRICKS_RUNTIME_VERSION")


def get_notebook_path() -> str | None:
    return os.environ.get("DATABRICKS_NOTEBOOK_PATH")
