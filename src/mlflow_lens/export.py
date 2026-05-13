"""Export enriched Lens index to Delta tables for BI integration."""

from __future__ import annotations

import pandas as pd

from mlflow_lens._config import is_on_databricks
from mlflow_lens.summary import build, load


def to_delta(
    experiment_ids: list[str] | None = None,
    table_name: str = "lens_index",
    cache_path: str | None = None,
) -> int:
    """Export the Lens index to a Delta table.

    On Databricks, writes via spark. Locally, raises an error.

    Args:
        experiment_ids: If provided, build/refresh the index first.
        table_name: Fully qualified table name (e.g. "catalog.schema.lens_index").
        cache_path: Path to existing cached index. Used if experiment_ids is None.

    Returns:
        Number of rows written.
    """
    if not is_on_databricks():
        raise RuntimeError(
            "to_delta() requires a Databricks runtime with Spark. "
            "Use summary.cache() for local parquet export."
        )

    if experiment_ids:
        index = build(experiment_ids, cache_path=cache_path)
    elif cache_path:
        index = load(cache_path)
    else:
        raise ValueError("Provide experiment_ids or cache_path.")

    if index.empty:
        return 0

    return _write_delta(index, table_name)


def _write_delta(df: pd.DataFrame, table_name: str) -> int:
    """Write a pandas DataFrame to a Delta table via Spark."""
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active Spark session.")

    sdf = spark.createDataFrame(df)
    sdf.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(table_name)
    return len(df)
