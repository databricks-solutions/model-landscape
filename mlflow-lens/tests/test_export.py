import pytest
from mlflow_lens.export import to_delta


def test_to_delta_raises_outside_databricks():
    """to_delta should fail cleanly when not on Databricks."""
    with pytest.raises(RuntimeError, match="requires a Databricks runtime"):
        to_delta(experiment_ids=["1"], table_name="test.table")


def test_to_delta_requires_args():
    """Must provide experiment_ids or cache_path."""
    with pytest.raises((RuntimeError, ValueError)):
        to_delta(table_name="test.table")
