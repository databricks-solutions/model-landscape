from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_spark_session():
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return SparkSession.builder.appName("model-landscape-refresh").getOrCreate()
