# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial Setup
# MAGIC
# MAGIC Shared configuration for the Model Landscape MLOps tutorial.
# MAGIC Run this cell at the top of every tutorial notebook via `%run ./_resources/00_setup`.
# MAGIC
# MAGIC Dependencies (mlflow, xgboost, scikit-learn) are provided by the
# MAGIC DAB serverless environment — no `%pip install` needed.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "model_landscape_tutorial", "Schema")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

import mlflow

# Set experiment path scoped to current user
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
EXPERIMENT_PATH = f"/Users/{current_user}/model_landscape_tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"Catalog:    {CATALOG}")
print(f"Schema:     {SCHEMA}")
print(f"Experiment: {EXPERIMENT_PATH}")
print(f"User:       {current_user}")

# COMMAND ----------

# Helper: fully qualified table name
def table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"

# Helper: registered model name
def model_name(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"
