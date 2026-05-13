# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4: The System That Catches What You'd Miss
# MAGIC
# MAGIC Your fraud model has been scoring transactions for 60 days. Somewhere around
# MAGIC day 26, a holiday shift changed the data distribution. The F1 score dropped.
# MAGIC A data pipeline broke. Features went null.
# MAGIC
# MAGIC Without monitoring, this goes unnoticed for weeks. With **Model Landscape**,
# MAGIC you see it in hours.
# MAGIC
# MAGIC This notebook programmatically:
# MAGIC 1. Sets up the observability store (monitoring state in Unity Catalog)
# MAGIC 2. Creates monitors for both models
# MAGIC 3. Runs a bootstrap refresh to compute every metric across the full history
# MAGIC 4. Shows you what it found

# COMMAND ----------

# MAGIC %run ./_resources/00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to the Observability Store
# MAGIC
# MAGIC Model Landscape stores monitoring state — drift metrics, performance metrics,
# MAGIC incidents, monitor configs — in Unity Catalog Delta tables. We call this the
# MAGIC **observability store**.

# COMMAND ----------

import os

OBSERVABILITY_CATALOG = "model_observability"
OBSERVABILITY_SCHEMA = "control_plane"

# Auto-discover a running SQL warehouse
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse ID")
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")

if not SQL_WAREHOUSE_ID:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    warehouses = [wh for wh in w.warehouses.list() if wh.state.value == "RUNNING"]
    if warehouses:
        SQL_WAREHOUSE_ID = warehouses[0].id
        print(f"Using warehouse: {warehouses[0].name} ({SQL_WAREHOUSE_ID})")
    else:
        raise ValueError("No running SQL warehouse. Set the sql_warehouse_id widget.")

# COMMAND ----------

from model_landscape.services.control_plane import build_repository

repository = build_repository(
    warehouse_id=SQL_WAREHOUSE_ID,
    catalog=OBSERVABILITY_CATALOG,
    schema=OBSERVABILITY_SCHEMA,
)
repository.ensure_control_plane(create_catalog=True)
print(f"✓ Observability store ready: {OBSERVABILITY_CATALOG}.{OBSERVABILITY_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Fraud Monitor
# MAGIC
# MAGIC A monitor defines: what table to watch, which columns are features vs predictions
# MAGIC vs timestamps, how to compute baselines, and how often to refresh.

# COMMAND ----------

from model_landscape.domain.models import (
    MonitorConfig, InferenceContract, BaselinePolicy, MLflowLineage,
)

fraud_monitor = MonitorConfig(
    model_key="fraud_detector_v1",
    display_name="Fraud Detector",
    source_table=table("fraud_inference"),
    problem_type="classification",
    contract=InferenceContract(
        timestamp_col="event_ts",
        prediction_col="prediction",
        prediction_score_col="prediction_proba",
        entity_id_col="entity_id",
        model_version_col="model_version",
        feature_columns=(
            "transaction_amount", "device_trust_score", "distance_from_home_km",
            "velocity_24h", "hour_of_day", "is_weekend", "account_age_days",
            "merchant_category", "region",
        ),
        slice_columns=("region",),
        categorical_columns=("merchant_category", "region"),
    ),
    baseline=BaselinePolicy(kind="rolling", n_days=7),
    labels_table=table("fraud_labels"),
    labels_join_col="entity_id",
    labels_order_col="label_timestamp",
    drift_cadence_preset="daily",
    performance_cadence_preset="daily_7d_repair",
    schedule_enabled=True,
    mlflow=MLflowLineage(experiment_name=EXPERIMENT_PATH),
    created_by="tutorial",
)

repository.upsert_monitor_config(fraud_monitor)
print(f"✓ Monitor created: {fraud_monitor.display_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Maintenance Monitor
# MAGIC
# MAGIC Different problem type (regression), different features (sensors),
# MAGIC different drift patterns. Same monitoring framework.

# COMMAND ----------

maintenance_monitor = MonitorConfig(
    model_key="rul_predictor_v1",
    display_name="Predictive Maintenance — RUL",
    source_table=table("maintenance_inference"),
    problem_type="regression",
    contract=InferenceContract(
        timestamp_col="event_ts",
        prediction_col="prediction",
        entity_id_col="entity_id",
        model_version_col="model_version",
        feature_columns=(
            "vibration_mm_s", "temperature_c", "pressure_kpa", "rpm",
            "oil_viscosity", "power_output_kw", "ambient_temp_c",
            "operating_hours_since_service", "load_factor",
            "equipment_class", "site",
        ),
        slice_columns=("site",),
        categorical_columns=("equipment_class", "site"),
    ),
    baseline=BaselinePolicy(kind="rolling", n_days=7),
    labels_table=table("maintenance_labels"),
    labels_join_col="entity_id",
    labels_order_col="label_timestamp",
    drift_cadence_preset="daily",
    performance_cadence_preset="daily_7d_repair",
    schedule_enabled=True,
    created_by="tutorial",
)

repository.upsert_monitor_config(maintenance_monitor)
print(f"✓ Monitor created: {maintenance_monitor.display_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Bootstrap Refresh
# MAGIC
# MAGIC This is where the magic happens. Model Landscape computes drift (PSI/KL/JS
# MAGIC per feature per day), performance (F1 over time), data quality (null rates,
# MAGIC row counts), and opens incidents when thresholds are breached.
# MAGIC
# MAGIC The bootstrap scans the full history. Subsequent refreshes are incremental.

# COMMAND ----------

from model_landscape.services.refresh_runner import run_refresh_cycle

print("Refreshing fraud monitor...")
fraud_result = run_refresh_cycle(
    repository=repository,
    model_key="fraud_detector_v1",
    scope="bootstrap",
)
print(f"✓ {fraud_result.drift_rows} drift rows, {fraud_result.performance_rows} performance rows, {fraud_result.incident_rows} incidents")

# COMMAND ----------

print("Refreshing maintenance monitor...")
maint_result = run_refresh_cycle(
    repository=repository,
    model_key="rul_predictor_v1",
    scope="bootstrap",
)
print(f"✓ {maint_result.drift_rows} drift rows, {maint_result.performance_rows} performance rows, {maint_result.incident_rows} incidents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What Did Model Landscape Find?

# COMMAND ----------

obs = f"{OBSERVABILITY_CATALOG}.{OBSERVABILITY_SCHEMA}"

print("=== Open Incidents ===")
print("These are the features that crossed drift thresholds:\n")
display(spark.sql(f"""
    SELECT model_key, feature_name, metric_name, severity,
           ROUND(metric_value, 4) as metric_value
    FROM {obs}.incidents
    ORDER BY severity DESC, model_key
"""))

# COMMAND ----------

print("=== Top Drifting Features (Fraud Model) ===\n")
display(spark.sql(f"""
    SELECT feature_name, metric_name,
           ROUND(MAX(metric_value), 4) as peak_drift,
           COUNT(*) as windows_drifted
    FROM {obs}.drift_metrics
    WHERE model_key = 'fraud_detector_v1'
      AND metric_name = 'psi'
      AND metric_value > 0.1
    GROUP BY feature_name, metric_name
    ORDER BY peak_drift DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Open Model Landscape

# COMMAND ----------

workspace_url = spark.conf.get("spark.databricks.workspaceUrl", "your-workspace")
app_url = f"https://{workspace_url}/apps/model-landscape"

print(f"Open Model Landscape → {app_url}")
print()
print("What you'll see:")
print("  Fleet Overview  → Two models, their max PSI, open incidents")
print("  Drift Analysis  → The holiday shift rippling across features over time")
print("  Performance     → F1 dropping during the drift, recovering after")
print("  Data Quality    → The device_trust_score null spike, clear as day")
print("  Incidents       → Exactly when each feature crossed the threshold")

# COMMAND ----------

# MAGIC %md
# MAGIC Remember the 2am phone call from the introduction?
# MAGIC
# MAGIC Model Landscape would have caught the holiday drift on **day 15** — not day 14
# MAGIC of silent degradation. It would have opened an incident, shown you which
# MAGIC features shifted, and told you the severity.
# MAGIC
# MAGIC That's the difference between reactive firefighting and proactive governance.
# MAGIC
# MAGIC **Next** → `05_close_the_loop`: The drift is detected. Now close the loop.
