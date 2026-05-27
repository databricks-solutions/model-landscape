# Databricks notebook source
# MAGIC %md
# MAGIC # Ship It, Watch It Break, Recover
# MAGIC
# MAGIC The Champion model is live. Over the next 60 days the data shifts: a
# MAGIC holiday surge changes the transaction mix, then an upstream pipeline
# MAGIC breaks and starts emitting nulls. F1 quietly drops from ~0.92 to ~0.78.
# MAGIC
# MAGIC In this notebook:
# MAGIC
# MAGIC 1. **Score** the test set with the Champion model end-to-end on Spark.
# MAGIC 2. **Fast-forward 60 days** of drifted inference (+ a second model for
# MAGIC    fleet monitoring).
# MAGIC 3. **Stand up monitors** in Model Landscape's observability store and
# MAGIC    bootstrap the metrics.
# MAGIC 4. **Investigate** the incidents and the F1 drop.
# MAGIC 5. **Retrain** on the new distribution, validate, promote v2, and verify
# MAGIC    recovery.

# COMMAND ----------

# MAGIC %run ./_resources/00_setup

# COMMAND ----------

# Precheck: notebook 01 must have run successfully — fail fast with a
# clear message if no Champion alias exists yet.
import mlflow

_registered = model_name("fraud_detector")
try:
    mlflow.MlflowClient().get_model_version_by_alias(_registered, "Champion")
except mlflow.exceptions.MlflowException as e:
    raise RuntimeError(
        f"No Champion alias on {_registered}. "
        "Run `01_train_and_ship` first (or `databricks bundle run tutorial_mlops` "
        "to run both notebooks in order)."
    ) from e
print(f"✓ Champion model present: {_registered}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score with the Champion model
# MAGIC
# MAGIC Production scoring on Databricks: load the Champion as a Spark UDF,
# MAGIC score at scale on serverless.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

registered = model_name("fraud_detector")
FEATURE_COLS = [
    "transaction_amount", "device_trust_score", "distance_from_home_km",
    "velocity_24h", "hour_of_day", "is_weekend", "account_age_days",
]

champion_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{registered}@Champion")
scored = (
    spark.table(table("fraud_test"))
    .withColumn("prediction", champion_udf(*[F.col(c) for c in FEATURE_COLS]))
)
print(f"✓ Scored {scored.count():,} rows with Champion")
display(scored.select("transaction_id", "prediction", "is_fraud", *FEATURE_COLS[:3]).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fast-forward 60 days of production inference
# MAGIC
# MAGIC | Days | Real world | What the model sees |
# MAGIC |------|------------|---------------------|
# MAGIC | 1–14 | Business as usual | Stable distributions, good performance |
# MAGIC | 15–25 | Transaction mix starts shifting | Amounts creep up, distances widen |
# MAGIC | 26–35 | **Holiday season hits** | Travel 10% → 35%, LATAM spikes, velocity jumps |
# MAGIC | 36–45 | Partial recovery + a data pipeline breaks | `device_trust_score` goes null |
# MAGIC | 46–60 | New steady state | Different from training, but stable |
# MAGIC
# MAGIC We also bring in a second model — a regression model predicting Remaining
# MAGIC Useful Life for industrial equipment — so you can see **fleet monitoring**:
# MAGIC two models, two failure modes, one view.

# COMMAND ----------

# MAGIC %run ./_resources/data_generator

# COMMAND ----------

from datetime import date, timedelta

fraud_start = date.today() - timedelta(days=60)
fraud_df = generate_fraud_inference(n_days=60, rows_per_day=500, start_date=fraud_start, seed=42, model_version="1")
fraud_labels = generate_fraud_labels(fraud_df, label_delay_days=(2, 5), seed=42)

spark.createDataFrame(fraud_df).write.mode("overwrite").saveAsTable(table("fraud_inference"))
spark.createDataFrame(fraud_labels).write.mode("overwrite").saveAsTable(table("fraud_labels"))
print(f"✓ {table('fraud_inference'):<60s} {len(fraud_df):>6,} rows / 60 days")
print(f"✓ {table('fraud_labels'):<60s} {len(fraud_labels):>6,} labels (2-5 day delay)")

maint_start = date.today() - timedelta(days=90)
maint_df = generate_maintenance_inference(n_days=90, rows_per_day=280, start_date=maint_start, seed=137)
maint_labels = generate_maintenance_labels(maint_df, failure_rate=0.03, seed=137)

spark.createDataFrame(maint_df).write.mode("overwrite").saveAsTable(table("maintenance_inference"))
spark.createDataFrame(maint_labels).write.mode("overwrite").saveAsTable(table("maintenance_labels"))
print(f"✓ {table('maintenance_inference'):<60s} {len(maint_df):>6,} rows / 90 days")
print(f"✓ {table('maintenance_labels'):<60s} {len(maint_labels):>6,} labels (sparse, at failures)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What the drift looks like up close

# COMMAND ----------

import pandas as pd

fraud_df["day"] = (pd.to_datetime(fraud_df["event_ts"]).dt.date - fraud_start).apply(lambda d: d.days)
baseline = fraud_df[fraud_df["day"] < 14]
shifted = fraud_df[(fraud_df["day"] >= 26) & (fraud_df["day"] < 36)]

merchant_shift = pd.DataFrame({
    "baseline (days 1-14)": baseline["merchant_category"].value_counts(normalize=True).round(3),
    "holiday (days 26-35)": shifted["merchant_category"].value_counts(normalize=True).round(3),
})
merchant_shift["shift"] = merchant_shift["holiday (days 26-35)"] - merchant_shift["baseline (days 1-14)"]
print("Travel surged from ~10% to ~35%. Retail and grocery dropped.\n")
display(merchant_shift.sort_values("shift", ascending=False))

null_by_day = (
    fraud_df.groupby("day")["device_trust_score"]
    .apply(lambda x: x.isna().mean())
    .reset_index(name="device_trust_null_rate")
)
print("And the null spike from the broken pipeline:")
display(null_by_day)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stand up monitors and bootstrap metrics
# MAGIC
# MAGIC Model Landscape stores monitor configs, drift metrics, performance
# MAGIC metrics, and incidents in Unity Catalog Delta tables — the
# MAGIC **observability store**. We point it at the inference tables, hand it a
# MAGIC contract (which columns are features / predictions / timestamps), and
# MAGIC the bootstrap pass scans the full history.

# COMMAND ----------

dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse ID")
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")

if not SQL_WAREHOUSE_ID:
    from databricks.sdk import WorkspaceClient
    warehouses = [w for w in WorkspaceClient().warehouses.list() if w.state.value == "RUNNING"]
    if not warehouses:
        raise ValueError("No running SQL warehouse. Set the sql_warehouse_id widget.")
    SQL_WAREHOUSE_ID = warehouses[0].id
    print(f"Using warehouse: {warehouses[0].name} ({SQL_WAREHOUSE_ID})")

OBSERVABILITY_CATALOG = "model_observability"
OBSERVABILITY_SCHEMA = "control_plane"
obs = f"{OBSERVABILITY_CATALOG}.{OBSERVABILITY_SCHEMA}"

from model_landscape.services.control_plane import build_repository

repository = build_repository(
    warehouse_id=SQL_WAREHOUSE_ID,
    catalog=OBSERVABILITY_CATALOG,
    schema=OBSERVABILITY_SCHEMA,
)
repository.ensure_control_plane(create_catalog=True)
print(f"✓ Observability store ready at {obs}")

# COMMAND ----------

# MAGIC %run ./_resources/monitors

# COMMAND ----------

fraud_monitor = make_monitor_config(
    model_key="fraud_detector_v1",
    display_name="Fraud Detector",
    source_table=table("fraud_inference"),
    problem_type="classification",
    feature_columns=FRAUD_FEATURES,
    prediction_score_col="prediction_proba",
    slice_columns=("region",),
    categorical_columns=("merchant_category", "region"),
    labels_table=table("fraud_labels"),
    labels_join_col="entity_id",
    labels_order_col="label_timestamp",
    mlflow_experiment=EXPERIMENT_PATH,
)

maintenance_monitor = make_monitor_config(
    model_key="rul_predictor_v1",
    display_name="Predictive Maintenance — RUL",
    source_table=table("maintenance_inference"),
    problem_type="regression",
    feature_columns=MAINT_FEATURES,
    slice_columns=("site",),
    categorical_columns=("equipment_class", "site"),
    labels_table=table("maintenance_labels"),
    labels_join_col="entity_id",
    labels_order_col="label_timestamp",
)

repository.upsert_monitor_config(fraud_monitor)
repository.upsert_monitor_config(maintenance_monitor)
print("✓ Monitors created: Fraud Detector + Predictive Maintenance")

# COMMAND ----------

from model_landscape.services.refresh_runner import run_refresh_cycle

for monitor in (fraud_monitor, maintenance_monitor):
    print(f"Bootstrapping {monitor.display_name}…")
    result = run_refresh_cycle(repository=repository, model_key=monitor.model_key, scope="bootstrap")
    print(f"  ✓ {result.drift_rows} drift rows · {result.performance_rows} performance rows · {result.incident_rows} incidents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What did Model Landscape find?

# COMMAND ----------

print("=== Open incidents ===")
display(spark.sql(f"""
    SELECT model_key, feature_name, metric_name, severity,
           ROUND(metric_value, 4) AS metric_value
    FROM {obs}.incidents
    ORDER BY severity DESC, model_key
"""))

print("=== Top drifting features (fraud model) ===")
display(spark.sql(f"""
    SELECT feature_name, metric_name,
           ROUND(MAX(metric_value), 4) AS peak_drift,
           COUNT(*) AS windows_drifted
    FROM {obs}.drift_metrics
    WHERE model_key = 'fraud_detector_v1' AND metric_name = 'psi' AND metric_value > 0.1
    GROUP BY feature_name, metric_name
    ORDER BY peak_drift DESC
"""))

workspace_url = spark.conf.get("spark.databricks.workspaceUrl", "your-workspace")
print(f"\nOpen Model Landscape → https://{workspace_url}/apps/model-landscape")
print("Fleet overview · Drift Analysis · Performance · Data Quality · Incidents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain on recent labeled data
# MAGIC
# MAGIC LATAM went from ~5% to ~30% of transactions. Travel surged. The model
# MAGIC was trained on a world where LATAM was 5% and travel was 10% — so it
# MAGIC doesn't know how to score these well. Combine the original training set
# MAGIC with recent labeled production data and retrain.

# COMMAND ----------

from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb

mlflow.set_experiment(EXPERIMENT_PATH)
client = MlflowClient()

original = spark.table(table("fraud_train")).toPandas()
recent = (
    spark.table(table("fraud_inference")).alias("inf")
    .join(spark.table(table("fraud_labels")).alias("lab"), on="entity_id", how="inner")
    .select(*[F.col(f"inf.{c}") for c in FEATURE_COLS], F.col("lab.label").alias("is_fraud"))
    .toPandas()
)
combined = pd.concat([original[FEATURE_COLS + ["is_fraud"]], recent], ignore_index=True)
X_train, y_train = combined[FEATURE_COLS], combined["is_fraud"]

test_pdf = spark.table(table("fraud_test")).toPandas()
X_test, y_test = test_pdf[FEATURE_COLS], test_pdf["is_fraud"]

print(f"Original: {len(original):,}  Recent: {len(recent):,}  Combined: {len(combined):,}")

with mlflow.start_run(run_name="xgboost_v2_retrained") as retrain_run:
    model_v2 = xgb.XGBClassifier(
        n_estimators=250, max_depth=7, learning_rate=0.08,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42, eval_metric="logloss",
    )
    model_v2.fit(X_train, y_train)
    y_pred = model_v2.predict(X_test)
    metrics = {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"task": "fraud_detection", "dataset_version": "v2", "retrain_reason": "drift_detected"})
    mlflow.sklearn.log_model(model_v2, artifact_path="model", input_example=X_test.head(5))

    # Lens panels for v2 so the warehouse app can diff v1 vs v2 visually
    from mlflow_lens.classifier import classification_report, confusion_matrix, roc_auc
    from mlflow_lens.model_selection import feature_importances
    roc_auc(model_v2, X_test, y_test, log=True)
    confusion_matrix(model_v2, X_test, y_test, labels=[0, 1], log=True)
    classification_report(model_v2, X_test, y_test, log=True)
    feature_importances(model_v2, FEATURE_COLS, log=True)

    print(f"✓ Retrained: F1={metrics['f1']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate, promote, and deploy v2

# COMMAND ----------

registered = model_name("fraud_detector")
mv = mlflow.register_model(f"runs:/{retrain_run.info.run_id}/model", registered)
client.update_model_version(
    name=registered, version=mv.version,
    description=(
        f"Retrained XGBoost v2. Includes holiday-season data. F1={metrics['f1']:.4f}. "
        "Retrained in response to drift detected by Model Landscape."
    ),
)

champion = mlflow.sklearn.load_model(f"models:/{registered}@Champion")
champion_f1 = f1_score(y_test, champion.predict(X_test))
print(f"Champion (v1):    F1={champion_f1:.4f}")
print(f"Challenger (v2):  F1={metrics['f1']:.4f}")

if metrics["f1"] >= champion_f1 - 0.01:
    client.set_registered_model_alias(registered, "Champion", mv.version)
    for k, v in [
        ("validation_passed", "true"),
        ("validation_f1", f"{metrics['f1']:.4f}"),
        ("promotion_reason", "drift_response"),
    ]:
        client.set_model_version_tag(registered, mv.version, k, v)
    print(f"\n✓ v{mv.version} promoted to Champion")
else:
    print("\n✗ Challenger didn't improve — keeping current Champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate the recovery
# MAGIC
# MAGIC Append 30 more days of inference scored by v2. On the next refresh,
# MAGIC Model Landscape sees the version transition and drift dropping back
# MAGIC toward baseline.

# COMMAND ----------

v2_start = date.today() - timedelta(days=30)
v2_df = generate_fraud_inference(n_days=30, rows_per_day=500, start_date=v2_start, seed=1000, model_version="2")
v2_labels = generate_fraud_labels(v2_df, label_delay_days=(2, 5), seed=1000)

spark.createDataFrame(v2_df).write.mode("append").saveAsTable(table("fraud_inference"))
if len(v2_labels):
    spark.createDataFrame(v2_labels).write.mode("append").saveAsTable(table("fraud_labels"))
print(f"✓ Appended {len(v2_df):,} v2 inference rows + {len(v2_labels):,} labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The loop is closed
# MAGIC
# MAGIC After the next refresh, Model Landscape shows:
# MAGIC
# MAGIC - **Version transition** v1 → v2 in the timeline
# MAGIC - **Drift decreasing** in the v2 period — the model sees familiar data
# MAGIC - **F1 recovering** toward baseline
# MAGIC - **Incidents closing** as features come back below thresholds
# MAGIC
# MAGIC ```
# MAGIC  Detect ──▶ Investigate ──▶ Retrain ──▶ Validate ──▶ Promote ──▶ Verify
# MAGIC    │                                                               │
# MAGIC    └───────────────── continuous loop ◀────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC That's the whole tour: two notebooks, raw data to live monitoring to a
# MAGIC closed loop. Auditable, governed, and one `databricks bundle run` away
# MAGIC from production.
