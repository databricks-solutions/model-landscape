# Databricks notebook source
# MAGIC %md
# MAGIC # Step 5: Close the Loop
# MAGIC
# MAGIC Model Landscape flagged critical drift around day 26. `merchant_category`
# MAGIC shifted. `region` shifted. `velocity_24h` spiked. The F1 score dropped
# MAGIC from ~0.92 to ~0.78.
# MAGIC
# MAGIC In the old world, someone would have noticed this in a quarterly review —
# MAGIC six weeks too late. Now you know within a day. Here's how you respond.

# COMMAND ----------

# MAGIC %run ./_resources/00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Investigate: What Drifted and Why?
# MAGIC
# MAGIC Start with the inference data. Compare the baseline distribution to the
# MAGIC drift window.

# COMMAND ----------

from pyspark.sql import functions as F

inference_df = spark.table(table("fraud_inference"))

# Derive windows from the data itself (not current_date) so this works regardless of when you run it
min_date = inference_df.select(F.min(F.col("event_ts")).cast("date")).first()[0]
baseline = inference_df.filter(F.col("event_ts").between(
    F.lit(min_date), F.date_add(F.lit(min_date), 14)))
drift_window = inference_df.filter(F.col("event_ts").between(
    F.date_add(F.lit(min_date), 26), F.date_add(F.lit(min_date), 35)))

print("Region distribution — Baseline vs Drift:")
display(
    baseline.groupBy("region").count().withColumnRenamed("count", "baseline_count")
    .join(
        drift_window.groupBy("region").count().withColumnRenamed("count", "drift_count"),
        on="region", how="outer"
    ).orderBy("region")
)

# COMMAND ----------

# MAGIC %md
# MAGIC LATAM went from ~5% to ~30% of transactions. Travel bookings surged.
# MAGIC The model was trained on a world where LATAM was 5% and travel was 10%.
# MAGIC It doesn't know how to score these transactions accurately.
# MAGIC
# MAGIC **The fix:** retrain on data that includes the new distribution.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain on Recent Data

# COMMAND ----------

import pandas as pd
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb

FEATURE_COLS = [
    "transaction_amount", "device_trust_score", "distance_from_home_km",
    "velocity_24h", "hour_of_day", "is_weekend", "account_age_days",
]

mlflow.set_experiment(EXPERIMENT_PATH)

# Combine original training data with recent labeled production data
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

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_v2_retrained") as retrain_run:
    model_v2 = xgb.XGBClassifier(
        n_estimators=250, max_depth=7, learning_rate=0.08,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42, eval_metric="logloss",
    )
    model_v2.fit(X_train, y_train)
    y_pred = model_v2.predict(X_test)

    metrics = {"f1": f1_score(y_test, y_pred), "precision": precision_score(y_test, y_pred), "recall": recall_score(y_test, y_pred)}
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"task": "fraud_detection", "dataset_version": "v2", "retrain_reason": "drift_detected"})
    mlflow.sklearn.log_model(model_v2, artifact_path="model", input_example=X_test.head(5))

    print(f"✓ Retrained: F1={metrics['f1']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate and Promote

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
registered = model_name("fraud_detector")

mv = mlflow.register_model(f"runs:/{retrain_run.info.run_id}/model", registered)
client.update_model_version(
    name=registered, version=mv.version,
    description=f"Retrained XGBoost v2. Includes holiday season data. F1={metrics['f1']:.4f}. Retrained due to drift detected by Model Landscape.",
)

# Compare against current Champion
champion_model = mlflow.sklearn.load_model(f"models:/{registered}@Champion")
champion_f1 = f1_score(y_test, champion_model.predict(X_test))

print(f"Champion (v1): F1={champion_f1:.4f}")
print(f"Challenger (v2): F1={metrics['f1']:.4f}")

if metrics["f1"] >= champion_f1 - 0.01:
    client.set_registered_model_alias(registered, "Champion", mv.version)
    client.set_model_version_tag(registered, mv.version, "validation_passed", "true")
    client.set_model_version_tag(registered, mv.version, "validation_f1", f"{metrics['f1']:.4f}")
    client.set_model_version_tag(registered, mv.version, "promotion_reason", "drift_response")
    print(f"\n✓ v{mv.version} promoted to Champion")
else:
    print("\n✗ Challenger didn't improve — keeping current Champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy v2 to Production

# COMMAND ----------

# MAGIC %run ./_resources/data_generator

# COMMAND ----------

from datetime import date, timedelta

# Start v2 data 30 days ago (not today) so labels have time to "arrive"
v2_start = date.today() - timedelta(days=30)
v2_df = generate_fraud_inference(n_days=30, rows_per_day=500, start_date=v2_start, seed=1000, model_version="2")
v2_labels = generate_fraud_labels(v2_df, label_delay_days=(2, 5), seed=1000)

spark.createDataFrame(v2_df).write.mode("append").saveAsTable(table("fraud_inference"))
if len(v2_labels) > 0:
    spark.createDataFrame(v2_labels).write.mode("append").saveAsTable(table("fraud_labels"))

print(f"✓ Appended {len(v2_df):,} v2 inference rows + {len(v2_labels):,} labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Loop is Closed
# MAGIC
# MAGIC After the next refresh, Model Landscape will show:
# MAGIC - **Version transition** visible in the timeline (v1 → v2)
# MAGIC - **Drift decreasing** in the v2 period — the model sees familiar data now
# MAGIC - **Performance recovering** as F1 climbs back toward baseline
# MAGIC - **Incidents closing** as features return below thresholds
# MAGIC
# MAGIC ```
# MAGIC  Detect ──▶ Investigate ──▶ Retrain ──▶ Validate ──▶ Promote ──▶ Verify
# MAGIC    │                                                               │
# MAGIC    └───────────────── continuous loop ◀────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC This is production MLOps. Not just training models — **keeping them healthy**.
# MAGIC Every step is auditable. Every decision is traceable. Every incident has a
# MAGIC response. And the whole thing runs as a single `databricks bundle run`.
# MAGIC
# MAGIC **That's it.** Five notebooks. Raw data to live monitoring to closed loop.
# MAGIC All automated. All governed. All CI/CD ready.
