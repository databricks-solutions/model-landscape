# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Production Inference
# MAGIC
# MAGIC Your Champion model is live. It's scoring every transaction in real time.
# MAGIC Every prediction gets logged to an **inference table** — a Delta table that
# MAGIC captures what the model saw and what it predicted.
# MAGIC
# MAGIC For the first two weeks, everything looks fine.
# MAGIC
# MAGIC Then the holiday season hits.

# COMMAND ----------

# MAGIC %run ./_resources/00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score with the Champion Model
# MAGIC
# MAGIC Here's how production scoring works on Databricks: load the Champion as a
# MAGIC Spark UDF and score at scale on serverless compute.

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

print(f"✓ Scored {scored.count():,} rows with Champion model")
display(scored.select("transaction_id", "prediction", "is_fraud", *FEATURE_COLS[:3]).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now Fast-Forward 60 Days
# MAGIC
# MAGIC In reality, your model scores transactions continuously. Over time, the data
# MAGIC changes — and your model doesn't know it.
# MAGIC
# MAGIC We're going to simulate 60 days of production inference with a realistic
# MAGIC drift narrative baked in. This isn't random data. Every shift is deliberate:
# MAGIC
# MAGIC | Days | What's happening in the real world | What the model sees |
# MAGIC |------|-----------------------------------|-------------------|
# MAGIC | 1–14 | Business as usual | Stable distributions, good performance |
# MAGIC | 15–25 | Transaction patterns start shifting | Amounts creep up, distances widen |
# MAGIC | 26–35 | **Holiday season hits** | Travel surges 10%→35%, LATAM region spikes, velocity jumps |
# MAGIC | 36–45 | Partial recovery, but a data pipeline breaks | Some features normalize, `device_trust_score` goes null |
# MAGIC | 46–60 | New steady state | Different from training, but stable |
# MAGIC
# MAGIC **This is exactly the scenario from the introduction.** The model's F1 drops
# MAGIC from ~0.92 to ~0.78 during the holiday shift — but without monitoring,
# MAGIC nobody would know.

# COMMAND ----------

# MAGIC %run ./_resources/data_generator

# COMMAND ----------

from datetime import date, timedelta

fraud_start = date.today() - timedelta(days=60)
fraud_df = generate_fraud_inference(n_days=60, rows_per_day=500, start_date=fraud_start, seed=42, model_version="1")
fraud_labels = generate_fraud_labels(fraud_df, label_delay_days=(2, 5), seed=42)

spark.createDataFrame(fraud_df).write.mode("overwrite").saveAsTable(table("fraud_inference"))
spark.createDataFrame(fraud_labels).write.mode("overwrite").saveAsTable(table("fraud_labels"))

print(f"✓ {table('fraud_inference')} — {len(fraud_df):,} rows over 60 days")
print(f"✓ {table('fraud_labels')} — {len(fraud_labels):,} labels (2-5 day delay, {len(fraud_labels)/len(fraud_df):.0%} coverage)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's peek at the drift
# MAGIC
# MAGIC Before we set up monitoring, let's see what the data actually looks like.
# MAGIC Compare `merchant_category` distribution between the baseline period and
# MAGIC the holiday shift:

# COMMAND ----------

import pandas as pd

fraud_df["day"] = (pd.to_datetime(fraud_df["event_ts"]).dt.date - fraud_start).apply(lambda d: d.days)

baseline = fraud_df[fraud_df["day"] < 14]
shifted = fraud_df[(fraud_df["day"] >= 26) & (fraud_df["day"] < 36)]

comparison = pd.DataFrame({
    "Baseline (days 1-14)": baseline["merchant_category"].value_counts(normalize=True).round(3),
    "Holiday (days 26-35)": shifted["merchant_category"].value_counts(normalize=True).round(3),
})
comparison["Shift"] = comparison["Holiday (days 26-35)"] - comparison["Baseline (days 1-14)"]
display(comparison.sort_values("Shift", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC Travel jumped from ~10% to ~35%. Retail and grocery dropped. This is
# MAGIC a massive distribution shift — and it's exactly the kind of thing that
# MAGIC degrades model performance silently.
# MAGIC
# MAGIC Let's also look at the `device_trust_score` null rate over time:

# COMMAND ----------

null_by_day = fraud_df.groupby("day")["device_trust_score"].apply(lambda x: x.isna().mean()).reset_index()
null_by_day.columns = ["day", "null_rate"]
display(null_by_day)

# COMMAND ----------

# MAGIC %md
# MAGIC The null rate spikes from ~2% to ~12% around days 30-45 — an upstream
# MAGIC data pipeline broke. Model Landscape will flag this automatically.

# COMMAND ----------

# MAGIC %md
# MAGIC ## A Second Model: Predictive Maintenance
# MAGIC
# MAGIC Real ML teams don't monitor one model. They monitor a fleet.
# MAGIC
# MAGIC Your company also has a **predictive maintenance** team monitoring industrial
# MAGIC equipment across three plants. Their regression model predicts remaining useful
# MAGIC life (RUL) in hours. It has completely different drift patterns:
# MAGIC
# MAGIC - **Seasonal temperature shifts** (summer onset changes ambient readings)
# MAGIC - **Equipment aging** (vibration increases as machinery wears out)
# MAGIC - **Sensor outages** (oil viscosity sensor goes offline at Plant East)
# MAGIC
# MAGIC Adding this second model lets you see **fleet monitoring** in action —
# MAGIC two models with different problems, visible at a glance.

# COMMAND ----------

maint_start = date.today() - timedelta(days=90)
maint_df = generate_maintenance_inference(n_days=90, rows_per_day=280, start_date=maint_start, seed=137)
maint_labels = generate_maintenance_labels(maint_df, failure_rate=0.03, seed=137)

spark.createDataFrame(maint_df).write.mode("overwrite").saveAsTable(table("maintenance_inference"))
spark.createDataFrame(maint_labels).write.mode("overwrite").saveAsTable(table("maintenance_labels"))

print(f"✓ {table('maintenance_inference')} — {len(maint_df):,} rows over 90 days")
print(f"✓ {table('maintenance_labels')} — {len(maint_labels):,} labels (sparse — only at equipment failures)")

# COMMAND ----------

# MAGIC %md
# MAGIC The data is in production. The drift is happening. The null spike is real.
# MAGIC The model is silently degrading.
# MAGIC
# MAGIC **Next** → `04_monitor`: The system that catches all of this.
