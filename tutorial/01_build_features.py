# Databricks notebook source
# MAGIC %md
# MAGIC # It's 2am. Your Phone Buzzes.
# MAGIC
# MAGIC The fraud model you deployed three months ago just approved $2.3M in
# MAGIC fraudulent transactions. A holiday shopping surge shifted the transaction mix —
# MAGIC more travel bookings, more international purchases, higher amounts. Your model
# MAGIC had never seen data like this. It didn't fail dramatically. It failed *quietly*.
# MAGIC
# MAGIC **Nobody noticed for 14 days.**
# MAGIC
# MAGIC This tutorial builds the system that would have caught this on day 1.
# MAGIC Five notebooks. ~45 minutes. From raw data to live monitoring.
# MAGIC
# MAGIC ```
# MAGIC  Features ──▶ Training ──▶ Champion/Challenger ──▶ Production Scoring
# MAGIC                                                         │
# MAGIC     Model Landscape ◀── Inference Tables ◀──────────────┘
# MAGIC     (drift, performance, quality, incidents)
# MAGIC         │
# MAGIC         ▼
# MAGIC     "Drift detected on day 1. F1 dropped. Here's why."
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./_resources/00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the Fraud Detection Dataset
# MAGIC
# MAGIC 100k e-commerce transactions with a realistic fraud signal: higher amounts,
# MAGIC lower device trust, far from home, and new accounts all increase fraud probability.
# MAGIC About 5% of transactions are fraudulent — a realistic class imbalance.

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
n = 100_000
start = datetime(2025, 7, 1)

amount = rng.lognormal(5, 1.5, n)
device_trust = rng.beta(8, 3, n)
distance_km = rng.exponential(40, n)
velocity_24h = rng.exponential(3, n)
account_age = rng.integers(10, 1500, n).astype(float)

# Fraud probability is a function of the features (not random)
logit = (
    1.5 * (np.log(amount) - 5) / 1.5       # high amounts → more fraud
    - 2.0 * (device_trust - 0.7) / 0.15     # low trust → more fraud
    + 1.0 * (distance_km - 40) / 40          # far from home → more fraud
    + 0.8 * (velocity_24h - 3) / 3           # rapid transactions → more fraud
    - 0.5 * (account_age - 750) / 400        # new accounts → more fraud
    + rng.normal(0, 0.4, n) - 2.5            # low noise for clean signal
)
is_fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

features = pd.DataFrame({
    "transaction_id": [f"txn_{i:07d}" for i in range(n)],
    "timestamp": [start + timedelta(seconds=int(s)) for s in np.sort(rng.integers(0, 180 * 86400, n))],
    "transaction_amount": np.round(amount, 2),
    "device_trust_score": np.round(device_trust, 4),
    "distance_from_home_km": np.round(distance_km, 1),
    "velocity_24h": np.round(velocity_24h, 2),
    "account_age_days": account_age.astype(int),
    "hour_of_day": rng.integers(0, 24, n).astype(int),
    "is_weekend": rng.choice([0, 1], n, p=[5/7, 2/7]).astype(int),
    "merchant_category": rng.choice(["retail", "online", "dining", "travel", "grocery"], n, p=[.30, .25, .20, .10, .15]),
    "region": rng.choice(["na", "eu", "apac", "latam"], n, p=[.45, .30, .20, .05]),
    "is_fraud": is_fraud,
})

print(f"Fraud rate: {is_fraud.mean():.1%} ({is_fraud.sum():,} / {n:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Unity Catalog
# MAGIC
# MAGIC Three tables: features, training split (first 80% by time), test split (last 20%).

# COMMAND ----------

features_spark = spark.createDataFrame(features)
features_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("fraud_features"))

ordered = features_spark.orderBy("timestamp")
split = int(n * 0.8)
train_df = ordered.limit(split)
test_df = ordered.subtract(train_df)

train_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("fraud_train"))
test_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("fraud_test"))

print(f"✓ {table('fraud_features')} — {n:,} rows")
print(f"✓ {table('fraud_train')} — {train_df.count():,} rows")
print(f"✓ {table('fraud_test')} — {test_df.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC Data is in Unity Catalog — governed, versioned, lineage tracked.
# MAGIC
# MAGIC **Next** → `02_train_and_promote`: Train three models, validate the best, ship it.
