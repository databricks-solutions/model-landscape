# Databricks notebook source
# MAGIC %md
# MAGIC # It's 2am. Your Phone Buzzes.
# MAGIC
# MAGIC The fraud model you deployed three months ago just approved $2.3M in
# MAGIC fraudulent transactions. A holiday shopping surge shifted the transaction mix —
# MAGIC more travel, more international, higher amounts. Your model had never seen
# MAGIC data like this. It didn't fail dramatically. It failed *quietly*.
# MAGIC
# MAGIC **Nobody noticed for 14 days.**
# MAGIC
# MAGIC This tutorial builds the system that catches this on day 1. **Two notebooks.**
# MAGIC
# MAGIC ```
# MAGIC  01_train_and_ship       ←  YOU ARE HERE
# MAGIC  └─ features → 3 candidates → mlflow-lens panels → registered Champion
# MAGIC
# MAGIC  02_observe_and_recover
# MAGIC  └─ score → 60 days of drift → monitors → incidents → retrain v2
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./_resources/00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the fraud detection dataset
# MAGIC
# MAGIC 100k e-commerce transactions with a realistic fraud signal: higher amounts,
# MAGIC lower device trust, far from home, and new accounts all increase fraud
# MAGIC probability. About 5% are fraudulent — a realistic class imbalance.
# MAGIC We split 80/20 by time so the test set is genuinely "the future".

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

logit = (
    1.5 * (np.log(amount) - 5) / 1.5
    - 2.0 * (device_trust - 0.7) / 0.15
    + 1.0 * (distance_km - 40) / 40
    + 0.8 * (velocity_24h - 3) / 3
    - 0.5 * (account_age - 750) / 400
    + rng.normal(0, 0.4, n) - 2.5
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

features_spark = spark.createDataFrame(features).orderBy("timestamp")
features_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("fraud_features"))

split = int(n * 0.8)
train_df = features_spark.limit(split)
test_df = features_spark.subtract(train_df)

train_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("fraud_train"))
test_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("fraud_test"))

print(f"✓ {table('fraud_train'):<60s} {train_df.count():>7,} rows")
print(f"✓ {table('fraud_test'):<60s} {test_df.count():>7,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train three candidates
# MAGIC
# MAGIC Each run is enriched with **mlflow-lens**: workspace context, structured
# MAGIC summary, cost attribution. These tags + artifacts power Model Landscape's
# MAGIC experiment comparison and the warehouse app.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow_lens import cost, experiment, summary
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb

mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.autolog(disable=True)
client = MlflowClient()

train_pdf = spark.table(table("fraud_train")).toPandas()
test_pdf = spark.table(table("fraud_test")).toPandas()

FEATURE_COLS = [
    "transaction_amount", "device_trust_score", "distance_from_home_km",
    "velocity_24h", "hour_of_day", "is_weekend", "account_age_days",
]
X_train, y_train = train_pdf[FEATURE_COLS], train_pdf["is_fraud"]
X_test, y_test = test_pdf[FEATURE_COLS], test_pdf["is_fraud"]
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Fraud rate: {y_train.mean():.1%}")

# COMMAND ----------

train_dataset = mlflow.data.from_pandas(train_pdf, source=table("fraud_train"), name="fraud_train")

candidates = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42),
    "xgboost": xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42, eval_metric="logloss",
    ),
}

run_ids, models = {}, {}
for name, model in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.get_params())
        mlflow.set_tags({"task": "fraud_detection", "dataset_version": "v1"})
        mlflow.log_input(train_dataset, context="training")
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_test.head(5))

        # mlflow-lens enrichment
        experiment.auto_log_context()
        summary.log(task="fraud_detection", primary_metric="f1", score=metrics["f1"],
                    dataset="fraud_train_v1", notes=f"{name} candidate")
        cost.log_cost_context()

        run_ids[name] = run.info.run_id
        models[name] = model
        print(f"  {name:<22s}  F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick the best and attach diagnostic panels
# MAGIC
# MAGIC The mlflow-lens panel quick functions log both the raw data
# MAGIC (`lens/panels/{type}.json` — the warehouse app reads these) and an
# MAGIC interactive Plotly figure (`lens/panels/{type}.html` — opens directly in
# MAGIC the MLflow UI). One line per panel.

# COMMAND ----------

run_records = [
    {"run_id": rid, "model_name": name, **{k: round(v, 4) for k, v in mlflow.get_run(rid).data.metrics.items()}}
    for name, rid in run_ids.items()
]
runs_df = pd.DataFrame(run_records).sort_values("f1", ascending=False).reset_index(drop=True)
best = runs_df.iloc[0]
best_model = models[best["model_name"]]
display(runs_df)

# COMMAND ----------

from mlflow_lens import drift
from mlflow_lens.classifier import (
    classification_report,
    confusion_matrix,
    precision_recall,
    roc_auc,
)
from mlflow_lens.model_selection import feature_importances, learning_curve

with mlflow.start_run(run_id=best["run_id"]):
    roc_auc(best_model, X_test, y_test, log=True)
    confusion_matrix(best_model, X_test, y_test, labels=[0, 1], log=True)
    precision_recall(best_model, X_test, y_test, log=True)
    classification_report(best_model, X_test, y_test, log=True)

    if hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_"):
        feature_importances(best_model, FEATURE_COLS, log=True)

    learning_curve(type(best_model)(**best_model.get_params()), X_train, y_train, cv=3, log=True)

    # Drift baseline snapshot for future comparison
    drift.log_drift(
        reference_run_id=None,
        features=X_train,
        reference_features=X_train,
        predictions=best_model.predict_proba(X_train)[:, 1],
    )
    print("✓ ROC, confusion matrix, PR curve, classification report,")
    print("  feature importances, learning curve, and drift snapshot logged")

# COMMAND ----------

# MAGIC %md
# MAGIC Open the best run in MLflow → Artifacts → `lens/panels/`. Each panel has
# MAGIC a `.json` data file and a `.html` interactive Plotly figure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register in Unity Catalog

# COMMAND ----------

registered = model_name("fraud_detector")
mv = mlflow.register_model(f"runs:/{best['run_id']}/model", registered)

client.update_registered_model(
    name=registered,
    description=(
        "Binary classifier for e-commerce fraud detection. "
        "Features: transaction amount, device trust, distance from home, velocity, account age."
    ),
)
client.update_model_version(
    name=registered, version=mv.version,
    description=f"Trained with {best['model_name']}. F1={best['f1']:.4f} on time-split test set.",
)
print(f"✓ Registered {registered} v{mv.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation gates → Promote
# MAGIC
# MAGIC Three checks before Champion. Every result tagged on the model version
# MAGIC so the audit trail is durable.

# COMMAND ----------

# Gate 1: documentation
model_info = client.get_registered_model(registered)
version_info = client.get_model_version(registered, mv.version)
assert len(model_info.description or "") >= 40, "Model needs a description"
assert len(version_info.description or "") >= 20, "Version needs a description"
print("✓ Gate 1 (documentation)")

# Gate 2: performance threshold
challenger = mlflow.sklearn.load_model(f"models:/{registered}/{mv.version}")
val_pred = challenger.predict(X_test)
val_metrics = {
    "f1": f1_score(y_test, val_pred),
    "precision": precision_score(y_test, val_pred),
    "recall": recall_score(y_test, val_pred),
}
F1_THRESHOLD = 0.50
assert val_metrics["f1"] >= F1_THRESHOLD, f"F1 ({val_metrics['f1']:.4f}) below {F1_THRESHOLD}"
print(f"✓ Gate 2 (performance)        F1={val_metrics['f1']:.4f}")

# Gate 3: no regression vs current Champion
try:
    champion = mlflow.sklearn.load_model(f"models:/{registered}@Champion")
    champion_f1 = f1_score(y_test, champion.predict(X_test))
    assert val_metrics["f1"] >= champion_f1 - 0.01, "Challenger regressed"
    print(f"✓ Gate 3 (no regression)      challenger {val_metrics['f1']:.4f} ≥ champion {champion_f1:.4f}")
except mlflow.exceptions.MlflowException:
    print("✓ Gate 3 (no regression)      first deployment — no existing Champion")

# Tag + promote
for key, value in val_metrics.items():
    client.set_model_version_tag(registered, mv.version, f"validation_{key}", f"{value:.4f}")
client.set_model_version_tag(registered, mv.version, "validation_passed", "true")
client.set_registered_model_alias(registered, "Champion", mv.version)
print(f"\n✓ {registered} v{mv.version} promoted to Champion")

# COMMAND ----------

# MAGIC %md
# MAGIC Model is trained, enriched, validated, promoted. Every check is tagged.
# MAGIC Rollback is one line.
# MAGIC
# MAGIC **Next** → `02_observe_and_recover`: ship it, watch 60 days of drift, retrain.
