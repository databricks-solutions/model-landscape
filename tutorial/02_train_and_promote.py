# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2: Train, Validate, and Promote
# MAGIC
# MAGIC Three candidates. One champion. Every experiment tracked in MLflow and
# MAGIC enriched with **MLflow Lens** for deeper comparison. The best model gets
# MAGIC registered in Unity Catalog, passes validation gates, and earns the
# MAGIC Champion alias.
# MAGIC
# MAGIC No model touches production without passing every gate.

# COMMAND ----------

# MAGIC %run ./_resources/00_setup

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
from mlflow import MlflowClient
from mlflow_lens import summary, experiment, drift, panels, cost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb

mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.autolog(disable=True)
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

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

# MAGIC %md
# MAGIC ## Train Three Candidates
# MAGIC
# MAGIC Each run is enriched with **MLflow Lens**: workspace context, structured
# MAGIC summaries, and cost attribution. These enrichments power Model Landscape's
# MAGIC experiment comparison page.

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

run_ids = {}
models = {}

for name, model in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        # --- Standard MLflow ---
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {"f1": f1_score(y_test, y_pred), "precision": precision_score(y_test, y_pred),
                   "recall": recall_score(y_test, y_pred), "roc_auc": roc_auc_score(y_test, y_proba)}
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.get_params())
        mlflow.set_tags({"task": "fraud_detection", "dataset_version": "v1"})
        mlflow.log_input(train_dataset, context="training")
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_test.head(5))

        # --- MLflow Lens enrichment ---
        experiment.auto_log_context()
        summary.log(task="fraud_detection", primary_metric="f1", score=metrics["f1"],
                    dataset="fraud_train_v1", notes=f"{name} candidate")
        cost.log_cost_context()

        run_ids[name] = run.info.run_id
        models[name] = model
        print(f"  {name}: F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Lens Panels to the Best Model
# MAGIC
# MAGIC Confusion matrix, ROC curve, and feature importance — structured data that
# MAGIC Model Landscape renders side-by-side across runs.

# COMMAND ----------

# Find best
run_records = []
for name, rid in run_ids.items():
    r = mlflow.get_run(rid)
    run_records.append({"run_id": rid, "model_name": name, **{k: round(v, 4) for k, v in r.data.metrics.items()}})
runs_df = pd.DataFrame(run_records).sort_values("f1", ascending=False).reset_index(drop=True)

best = runs_df.iloc[0]
best_model = models[best["model_name"]]
display(runs_df)

# COMMAND ----------

with mlflow.start_run(run_id=best["run_id"]):
    # Confusion matrix
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_data = [{"actual": int(i), "predicted": int(j), "count": int(cm[i, j])}
               for i in range(cm.shape[0]) for j in range(cm.shape[1])]
    panels.log_panel("confusion_matrix", cm_data, labels=["legit", "fraud"])

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    roc_data = [{"fpr": float(f), "tpr": float(t), "threshold": float(th)}
                for f, t, th in zip(fpr[::10], tpr[::10], thresholds[::10])]
    panels.log_panel("roc_curve", roc_data)

    # Feature importance (tree models only)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": best_model.feature_importances_})
        panels.log_panel("feature_importance", fi.sort_values("importance", ascending=False))
        display(fi.sort_values("importance", ascending=False))

    # Drift snapshot for future comparison (baseline — pass features as reference)
    drift.log_drift(reference_run_id=None, features=X_train,
                    reference_features=X_train,
                    predictions=best_model.predict_proba(X_train)[:, 1])

    print("✓ Panels + drift snapshot logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register in Unity Catalog

# COMMAND ----------

registered = model_name("fraud_detector")
mv = mlflow.register_model(f"runs:/{best['run_id']}/model", registered)

client.update_registered_model(
    name=registered,
    description="Binary classifier for e-commerce fraud detection. "
    "Features: transaction amount, device trust, distance from home, velocity, account age.",
)
client.update_model_version(
    name=registered, version=mv.version,
    description=f"Trained with {best['model_name']}. F1={best['f1']:.4f} on time-split test set.",
)

print(f"✓ Registered {registered} v{mv.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation Gates
# MAGIC
# MAGIC Three checks before promotion. All results tagged on the model version
# MAGIC for the audit trail.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gate 1: Documentation exists

# COMMAND ----------

model_info = client.get_registered_model(registered)
version_info = client.get_model_version(registered, mv.version)
assert len(model_info.description or "") >= 40, "Model needs a description"
assert len(version_info.description or "") >= 20, "Version needs a description"
print("✓ Documentation gate passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gate 2: Performance above threshold

# COMMAND ----------

challenger_model = mlflow.sklearn.load_model(f"models:/{registered}/{mv.version}")
y_pred_val = challenger_model.predict(X_test)
val_metrics = {"f1": f1_score(y_test, y_pred_val), "precision": precision_score(y_test, y_pred_val),
               "recall": recall_score(y_test, y_pred_val)}

F1_THRESHOLD = 0.50
assert val_metrics["f1"] >= F1_THRESHOLD, f"F1 ({val_metrics['f1']:.4f}) below {F1_THRESHOLD}"
print(f"✓ Performance gate passed — F1={val_metrics['f1']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gate 3: No regression vs existing Champion

# COMMAND ----------

try:
    champion_model = mlflow.sklearn.load_model(f"models:/{registered}@Champion")
    champion_f1 = f1_score(y_test, champion_model.predict(X_test))
    assert val_metrics["f1"] >= champion_f1 - 0.01, "Challenger regressed"
    print(f"✓ No-regression gate — Challenger ({val_metrics['f1']:.4f}) >= Champion ({champion_f1:.4f})")
except mlflow.exceptions.MlflowException:
    print("  First deployment — no existing Champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote to Champion

# COMMAND ----------

for key, value in val_metrics.items():
    client.set_model_version_tag(registered, mv.version, f"validation_{key}", f"{value:.4f}")
client.set_model_version_tag(registered, mv.version, "validation_passed", "true")

client.set_registered_model_alias(registered, "Champion", mv.version)
print(f"✓ {registered} v{mv.version} promoted to Champion")

# COMMAND ----------

# MAGIC %md
# MAGIC The model is trained, enriched with Lens, validated, and promoted.
# MAGIC Every check is tagged. Rolling back is one line.
# MAGIC
# MAGIC **Next** → `03_production_inference`: Ship it. Watch what happens over 60 days.
