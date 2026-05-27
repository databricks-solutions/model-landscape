# Model Landscape MLOps Tutorial

End-to-end MLOps pipeline in two notebooks: generate data, train and ship a
Champion model, watch it drift through 60 days of production, and close the
loop with a retrained v2.

## Quick start

```bash
# Deploy the app + the tutorial job
databricks bundle deploy -t warehouse_only \
  --var "sql_warehouse_id=<your-warehouse-id>"

# Run the tutorial end-to-end
databricks bundle run tutorial_mlops
```

## Notebooks

| # | Notebook | What happens |
|---|----------|--------------|
| 01 | `01_train_and_ship` | Build a 100k-row fraud dataset, train 3 candidates, log mlflow-lens panels (ROC, confusion matrix, PR curve, classification report, feature importances, learning curve), register the best as Champion. |
| 02 | `02_observe_and_recover` | Score with Champion, fast-forward 60 days of drifted inference (+ a second model for fleet monitoring), create monitors, bootstrap-refresh, investigate incidents, retrain v2, promote, redeploy. |

The mlflow-lens panel quick functions in notebook 1 log **both** the
structured JSON payload (`lens/panels/{type}.json`, consumed by the
warehouse app) **and** an interactive Plotly figure
(`lens/panels/{type}.html`, viewable directly in the MLflow UI).

## Data generator

`_resources/data_generator.py` produces two synthetic scenarios with
calibrated drift patterns:

- **Fraud detection** (classification): gradual then sudden drift in
  transaction patterns, holiday season surge, data pipeline null spike.
- **Predictive maintenance** (regression): seasonal temperature shifts,
  equipment aging, sensor outages.

## Prerequisites

- Databricks workspace with Unity Catalog
- Serverless compute enabled
- A running SQL warehouse
- Model Landscape app deployed (`databricks bundle deploy`)
