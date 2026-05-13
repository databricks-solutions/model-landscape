# Model Landscape MLOps Tutorial

End-to-end MLOps pipeline that generates realistic data, trains models,
produces drifted production inference, sets up live monitoring, and closes
the loop with retraining.

## Quick Start

```bash
# Deploy everything (app + tutorial job + refresh job)
databricks bundle deploy -t warehouse_only \
  --var "sql_warehouse_id=<your-warehouse-id>"

# Run the tutorial pipeline
databricks bundle run tutorial_mlops
```

## Notebooks

| # | Notebook | What happens |
|---|----------|-------------|
| 01 | `01_build_features` | Generate 100k fraud transactions, save to Unity Catalog |
| 02 | `02_train_and_promote` | Train 3 models, enrich with mlflow-lens, validate, promote Champion |
| 03 | `03_production_inference` | Score with Champion, generate 60 days of drifted inference + a second model (predictive maintenance) |
| 04 | `04_monitor` | Create monitors, run bootstrap refresh, verify drift and incidents |
| 05 | `05_close_the_loop` | Detect drift, retrain on recent data, promote v2, verify recovery |

## Data Generator

`_resources/data_generator.py` generates two synthetic scenarios with
calibrated drift patterns:

- **Fraud detection** (classification): gradual then sudden drift in
  transaction patterns, holiday season surge, data pipeline null spike
- **Predictive maintenance** (regression): seasonal temperature shifts,
  equipment aging, sensor outages

## Prerequisites

- Databricks workspace with Unity Catalog
- Serverless compute enabled
- A running SQL warehouse
- Model Landscape app deployed (`databricks bundle deploy`)
