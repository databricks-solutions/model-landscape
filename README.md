# ML Drift Monitor Next

Databricks-native model observability control plane for customer workspaces.

This repo is a ground-up rework of the original prototype. The product goal is:

- deploy cleanly into a customer Databricks workspace
- keep workspace state declarative and bundle-managed
- use one control-plane job instead of per-model job sprawl
- enforce one canonical inference log contract
- keep metrics and incidents idempotent
- stay portable across Databricks clouds

## What Changed

The original prototype mixed app concerns, job provisioning, workspace discovery, and metrics computation in the same runtime path. This repo separates those responsibilities:

- `databricks.yml` and `resources/` define the app and workflow as bundle-managed resources
- `src/ml_drift_monitor_next/domain/` defines the customer-facing contract
- `src/ml_drift_monitor_next/services/` owns naming, onboarding payloads, schemas, and incident lifecycle
- `src/ml_drift_monitor_next/workflows/refresh_job.py` is the single refresh entry point
- `src/ml_drift_monitor_next/analytics/` contains reusable drift and performance logic

## Architecture

1. Customer maps an inference table into the canonical contract.
2. App writes one `monitor_configs` row per model.
3. One bundle-managed workflow refreshes active models.
4. Refresh writes global metrics tables into one Unity Catalog schema.
5. Incident generation deduplicates on business keys.
6. Dash app reads control-plane tables and lets operators trigger refreshes.

## Canonical Inference Contract

Required columns:

- `event_ts`
- `model_id`
- `prediction`

Recommended columns:

- `model_version`
- `prediction_proba`
- `label`
- `entity_id`

All remaining mapped columns are monitored features or slice dimensions.

## Local Development

Run tests:

```bash
python3 -m pytest
```

Run the app locally:

```bash
PYTHONPATH=src python3 -m ml_drift_monitor_next.app
```

Run in a container:

```bash
docker build -t ml-drift-monitor-next .
docker run --rm -p 8080:8080 ml-drift-monitor-next
```

## Databricks Deployment

The intended deployment path is Databricks Asset Bundles.

```bash
databricks bundle validate
databricks bundle deploy -t dev
```

Then bind app resources in Databricks Apps:

- SQL warehouse
- refresh job
- optional Genie space

`app.yaml` uses `valueFrom` instead of hardcoded IDs so deployments stay portable.
