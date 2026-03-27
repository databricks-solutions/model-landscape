# Model Lens

Model Lens is a Databricks-native model observability product for customer workspaces.

It is built to live inside the customer environment, monitor production inference tables, and give operators a simple control plane for:

- onboarding models from warehouse tables
- computing drift, quality, and performance summaries
- tracking open incidents
- refreshing monitoring state from one managed workflow

## Product Scope

Model Lens is intended to be the in-workspace monitoring layer for teams that want Arize-style visibility without shipping inference data out of Databricks.

The product currently includes:

- a Databricks App for setup, onboarding, refresh, and readback
- a Databricks workflow for scheduled or manual refreshes
- Unity Catalog control-plane tables for configs, metrics, and incidents
- a canonical inference-log contract
- idempotent writes for refreshed model state

## How It Works

1. Deploy the app and refresh workflow with Databricks Asset Bundles.
2. Open Model Lens and initialize the control-plane schema.
3. Scan a fully qualified inference table.
4. Map the table into the monitoring contract.
5. Save the monitor and run the initial refresh.
6. Review model summaries and open incidents in the app.

The default control-plane namespace is:

- catalog: `model_observability`
- schema: `control_plane`

## Architecture

- `app.yaml` defines the Databricks App entrypoint and runtime bindings.
- `resources/app.yml` defines the Databricks App resource.
- `resources/jobs.yml` defines the refresh workflow.
- `src/ml_drift_monitor_next/app.py` implements the Model Lens UI.
- `src/ml_drift_monitor_next/services/control_plane.py` owns Databricks SQL reads and writes.
- `src/ml_drift_monitor_next/services/refresh_engine.py` and `src/ml_drift_monitor_next/services/refresh_runner.py` compute and persist monitoring output.
- `src/ml_drift_monitor_next/analytics/` contains reusable drift and performance logic.

## Monitoring Contract

Required mapped fields:

- `event_ts`
- `model_id`
- `prediction`

Optional mapped fields:

- `model_version`
- `prediction_proba`
- `label`
- `entity_id`

All other mapped fields become monitored features, categorical fields, or slice fields.

Current behavior:

- drift metrics run on numeric feature columns
- quality metrics summarize table coverage and volume
- degradation contributors are computed only when labels are available
- incidents are generated from breached drift thresholds

## Customer Deployment

Model Lens is deployed with Databricks Asset Bundles.

Required input:

- `sql_warehouse_id`: SQL warehouse used by the app and the refresh workflow

Validate:

```bash
databricks bundle validate \
  --var "sql_warehouse_id=<sql-warehouse-id>"
```

Deploy:

```bash
databricks bundle deploy -t dev \
  --var "sql_warehouse_id=<sql-warehouse-id>"
```

The refresh workflow is configured for Databricks serverless jobs. Serverless jobs must be enabled in the target workspace.

## First-Run Onboarding

After deployment:

1. Open the `model-lens` Databricks App.
2. Confirm that `SQL_WAREHOUSE_ID` is bound.
3. Click `Setup Control Plane`.
4. Enter a fully qualified source table such as `catalog.schema.inference_logs`.
5. Click `Scan`.
6. Review the detected schema and sample rows.
7. Adjust the inferred field mapping.
8. Select at least one feature column.
9. Save the monitor and run the initial refresh.
10. Review summary metrics and open incidents.

The onboarding UI is schema-aware:

- it shows Databricks column types during scan
- it keeps reserved contract fields out of the feature selector
- it warns when selected feature columns are non-numeric and will not participate in the current drift engine

## Control-Plane Tables

Model Lens manages these Delta tables:

- `monitor_configs`
- `drift_metrics`
- `quality_metrics`
- `performance_metrics`
- `incidents`

Write behavior is idempotent at the model/window level:

- one active config per `model_key`
- drift and performance rows replaced for the refreshed model window
- quality and incident rows replaced for the refreshed model snapshot

## Local Development

Run tests:

```bash
python3 -m pytest
```

Run the app locally:

```bash
PYTHONPATH=src python3 -m ml_drift_monitor_next.app
```

Run setup locally:

```bash
PYTHONPATH=src python3 scripts/setup_control_plane.py --warehouse-id <sql-warehouse-id>
```

Run refresh locally:

```bash
PYTHONPATH=src python3 scripts/refresh_control_plane.py --warehouse-id <sql-warehouse-id>
```

Build the container:

```bash
docker build -t model-lens .
docker run --rm -p 8080:8080 model-lens
```

## Product Limits

Current limits in this version:

- categorical-specific drift metrics are not implemented yet
- slice-level rollups are not exposed in the UI yet
- alert delivery integrations are not implemented yet
- incident lifecycle is still current-state oriented rather than full acknowledge/resolve workflow

## Positioning

Model Lens is meant to be a product, not a demo repo:

- deployable inside customer Databricks workspaces
- minimal infrastructure outside Databricks
- explicit monitoring contract
- one control plane instead of per-model orchestration sprawl
- customer-visible setup and onboarding path from the app itself
