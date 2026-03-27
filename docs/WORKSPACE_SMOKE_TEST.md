# Workspace Smoke Test

This guide is the pre-client validation path for Model Lens in your own Databricks workspace.

Use it before sharing the product with a customer.

## Goal

By the end of this test, you should have verified:

- deployment works in your workspace
- the app can initialize the control plane
- a monitor can be onboarded from a real table
- the refresh workflow writes metrics and incidents
- the Lakebase projection is populated
- the UI readback matches the persisted warehouse and Lakebase state

## Test Strategy

Run the test in this order:

1. local validation
2. create a deterministic scratch dataset
3. deploy the app and workflow
4. onboard one monitor
5. verify persisted warehouse state
6. verify app readback
7. verify manual workflow refresh
8. exercise a few failure paths

## Prerequisites

You need:

- Databricks CLI auth configured for the target workspace
- a SQL warehouse you can use
- serverless jobs enabled in the target workspace
- privileges to create tables in a test catalog/schema
- privileges to deploy Databricks Asset Bundles and open Databricks Apps

If you are testing the Lakebase path, you also need:

- a Lakebase instance
- a Lakebase database
- a Lakebase DB user for the refresh workflow

## Step 1: Local Validation

Run this from the repo root:

```bash
cd /Users/volo.vragov/Desktop/work/model-lens
python3 -m pytest
python3 scripts/model_lens_setup.py --help
python3 scripts/model_lens_refresh.py --help
databricks bundle validate -t warehouse_only --var "sql_warehouse_id=<sql-warehouse-id>"
```

Expected result:

- tests pass
- both wrapper scripts parse
- bundle validation passes

## Step 2: Create Scratch Data

Open a Databricks SQL editor and run [`examples/scratch_dataset.sql`](/Users/volo.vragov/Desktop/work/model-lens/examples/scratch_dataset.sql).

Before running it:

- update the catalog if `main` is not writable in your workspace

The script creates:

- `main.model_lens_demo.inference_logs`
- `main.model_lens_demo.labels`

Expected dataset properties:

- 840 inference rows
- 840 label rows
- dates from `2026-01-01` to `2026-01-21`
- strong numeric drift after day 7

Verify that:

```sql
SELECT COUNT(*) AS inference_rows FROM main.model_lens_demo.inference_logs;
SELECT COUNT(*) AS label_rows FROM main.model_lens_demo.labels;
SELECT MIN(DATE(event_ts)) AS min_date, MAX(DATE(event_ts)) AS max_date
FROM main.model_lens_demo.inference_logs;
```

Expected:

- `inference_rows = 840`
- `label_rows = 840`
- `min_date = 2026-01-01`
- `max_date = 2026-01-21`

## Step 3: Deploy Model Lens

Deploy into your test target:

```bash
databricks bundle deploy \
  -t warehouse_only \
  --var "sql_warehouse_id=<sql-warehouse-id>"

databricks apps deploy model-lens \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-lens/warehouse_only/files
```

Expected result:

- bundle deploy succeeds
- the Databricks app `model-lens` exists
- the workflow `model-lens-refresh` exists

## Step 4: Open The App

Open the `model-lens` Databricks App.

Check immediately:

- the app title reads `Model Lens`
- `SQL_WAREHOUSE_ID` is populated
- `USE_LAKEBASE_READ_MODEL` is `false`
- there is no pre-rename product naming anywhere

## Step 5: Initialize The Control Plane

In the app, click `Setup Control Plane`.

Expected result:

- success banner
- control-plane tables are created under `model_observability.control_plane`

Verify in SQL:

```sql
SHOW TABLES IN model_observability.control_plane;
```

Expected tables:

- `monitor_configs`
- `drift_metrics`
- `quality_metrics`
- `performance_metrics`
- `incidents`

If you are testing the Lakebase target instead of `warehouse_only`, also verify:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'model_lens_ui'
ORDER BY table_name;
```

Expected:

- `monitor_inventory`
- `monitor_summary`
- `open_incidents`

## Step 6: Onboard The Scratch Monitor

In the app:

1. Set the source table to:
   - `main.model_lens_demo.inference_logs`
2. Click `Scan`.
3. Review the schema and sample rows.

Use these field mappings:

- `Display Name`: `Fraud Model Demo`
- `Model Key`: `fraud_model_demo`
- `Timestamp Column`: `event_ts`
- `Model ID Column`: `model_id`
- `Prediction Column`: `prediction`
- `Model Version Column`: `(none)`
- `Prediction Score Column`: `(none)`
- `Entity ID Column`: `entity_id`
- `Label Column In Source`: `(none)`
- `External Labels Table`: `main.model_lens_demo.labels`
- `External Labels Join Column`: `entity_id`
- `External Label Column`: `label`
- `Problem Type`: `classification`
- `Baseline Days`: `7`

For the happy-path feature selection, choose:

- `amount`
- `velocity_7d`
- `device_score`

Optional categorical/slice test after the happy path:

- add `region`
- add `merchant_segment`

Save the monitor:

- click `Save Monitor And Run Initial Refresh`

Expected result:

- success banner
- refresh counts are non-zero
- the monitor appears in the summary table

## Step 7: Verify Persisted State In SQL

Run these queries:

```sql
SELECT model_key, display_name, source_table, feature_columns, categorical_columns, slice_columns, status
FROM model_observability.control_plane.monitor_configs
WHERE model_key = 'fraud_model_demo';
```

Expected:

- one row
- `status = 'active'`

```sql
SELECT model_key, total_rows, min_date, max_date
FROM model_observability.control_plane.quality_metrics
WHERE model_key = 'fraud_model_demo';
```

Expected:

- one row
- `total_rows = 840`
- `min_date = 2026-01-01`
- `max_date = 2026-01-21`

```sql
SELECT model_key, feature_name, metric_name, COUNT(*) AS row_count
FROM model_observability.control_plane.drift_metrics
WHERE model_key = 'fraud_model_demo'
GROUP BY 1,2,3
ORDER BY feature_name, metric_name;
```

Expected:

- rows exist for at least:
  - `amount`
  - `velocity_7d`
  - `device_score`

```sql
SELECT model_key, COUNT(*) AS perf_rows
FROM model_observability.control_plane.performance_metrics
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- at least one row

```sql
SELECT model_key, feature_name, metric_name, severity, status
FROM model_observability.control_plane.incidents
WHERE model_key = 'fraud_model_demo'
ORDER BY observed_at DESC;
```

Expected:

- zero or more rows
- with this dataset, at least one drift incident is likely

If you are testing the Lakebase target instead of `warehouse_only`, also verify:

```sql
SELECT * FROM model_lens_ui.monitor_inventory ORDER BY display_name;
SELECT * FROM model_lens_ui.monitor_summary ORDER BY display_name;
SELECT * FROM model_lens_ui.open_incidents ORDER BY observed_at DESC;
```

Expected:

- one monitor inventory row for `Fraud Model Demo`
- one summary row for `fraud_model_demo`
- zero or more open incident rows

## Step 8: Verify App Readback

Back in the app, verify:

- `Fraud Model Demo` appears in `Monitors`
- `total_rows` is shown
- `latest_data_date` is shown
- `last_refresh_at` is shown
- `open_incident_count` is shown
- `Open Incidents` renders without a callback failure

## Step 9: Verify Manual Refresh Paths

In the app:

1. select `Fraud Model Demo` in `Refresh One Monitor`
2. click `Refresh Selected Monitor`
3. click `Refresh All Monitors`

Expected:

- both actions succeed
- no red error banner
- counts are non-zero

## Step 10: Verify Workflow Outside The App

Open the Databricks workflow `model-lens-refresh` and run it manually once.

Then rerun:

```sql
SELECT model_key, MAX(computed_at) AS last_quality_refresh
FROM model_observability.control_plane.quality_metrics
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- timestamp updates after the workflow run

## Step 11: Exercise Failure Paths

Test these before client handoff:

### Missing feature selection

- clear all feature columns
- try to save

Expected:

- warning banner
- config not saved

### Targeted refresh without selection

- clear the selected monitor in the refresh dropdown
- click `Refresh Selected Monitor`

Expected:

- warning banner

### Non-numeric feature warning

Edit the monitor and include:

- `region`
- `merchant_segment`

Expected:

- save succeeds
- warning explains those fields are stored in the contract but skipped by the current drift engine

### Short-history dataset

Create a second scratch table with fewer than 8 days of data and onboard it.

Expected:

- save succeeds
- refresh warns that there is no comparable baseline/current window yet

## Step 12: Pre-Client Acceptance Checklist

Do not send to a client until all of these are true:

- `python3 -m pytest` passes
- bundle validate passes
- bundle deploy succeeds in your workspace
- the app opens and is branded correctly
- control-plane setup succeeds
- one monitor can be onboarded end to end
- warehouse tables contain the expected rows
- if you tested the Lakebase target, Lakebase tables contain the expected projected rows
- the app summary matches the persisted state
- the workflow refresh path works outside the app
- there are no old names left in the product surface
