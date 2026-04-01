# Workspace Smoke Test

This guide is the pre-client validation path for Model Lens in your own Databricks workspace.

Use it before sharing the product with a customer.

## Goal

By the end of this test, you should have verified:

- deployment works in your workspace
- the app can initialize the control plane
- a monitor can be onboarded from a real table
- the refresh workflow writes metrics and incidents
- if you test the Lakebase target, the Lakebase projection is populated
- the UI readback matches the persisted state for the mode you deployed

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
- a chosen Unity Catalog catalog/schema for the Model Lens control plane

If you are testing the Lakebase path, you also need:

- a Lakebase instance
- a Lakebase database
- a Lakebase DB user for the refresh workflow

The Databricks App service principal also needs:

- `CAN_USE` on the SQL warehouse
- read access to the source tables
- read/write access to the control-plane namespace

Treat the warehouse grant as something you verify after deployment, not as a one-time guarantee from the bundle binding. If the app is restarted, recreated, or source-deployed separately, recheck that the app service principal still has `CAN_USE` on the SQL warehouse.

Identity checklist for this smoke test:

- deployer or platform operator:
  can deploy the bundle/app and has approved the control-plane namespace
- app service principal:
  `CAN_USE` on the SQL warehouse; `CAN MANAGE RUN` on the refresh workflow; source data `USE CATALOG`, `USE SCHEMA`, `SELECT`; control plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`
- app service principal, if Setup should create missing objects:
  `CREATE TABLE` in the control-plane schema; `CREATE SCHEMA` if the schema is missing; `CREATE CATALOG` only if you plan to use the toggle
- refresh workflow identity:
  the same warehouse, source-data, and control-plane permissions as the app
- optional MLflow-assisted onboarding:
  read access to the target experiment and/or registered model metadata
- optional Lakebase read model:
  permission to resolve the Lakebase instance and connect to the target database; workflow sync also needs write access to the Lakebase schema

If you manually deleted the app in this workspace before redeploying, clear stale bundle state first:

```bash
databricks workspace delete /Workspace/Users/<your-email>/.bundle/model-lens --recursive
```

The commands below assume the default bundle variable `app_name=model-lens`.
If you override `app_name`, replace the app name in every `databricks apps ...` command and expect the workflow name to become `<app-name>-refresh`.

## Step 1: Local Validation

Run this from the repo root:

```bash
cd /Users/volo.vragov/Desktop/work/model-lens
python3 -m pytest
python3 scripts/model_lens_setup.py --help
python3 scripts/model_lens_refresh.py --help
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
databricks bundle validate \
  -t warehouse_only \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>"
```

Expected result:

- tests pass
- both wrapper scripts parse
- wheel build succeeds
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
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>"

databricks apps start model-lens

databricks apps deploy model-lens \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-lens/warehouse_only/files

databricks apps get model-lens
databricks apps get model-lens -o json
```

Expected result:

- bundle deploy succeeds
- the Databricks app `model-lens` exists
- `databricks apps start model-lens` reaches `ACTIVE`
- the workflow `<app-name>-refresh` exists, where `<app-name>` is `model-lens` unless you overrode `app_name`
- the app service principal shown in `databricks apps get model-lens -o json` still has `CAN_USE` on the SQL warehouse
- the same app identity has `CAN MANAGE RUN` on the refresh workflow

If you are reusing an existing app and cannot manage its SQL warehouse app resource, do not use this smoke-test deploy path. Use the generated manual existing-app path in [Manual Setup With An Existing Databricks App](/Users/volo.vragov/Desktop/work/model-lens/docs/MANUAL_EXISTING_APP_SETUP.md) instead.

## Step 4: Open The App

Open the `model-lens` Databricks App.

Check immediately:

- the app title reads `Model Lens`
- `SQL_WAREHOUSE_ID` is populated
- `REFRESH_JOB_ID` is populated or `REFRESH_JOB_NAME` matches the deployed workflow name
- `USE_LAKEBASE_READ_MODEL` is `false`
- `Control Plane Catalog` and `Control Plane Schema` show the namespace you want to use
- if Lakebase exists in the workspace and is visible to the app identity, an informational banner recommends Lakebase
- optional `Lakebase Instance Name` / `Lakebase Database Name` inputs are visible for session-level acceleration
- there is no pre-rename product naming anywhere

## Step 5: Initialize The Control Plane

In the app `Setup` step, click `Setup Control Plane`.

The wizard should keep `Continue to Discover` disabled until setup succeeds for the current control-plane namespace. If setup fails, fix the underlying issue and click `Setup Control Plane` again to retry.

Recommended:

- pre-create the target namespace outside the app
- leave `Create catalog if missing` off unless you are testing with an admin identity
- keep the app namespace fields aligned with the bundle vars so in-app refreshes and workflow refreshes hit the same control plane
- if you want to test fast app reads, enter `Lakebase Instance Name` and `Lakebase Database Name` before clicking setup

Expected result:

- success banner
- control-plane tables are created under your chosen `<control-plane-catalog>.<control-plane-schema>`

Verify in SQL:

```sql
SHOW TABLES IN <control-plane-catalog>.<control-plane-schema>;
```

Expected tables:

- `monitor_configs`
- `drift_metrics`
- `quality_metrics`
- `quality_history`
- `performance_metrics`
- `incidents`
- `incident_history`
- `refresh_runs`
- `comparison_windows`

If you are testing Lakebase-backed reads, also verify:

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

1. Continue to the `Discover` step and set the source table to:
   - `main.model_lens_demo.inference_logs`
2. Optional draft inputs:
   - `Optional Labels Table`: `main.model_lens_demo.labels`
   - `Optional MLflow Experiment`: leave blank for this smoke test unless you have one ready
   - `Optional Registered Model`: leave blank for this smoke test unless you have one ready
3. Click `Discover`.
   With the labels table filled in, verify the app now shows:
   - labels columns: `entity_id`, `label_timestamp`, `label`
   - sample label rows
   - inferred `Join Column=entity_id`, `Label Column=label`, `Order Column=label_timestamp`
   - join validation with matched rows, unmatched rows, and duplicate label-key counts
   If your tables use ISO timestamps stored as strings or a shared string key such as `unique_hash`, discovery should still detect those correctly without requiring manual overrides.
   If you ever see `matched=0`, stop there and correct the join column before you continue; the current build surfaces that as a red warning instead of a quiet table row.
4. Review the schema and sample rows, then continue to the `Confirm` step.
5. Confirm the inferred contract:
   - `Display Name`: `Fraud Model Demo`
   - `Model Key`: `fraud_model_demo`
   - `Problem Type`: `classification`
   - `Baseline Policy`: `Rolling`
   - `Baseline Days`: `7`
   Keep the default rolling baseline for this smoke test. Fixed baselines are supported, but the default rolling window is enough to exercise the full pipeline.
6. Open `Advanced mappings and overrides` and confirm:
   - `Timestamp Column`: `event_ts`
   - `Model ID Column`: `model_id`
   - `Monitored Model ID Value`: `fraud_model_v1`
   - `Prediction Column`: `prediction`
   - `Model Version Column`: `(none)`
   - `Prediction Score Column`: `(none)`
   - `Entity ID Column`: `entity_id`
   - `Label Column In Source`: `(none)`
   - `External Labels Join Column`: `entity_id`
   - `External Label Column`: `label`
   - `External Labels Order Column`: `label_timestamp`
7. In the same `Advanced` section, confirm feature selection:
   - `amount`
   - `velocity_7d`
   - `device_score`
   For wider real-world tables, expect all numeric features to remain selected by default unless you intentionally narrow them.
8. Optional slice check after the happy path:
   - confirm `region`
   - confirm `merchant_segment` if you added it to the dataset
9. Continue to the `Activate` step, then save the monitor:

- click `Save Monitor And Trigger Refresh`

Expected result:

- success banner
- the success banner says the monitor was saved and the refresh job was triggered asynchronously
- if the app cannot resolve the workflow from `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`, it warns that the monitor was saved but the workflow must be run manually
- the monitor appears on the overview page
- after the workflow finishes, Drift and Performance should already show historical windows rather than a single snapshot
- after the workflow finishes, Data Quality should show window-history charts instead of only the latest summary row
- with the scratch dataset and `Baseline Days = 7`, you should have 8 daily comparison windows immediately
- the remaining historical hardening work is tracked in [Historical Backfill Plan](/Users/volo.vragov/Desktop/work/model-lens/docs/HISTORICAL_BACKFILL_PLAN.md)

## Step 7: Verify Persisted State In SQL

Run these queries:

```sql
SELECT model_key, display_name, source_table, feature_columns, categorical_columns, slice_columns, status
FROM <control-plane-catalog>.<control-plane-schema>.monitor_configs
WHERE model_key = 'fraud_model_demo';
```

Expected:

- one row
- `status = 'active'`

```sql
SELECT model_key, total_rows, min_date, max_date
FROM <control-plane-catalog>.<control-plane-schema>.quality_metrics
WHERE model_key = 'fraud_model_demo';
```

Expected:

- one row
- `total_rows = 840`
- `min_date = 2026-01-01`
- `max_date = 2026-01-21`

```sql
SELECT model_key, COUNT(*) AS quality_windows
FROM <control-plane-catalog>.<control-plane-schema>.quality_history
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- one row
- `quality_windows = 8`

```sql
SELECT model_key, COUNT(DISTINCT window_end) AS drift_windows
FROM <control-plane-catalog>.<control-plane-schema>.drift_metrics
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- one row
- `drift_windows = 8`

```sql
SELECT model_key, feature_name, metric_name, COUNT(*) AS row_count
FROM <control-plane-catalog>.<control-plane-schema>.drift_metrics
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
FROM <control-plane-catalog>.<control-plane-schema>.performance_metrics
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- at least one row

```sql
SELECT model_key, feature_name, metric_name, severity, status
FROM <control-plane-catalog>.<control-plane-schema>.incidents
WHERE model_key = 'fraud_model_demo'
ORDER BY observed_at DESC;
```

Expected:

- zero or more rows
- with this dataset, at least one drift incident is likely

```sql
SELECT model_key, requested_mode, run_kind, status, window_count, data_min_date, data_max_date
FROM <control-plane-catalog>.<control-plane-schema>.refresh_runs
WHERE model_key = 'fraud_model_demo'
ORDER BY started_at DESC
LIMIT 5;
```

Expected:

- at least one row
- the first refresh is typically `requested_mode = 'auto'`
- `run_kind` resolves to `backfill` on first run
- `status = 'completed'`
- `window_count = 8` for the scratch dataset with `Baseline Days = 7`

```sql
SELECT model_key, COUNT(*) AS comparison_windows
FROM <control-plane-catalog>.<control-plane-schema>.comparison_windows
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- one row
- `comparison_windows = 8`

```sql
SELECT model_key, COUNT(*) AS incident_events
FROM <control-plane-catalog>.<control-plane-schema>.incident_history
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- zero or more rows
- rows appear when drift crosses thresholds and also when a later window records recovery

If Lakebase is configured for the current app session or refresh workflow, also verify:

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

- the overview page shows a `Fraud Model Demo` card
- the selected monitor can be opened on the drift, quality, and performance pages
- the quality page shows `Rows Per Comparison Window`, `Null Rate Trends`, and `Prediction Mean Over Time`
- the performance page still shows KPI cards, feature options, and charts even when recent performance deltas are near zero; in that case the page should show an informational stability message instead of appearing blank
- open incidents still show on the overview/current inbox surfaces, while deeper incident lifecycle history is now persisted in the warehouse
- `total_rows`, `latest_data_date`, and `last_refresh_at` are populated in app readback

## Step 9: Verify Workflow Refresh

If the async trigger is not configured or fails, open the Databricks workflow `<app-name>-refresh` and run it manually once.

Expected:

- the workflow succeeds
- counts are non-zero in warehouse tables after the run

This workflow is packaged as a wheel task. If the job fails before your code runs, re-run bundle deploy first so the latest wheel artifact is uploaded.

Then rerun:

```sql
SELECT model_key, MAX(computed_at) AS last_quality_refresh
FROM <control-plane-catalog>.<control-plane-schema>.quality_metrics
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- timestamp updates after the workflow run

## Step 10: Exercise Failure Paths

Test these before client handoff:

### Missing feature selection

- clear all feature columns
- try to save

Expected:

- warning banner
- config not saved

### Non-numeric feature warning

Edit the monitor and include:

- `region`
- `merchant_segment`

Expected:

- save succeeds
- warning explains those fields are stored in the contract but skipped by the current drift engine

### Shared-table validation

- clear `Monitored Model ID Value`
- try to save the scratch monitor again

Expected:

- save fails with a validation error explaining that the source contains multiple model IDs

### External-label dedupe validation

- clear `External Labels Order Column`
- try to save again while still using the external labels table

Expected:

- save fails with a validation error explaining that repeated label keys require an order column

### Zero-match label join review

- keep the external labels table enabled
- intentionally choose the wrong `External Labels Join Column`
- click `Discover` again

Expected:

- the join validation shows `matched=0`
- the app shows a red warning telling you to correct the join column before continuing

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
