# Deploy To A Workspace

This is the step-by-step path to deploy Model Lens into your own Databricks workspace before handing it to a client.

## Choose A Deployment Mode

Model Lens supports two modes:

1. `warehouse_only`
   Use this if you want the simplest path today.

2. `dev` or `prod`
   Use this when you want the Lakebase-accelerated UI.

## What You Need Before Deploy

You need a workspace with:

- Databricks CLI auth already working
- one SQL warehouse
- serverless jobs enabled
- privileges to deploy apps and workflows
- privileges to create and write:
  - `model_observability.control_plane`
  - a test catalog/schema for scratch data
  - the Lakebase database if using Lakebase mode

If you use Lakebase mode, you also need a Lakebase database user for the refresh workflow. In many workspaces this is the user or service principal that will run the job.

You do not need to set manual Postgres environment variables for the app. In Lakebase mode, the Databricks App resource supplies the managed database connection context.

## Variables You Must Supply

For `warehouse_only`, Model Lens expects:

- `sql_warehouse_id`

For `dev` or `prod`, Model Lens expects:

- `sql_warehouse_id`
- `lakebase_instance_name`
- `lakebase_database_name`
- `lakebase_pguser`

## 1. Validate Locally

From the repo root:

Warehouse-only:

```bash
cd /Users/volo.vragov/Desktop/work/model-lens
python3 -m pytest

databricks bundle validate \
  -t warehouse_only \
  --var "sql_warehouse_id=<sql-warehouse-id>"
```

Lakebase-enabled:

```bash
databricks bundle validate \
  -t dev \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "lakebase_instance_name=<lakebase-instance-name>" \
  --var "lakebase_database_name=<lakebase-database-name>" \
  --var "lakebase_pguser=<lakebase-db-user>"
```

Expected result:

- tests pass
- bundle validation succeeds
- the wheel build path is valid when the bundle packages the refresh job

## 2. Deploy The Bundle

Warehouse-only:

```bash
databricks bundle deploy \
  -t warehouse_only \
  --var "sql_warehouse_id=<sql-warehouse-id>"

databricks apps deploy model-lens \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-lens/warehouse_only/files

databricks apps get model-lens
```

Lakebase-enabled:

```bash
databricks bundle deploy \
  -t dev \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "lakebase_instance_name=<lakebase-instance-name>" \
  --var "lakebase_database_name=<lakebase-database-name>" \
  --var "lakebase_pguser=<lakebase-db-user>"

databricks apps deploy model-lens \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-lens/dev/files

databricks apps get model-lens
```

Expected result:

- the app `model-lens` is created
- the workflow `model-lens-refresh` is created
- in Lakebase mode, the app gets a Lakebase database resource
- after `databricks apps deploy`, the app source is deployed to compute

## 3. Open The App

Open the deployed Databricks App named `model-lens`.

Verify:

- the title is `Model Lens`
- `SQL_WAREHOUSE_ID` is populated
- in warehouse-only mode, `USE_LAKEBASE_READ_MODEL` is `false`
- in Lakebase mode, `USE_LAKEBASE_READ_MODEL` is `true` and `LAKEBASE_DATABASE_NAME` is shown
- in warehouse-only mode, the app may show an informational banner recommending Lakebase if the workspace exposes Lakebase instances

## 4. Initialize The Control Plane

In the app, click `Setup Control Plane`.

Expected result:

- the Unity Catalog control-plane tables are created
- in Lakebase mode, the Lakebase projection schema is created

Verify in SQL:

```sql
SHOW TABLES IN model_observability.control_plane;
```

Expected:

- `monitor_configs`
- `drift_metrics`
- `quality_metrics`
- `performance_metrics`
- `incidents`

## 5. Load Test Data

Run [`examples/scratch_dataset.sql`](/Users/volo.vragov/Desktop/work/model-lens/examples/scratch_dataset.sql) in Databricks SQL.

If `main` is not writable in your workspace, replace the catalog name first.

## 6. Create The First Monitor

In the app:

1. Source table:
   - `main.model_lens_demo.inference_logs`
2. Click `Scan`
3. Use:
   - `Display Name`: `Fraud Model Demo`
   - `Model Key`: `fraud_model_demo`
   - `Timestamp Column`: `event_ts`
   - `Model ID Column`: `model_id`
   - `Prediction Column`: `prediction`
   - `Entity ID Column`: `entity_id`
   - `External Labels Table`: `main.model_lens_demo.labels`
   - `External Labels Join Column`: `entity_id`
   - `External Label Column`: `label`
   - `Problem Type`: `classification`
   - `Baseline Days`: `7`
4. Select features:
   - `amount`
   - `velocity_7d`
   - `device_score`
5. Click `Save Monitor And Run Initial Refresh`

Expected result:

- the config is saved
- the refresh runs
- the monitor appears in the app
- the summary and incident queries come back quickly

## 7. Verify Persisted State

Verify the warehouse system of record:

```sql
SELECT model_key, display_name, source_table, status
FROM model_observability.control_plane.monitor_configs
WHERE model_key = 'fraud_model_demo';

SELECT model_key, total_rows, min_date, max_date
FROM model_observability.control_plane.quality_metrics
WHERE model_key = 'fraud_model_demo';

SELECT model_key, feature_name, metric_name, metric_value
FROM model_observability.control_plane.drift_metrics
WHERE model_key = 'fraud_model_demo'
ORDER BY feature_name, metric_name;
```

If you deployed Lakebase mode, verify the Lakebase UI projection:

```sql
SELECT * FROM model_lens_ui.monitor_inventory ORDER BY display_name;
SELECT * FROM model_lens_ui.monitor_summary ORDER BY display_name;
SELECT * FROM model_lens_ui.open_incidents ORDER BY observed_at DESC;
```

Expected result:

- warehouse tables contain the durable metrics
- in Lakebase mode, Lakebase tables contain the latest monitor inventory, summary, and open incidents

## 8. Verify The Workflow

Open the workflow `model-lens-refresh` and run it once manually.

Then repeat the same warehouse checks and, if applicable, the Lakebase checks.

Expected result:

- the workflow completes successfully
- warehouse metrics update
- in Lakebase mode, the Lakebase projection updates too

## 9. What To Check Before Sending To A Client

Do not send this to a client until all of the following are true:

- app deploy succeeds in your workspace
- setup works from the UI
- a real or scratch monitor can be created
- the refresh workflow completes
- warehouse tables populate correctly
- in Lakebase mode, Lakebase tables populate correctly
- the app loads summary and incidents without errors

## Common Failure Modes

- Missing SQL warehouse permission:
  - the app opens but scan/setup/refresh calls fail
- Missing Lakebase access for the app:
  - monitor summary falls back to the warehouse or fails to accelerate
- Missing Lakebase access for the job identity:
  - warehouse metrics refresh, but the Lakebase projection does not update
- App compute stopped:
  - `databricks apps get model-lens` shows stopped or unavailable compute; run `databricks apps start model-lens`
- App resource exists but source is not deployed:
  - the app reports that source code has not been deployed yet; rerun `databricks apps deploy model-lens --source-code-path ...`
- Insufficient source history:
  - the monitor saves, but refresh reports that no comparable baseline/current window exists yet
