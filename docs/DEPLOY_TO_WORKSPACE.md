# Deploy To A Workspace

This is the step-by-step path to deploy Model Lens into your own Databricks workspace before handing it to a client.

## Choose A Deployment Mode

Model Lens supports two modes:

1. `warehouse_only`
   Use this if you want the simplest path today.

2. `dev` or `prod`
   Use this when you want the scheduled refresh workflow to keep a Lakebase projection current.

## What You Need Before Deploy

You need a workspace with:

- Databricks CLI auth already working
- one SQL warehouse
- serverless jobs enabled
- privileges to deploy apps and workflows
- privileges to write into a chosen Unity Catalog namespace for the control plane
- optional privileges to create:
  - the control-plane catalog if you want Model Lens to create it from the UI
  - a test catalog/schema for scratch data
  - the Lakebase database if using Lakebase mode

If you use Lakebase mode, you also need a Lakebase database user for the refresh workflow. In many workspaces this is the user or service principal that will run the job.

On Databricks CLI `v0.260.0`, the bundle can bind the SQL warehouse to the app but cannot automatically attach app-level `job` or `database` resources. That means:

- `warehouse_only` is fully automated
- `dev` / `prod` automate the workflow-side Lakebase sync inputs
- the app-side Lakebase read path is enabled either from the workspace setup fields in the UI or by pre-populating `LAKEBASE_INSTANCE_NAME` / `LAKEBASE_DATABASE_NAME` in `app.yaml` before `databricks apps deploy`

## Variables You Must Supply

For `warehouse_only`, Model Lens expects:

- `sql_warehouse_id`
- `control_plane_catalog`
- `control_plane_schema`

For `dev` or `prod`, Model Lens expects:

- `sql_warehouse_id`
- `control_plane_catalog`
- `control_plane_schema`
- `lakebase_instance_name`
- `lakebase_database_name`
- `lakebase_pguser`

## 1. Cleanup If You Are Re-Deploying

If you manually deleted the app or are reusing a workspace with old bundle state, clean up the stale workspace bundle directory first:

```bash
databricks workspace delete /Workspace/Users/<your-email>/.bundle/model-lens --recursive
```

Do this before the next `databricks bundle deploy`.

## 2. Validate Locally

From the repo root:

Warehouse-only:

```bash
cd /Users/volo.vragov/Desktop/work/model-lens
python3 -m pytest
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .

databricks bundle validate \
  -t warehouse_only \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>"
```

Lakebase-enabled:

```bash
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .

databricks bundle validate \
  -t dev \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>" \
  --var "lakebase_instance_name=<lakebase-instance-name>" \
  --var "lakebase_database_name=<lakebase-database-name>" \
  --var "lakebase_pguser=<lakebase-db-user>"
```

Expected result:

- tests pass
- bundle validation succeeds
- the wheel build succeeds and the bundle can resolve `../dist/*.whl` for the serverless workflow environment

## 3. Deploy The Bundle

Warehouse-only:

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
```

Lakebase-enabled:

```bash
databricks bundle deploy \
  -t dev \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>" \
  --var "lakebase_instance_name=<lakebase-instance-name>" \
  --var "lakebase_database_name=<lakebase-database-name>" \
  --var "lakebase_pguser=<lakebase-db-user>"

databricks apps start model-lens

databricks apps deploy model-lens \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-lens/dev/files

databricks apps get model-lens
```

Expected result:

- the app `model-lens` is created
- the workflow `model-lens-refresh` is created
- `databricks apps start model-lens` brings the app compute into `ACTIVE`
- after `databricks apps deploy`, the app source is deployed to compute
- on CLI `v0.260.0`, the app resource only binds the SQL warehouse; Lakebase app reads are configured from the app session fields or app env overrides

## 4. Grant The App Access

Model Lens creates an app service principal automatically, but it does not receive warehouse or Unity Catalog access by default.

Find the app identity:

```bash
databricks apps get model-lens -o json
```

Then grant the app identity all of the following:

- `CAN_USE` on the SQL warehouse used by Model Lens
- read access to the source data catalog/schema/tables
- read/write access to the control-plane catalog/schema/tables
- if Model Lens should create the control-plane tables itself, `CREATE TABLE` in the control-plane schema

At a minimum, the app identity and the scheduled refresh job identity must be able to do this:

- source data: `USE CATALOG`, `USE SCHEMA`, `SELECT`
- control plane: `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`

If you want Model Lens to initialize the control plane from the UI, the identity running setup also needs create privileges in that namespace.

## 5. Open The App

Open the deployed Databricks App named `model-lens`.

Verify:

- the title is `Model Lens`
- `SQL_WAREHOUSE_ID` is populated
- the `Control Plane Catalog` and `Control Plane Schema` fields point at the namespace you intend to use
- by default, `USE_LAKEBASE_READ_MODEL` reflects the deployed app environment and will usually be `false`
- if you want fast app reads, enter `Lakebase Instance Name` and `Lakebase Database Name` in the setup card before loading the dashboard
- in warehouse-only mode, the app may show an informational banner recommending Lakebase if the workspace exposes Lakebase instances

## 6. Initialize The Control Plane

In the app `Setup` step, click `Setup Control Plane`.

The wizard does not unlock the `Discover` step until setup succeeds for the current namespace and optional Lakebase session values.

Recommended:

- point the app at a pre-created namespace
- leave `Create catalog if missing` off unless you are using an admin identity and intentionally want Model Lens to create the catalog
- keep the app namespace fields aligned with the bundle `control_plane_catalog` / `control_plane_schema` vars so the workflow and the app write to the same place
- if you want app-side Lakebase reads immediately, fill in `Lakebase Instance Name` and `Lakebase Database Name` first

Expected result:

- the Unity Catalog control-plane tables are created
- if Lakebase session fields are populated, the Lakebase projection schema is created too

Verify in SQL:

```sql
SHOW TABLES IN <control-plane-catalog>.<control-plane-schema>;
```

Expected:

- `monitor_configs`
- `drift_metrics`
- `quality_metrics`
- `performance_metrics`
- `incidents`

## 7. Load Test Data

Run [`examples/scratch_dataset.sql`](/Users/volo.vragov/Desktop/work/model-lens/examples/scratch_dataset.sql) in Databricks SQL.

If `main` is not writable in your workspace, replace the catalog name first.

## 8. Create The First Monitor

In the app:

1. Continue to the `Discover` step and set the source table:
   - `main.model_lens_demo.inference_logs`
2. Optional draft inputs:
   - `Optional Labels Table`: `main.model_lens_demo.labels`
   - `Optional MLflow Experiment`: `<your experiment path>` if you want lineage-assisted discovery
   - `Optional Registered Model`: `<catalog>.<schema>.<model>` if you want registry-assisted discovery
3. Click `Discover`
4. Continue to the `Confirm` step. The core fields should already be inferred. Use:
   - `Display Name`: `Fraud Model Demo`
   - `Model Key`: `fraud_model_demo`
   - `Problem Type`: `classification`
   - `Baseline Days`: `7`
5. Open `Advanced mappings and overrides` and confirm:
   - `Timestamp Column`: `event_ts`
   - `Model ID Column`: `model_id`
   - `Monitored Model ID Value`: `fraud_model_v1`
   - `Prediction Column`: `prediction`
   - `Entity ID Column`: `entity_id`
   - `External Labels Join Column`: `entity_id`
   - `External Label Column`: `label`
   - `External Labels Order Column`: `label_timestamp`
6. In the same `Advanced` section, confirm features:
   - `amount`
   - `velocity_7d`
   - `device_score`
7. Continue to `Activate`
8. Click `Save Monitor And Run Initial Refresh`

Expected result:

- the config is saved
- the refresh runs
- the monitor appears in the app
- the new monitor appears on the overview page and the analysis pages can load it

## 9. Verify Persisted State

Verify the warehouse system of record:

```sql
SELECT model_key, display_name, source_table, status
FROM <control-plane-catalog>.<control-plane-schema>.monitor_configs
WHERE model_key = 'fraud_model_demo';

SELECT model_key, total_rows, min_date, max_date
FROM <control-plane-catalog>.<control-plane-schema>.quality_metrics
WHERE model_key = 'fraud_model_demo';

SELECT model_key, feature_name, metric_name, metric_value
FROM <control-plane-catalog>.<control-plane-schema>.drift_metrics
WHERE model_key = 'fraud_model_demo'
ORDER BY feature_name, metric_name;
```

If Lakebase is configured for the current app session or refresh workflow, verify the Lakebase UI projection:

```sql
SELECT * FROM model_lens_ui.monitor_inventory ORDER BY display_name;
SELECT * FROM model_lens_ui.monitor_summary ORDER BY display_name;
SELECT * FROM model_lens_ui.open_incidents ORDER BY observed_at DESC;
```

Expected result:

- warehouse tables contain the durable metrics
- if Lakebase is configured for the current app session or refresh workflow, Lakebase tables contain the latest monitor inventory, summary, and open incidents

## 10. Verify The Workflow

Open the workflow `model-lens-refresh` and run it once manually.

Then repeat the same warehouse checks and, if applicable, the Lakebase checks.

Expected result:

- the workflow completes successfully
- warehouse metrics update
- in `dev` / `prod`, the workflow also updates the Lakebase projection
- in `warehouse_only`, the app can still use Lakebase reads if you entered Lakebase instance/database values in the UI, but scheduled refreshes do not sync Lakebase automatically

## 11. What To Check Before Sending To A Client

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
- Missing control-plane namespace privileges:
  - `Setup Control Plane` fails or refreshes cannot read/write monitor state in the selected namespace
- Shared inference table without `Monitored Model ID Value`:
  - save fails with a validation message because the source contains multiple model IDs
- External labels table with duplicate join keys but no `External Labels Order Column`:
  - save fails with a validation message instead of silently duplicating inference rows
- Missing Lakebase access for the app:
  - monitor summary falls back to the warehouse instead of accelerating
- Missing Lakebase access for the job identity:
  - warehouse metrics refresh, but the Lakebase projection does not update
- App compute stopped:
  - `databricks apps get model-lens` shows stopped or unavailable compute; run `databricks apps start model-lens`
- App resource exists but source is not deployed:
  - the app reports that source code has not been deployed yet; rerun `databricks apps deploy model-lens --source-code-path ...`
- Insufficient source history:
  - the monitor saves, but refresh reports that no comparable baseline/current window exists yet
