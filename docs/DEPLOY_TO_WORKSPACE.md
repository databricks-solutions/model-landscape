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

If you already have an app and want to keep its existing compute and app service principal, use the dedicated manual walkthrough:

- [Manual Setup With An Existing Databricks App](/Users/volo.vragov/Desktop/work/model-lens/docs/MANUAL_EXISTING_APP_SETUP.md)

If the app already exists, do not use the generic `bundle deploy` path below with that same app name unless you first bind the bundle app resource to the existing app. Otherwise Databricks tries to create the app resource again and the deploy fails with an "App already exists" error.
If the operator cannot manage the app's `sql_warehouse` resource, do not keep retrying that bind/deploy path. Use the generated manual existing-app source path from [prepare_existing_app_source.py](/Users/volo.vragov/Desktop/work/model-lens/scripts/prepare_existing_app_source.py) instead.
The manual guide also now includes a detailed refresh-job creation and verification sequence for that path, including `jobs create`, `jobs reset`, `jobs get`, `jobs run-now`, and the `REFRESH_JOB_ID` hardening step.

On Databricks CLI `v0.260.0`, the bundle can bind the SQL warehouse to the app but cannot automatically attach app-level `job` or `database` resources. That means:

- `warehouse_only` is fully automated
- `dev` / `prod` automate the workflow-side Lakebase sync inputs
- the app-side Lakebase read path is enabled either from the workspace setup fields in the UI or by pre-populating `LAKEBASE_INSTANCE_NAME` / `LAKEBASE_DATABASE_NAME` in `app.yaml` before `databricks apps deploy`
- the app triggers onboarding refreshes asynchronously by resolving `REFRESH_JOB_ID` first, then falling back to `REFRESH_JOB_NAME` (default `model-lens-refresh`)
- when `REFRESH_JOB_NAME` is used, the resolver now also accepts Databricks Asset Bundles development job names that end with the configured base name, such as `[dev user] model-lens-refresh`

The commands below assume the default bundle variable `app_name=model-lens`.
If you override `app_name`, replace the app name in every `databricks apps ...` command and either:

- set `REFRESH_JOB_ID=<job-id>` before `databricks apps deploy`, or
- set `REFRESH_JOB_NAME=<app-name>-refresh`

## Permission Matrix

Treat permissions as identity-specific:

- Deployer or platform operator:
  deploy apps and workflows, select the SQL warehouse, and provision or approve the control-plane namespace.
- Deployer or platform operator, when adding app resources manually in the Databricks Apps UI:
  `Can manage` on the app and `Can manage` on the resource being attached, such as the SQL warehouse.
- App service principal:
  `CAN_USE` on the SQL warehouse; `CAN MANAGE RUN` on the refresh workflow; source data `USE CATALOG`, `USE SCHEMA`, `SELECT`; control plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`.
- App service principal, if Setup should create missing objects:
  `CREATE TABLE` in the control-plane schema; `CREATE SCHEMA` if the schema may not exist yet; `CREATE CATALOG` only if you intend to use the `Create catalog if missing` toggle.
- Refresh workflow identity:
  the same warehouse, source-data, and control-plane permissions as the app, because the workflow reads source data and writes monitoring results.
- Optional MLflow-assisted onboarding:
  read access to the target experiment and/or registered model metadata.
- Optional Lakebase app reads or workflow sync:
  permission to resolve the Lakebase instance and connect to the target database; workflow sync also needs write access to the target Lakebase schema.

## Variables You Must Supply

For `warehouse_only`, Model Lens expects:

- `sql_warehouse_id`
- `control_plane_catalog`
- `control_plane_schema`

Optional but important when you do not want the default names:

- `app_name`

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

This section assumes the bundle is managing creation of the Databricks App resource.
If the app already exists and you want to preserve its current service principal, use [Manual Setup With An Existing Databricks App](/Users/volo.vragov/Desktop/work/model-lens/docs/MANUAL_EXISTING_APP_SETUP.md) instead of this section.

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
- the workflow `<app-name>-refresh` is created, where `<app-name>` is `model-lens` unless you overrode `app_name`
- `databricks apps start model-lens` brings the app compute into `ACTIVE`
- after `databricks apps deploy`, the app source is deployed to compute
- on CLI `v0.260.0`, the app resource only binds the SQL warehouse; Lakebase app reads are configured from the app session fields or app env overrides

## 4. Grant The App Access

Model Lens creates an app service principal automatically, but it does not receive warehouse or Unity Catalog access by default.
Also do not assume the bundle's `sql_warehouse: CAN_USE` binding is sufficient forever. If the app is started, source-deployed, restarted, or otherwise managed outside the bundle lifecycle, explicitly verify the warehouse grant after deployment.
If you add or edit the `sql_warehouse` app resource manually in the Databricks Apps UI, the operator doing that must have `Can manage` on both the app and the warehouse.

Find the app identity:

```bash
databricks apps get model-lens -o json
```

Then grant the app identity all of the following:

- `CAN_USE` on the SQL warehouse used by Model Lens
- `CAN MANAGE RUN` on the refresh workflow used by Model Lens
- read access to the source data catalog/schema/tables
- read/write access to the control-plane catalog/schema/tables
- if Model Lens should create the control-plane tables itself, `CREATE TABLE` in the control-plane schema

Treat the warehouse grant as a post-deploy check, not a one-time assumption. After every `databricks apps start model-lens` + `databricks apps deploy model-lens ...` cycle, verify the same app identity still has `CAN_USE` on the configured SQL warehouse and regrant it if the app shows warehouse-access errors.
Do the same for `CAN MANAGE RUN` on the refresh workflow if the app is expected to trigger onboarding refreshes asynchronously.

At a minimum, the app identity and the scheduled refresh job identity must be able to do this:

- source data: `USE CATALOG`, `USE SCHEMA`, `SELECT`
- control plane: `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`

Additionally:

- app identity: `CAN MANAGE RUN` on the refresh job
- refresh job Run as identity: the data and control-plane privileges above, because `Run now` uses the job owner's or Run as identity's privileges for the actual compute

If you want Model Lens to initialize the control plane from the UI, the identity running setup also needs create privileges in that namespace.

## 5. Open The App

Open the deployed Databricks App named `model-lens`.

Verify:

- the title is `Model Lens`
- `SQL_WAREHOUSE_ID` is populated
- `REFRESH_JOB_ID` is populated or `REFRESH_JOB_NAME` matches the deployed workflow name
- the `Control Plane Catalog` and `Control Plane Schema` fields point at the namespace you intend to use
- by default, `USE_LAKEBASE_READ_MODEL` reflects the deployed app environment and will usually be `false`
- if you want fast app reads, enter `Lakebase Instance Name` and `Lakebase Database Name` in the setup card before loading the dashboard
- in warehouse-only mode, the app may show an informational banner recommending Lakebase if the workspace exposes Lakebase instances

## 6. Initialize The Control Plane

In the app `Setup` step, click `Setup Control Plane`.

The wizard does not unlock the `Discover` step until setup succeeds for the current control-plane namespace. If setup fails, fix the underlying issue and click `Setup Control Plane` again to retry.

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
- `quality_history`
- `performance_metrics`
- `incidents`
- `incident_history`
- `refresh_runs`
- `comparison_windows`

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
   If you entered a labels table, the app should also show:
   - labels table columns and sample rows
   - detected `Join Column`, `Label Column`, and `Order Column`
   - join validation with matched rows, unmatched rows, and duplicate label keys
   Discovery should prefer the shared string key if both tables expose one, accept ISO timestamp strings as timestamp/order candidates, and keep all detected numeric features selected by default.
   If the labels join shows `matched=0`, treat that as an error and fix the join column before continuing.
4. Continue to the `Confirm` step. The core fields should already be inferred. Use:
   - `Display Name`: `Fraud Model Demo`
   - `Model Key`: `fraud_model_demo`
   - `Problem Type`: `classification`
   - `Baseline Policy`: `Rolling`
   - `Baseline Days`: `7`
   Fixed baselines are also supported here if you want to compare against a known-good historical window instead of trailing windows.
   After the first successful refresh, a stable comparison window should still render the Performance page with KPI cards, charts, and an informational stability message instead of an empty state.
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
   Model Lens now keeps the full detected numeric feature set on by default. If your real table has dozens of numeric features, the first refresh may still take longer, but the app should not silently trim the contract.
7. Continue to `Activate`
8. Click `Save Monitor And Trigger Refresh`

Expected result:

- the config is saved
- the app returns immediately instead of blocking on the refresh computation
- the refresh job is triggered asynchronously
- if the app cannot resolve the workflow from `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`, it warns that the monitor was saved but the workflow must be run manually
- the monitor appears in the app
- after the workflow finishes, the new monitor appears on the overview page and the analysis pages can load it
- the first refresh still backfills historical daily comparison windows immediately instead of writing only a single latest snapshot

## 9. Verify Persisted State

Verify the warehouse system of record after the triggered workflow finishes:

```sql
SELECT model_key, display_name, source_table, status
FROM <control-plane-catalog>.<control-plane-schema>.monitor_configs
WHERE model_key = 'fraud_model_demo';

SELECT model_key, total_rows, min_date, max_date
FROM <control-plane-catalog>.<control-plane-schema>.quality_metrics
WHERE model_key = 'fraud_model_demo';

SELECT model_key, COUNT(*) AS quality_windows
FROM <control-plane-catalog>.<control-plane-schema>.quality_history
WHERE model_key = 'fraud_model_demo'
GROUP BY model_key;

SELECT model_key, COUNT(DISTINCT window_end) AS drift_windows
FROM <control-plane-catalog>.<control-plane-schema>.drift_metrics
WHERE model_key = 'fraud_model_demo'
GROUP BY model_key;

SELECT model_key, COUNT(*) AS incident_events
FROM <control-plane-catalog>.<control-plane-schema>.incident_history
WHERE model_key = 'fraud_model_demo'
GROUP BY model_key;

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
- `quality_windows` is populated immediately after the first refresh for datasets with enough history
- `drift_windows` is populated immediately after the first refresh for datasets with enough history
- `incident_events` is populated when drift crosses thresholds or later recovers across comparison windows
- `refresh_runs` records the refresh mode, status, and window counts for audit/debugging
- `comparison_windows` records one row per logical baseline/current pairing
- if Lakebase is configured for the current app session or refresh workflow, Lakebase tables contain the latest monitor inventory, summary, and open incidents

## 10. Verify The Workflow

If the async trigger is not configured or fails, open the workflow `<app-name>-refresh` and run it once manually.

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
