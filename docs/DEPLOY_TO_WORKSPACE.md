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
- privileges to create or run a Spark-capable Databricks workflow job
- an approved Databricks node type for the shared refresh cluster
- privileges to deploy apps and workflows
- privileges to write into a chosen Unity Catalog namespace for the control plane
- optional privileges to create:
  - the control-plane catalog if you want Model Lens to create it from the UI
  - a test catalog/schema for scratch data
  - the Lakebase database if using Lakebase mode

If you use Lakebase mode, you also need a Lakebase database user for the refresh workflow. In many workspaces this is the user or service principal that will run the job.

If you already have an app and want to keep its existing compute and app service principal, use the dedicated manual walkthrough:

- [Manual Setup With An Existing Databricks App](./MANUAL_EXISTING_APP_SETUP.md)
- [Constrained Workspace Runbook](./CONSTRAINED_WORKSPACE_RUNBOOK.md)

If the app already exists, do not use the generic `bundle deploy` path below with that same app name unless you first bind the bundle app resource to the existing app. Otherwise Databricks tries to create the app resource again and the deploy fails with an "App already exists" error.
If the operator cannot manage the app's `sql_warehouse` resource, do not keep retrying that bind/deploy path. Use the generated manual existing-app source path from [prepare_existing_app_source.py](../scripts/prepare_existing_app_source.py) instead.
The manual guide also now includes a detailed refresh-job creation and verification sequence for that path, including `jobs create`, `jobs reset`, `jobs get`, `jobs run-now`, and the `REFRESH_JOB_ID` hardening step.
If the customer workspace already has a shared refresh job, prefer reusing that exact job ID by resetting it to the generated `refresh-job.json` contract and then wiring the app with `REFRESH_JOB_ID`.

On Databricks CLI `v0.260.0`, the bundle can bind the SQL warehouse to the app but cannot automatically attach app-level `job` or `database` resources. That means:

- `warehouse_only` is fully automated
- `dev` / `prod` automate the workflow-side Lakebase sync inputs
- the app-side Lakebase read path is enabled either from the workspace setup fields in the UI or by pre-populating `LAKEBASE_INSTANCE_NAME` / `LAKEBASE_DATABASE_NAME` in `app.yaml` before `databricks apps deploy`
- the bundle-managed refresh workflow is one shared Spark job (`<app-name>-refresh`) scheduled hourly by default
- each monitor stores its own drift/performance cadence in the control plane, so one shared job can service many monitors without creating one Databricks job per model
- the app can accelerate onboarding refreshes asynchronously by resolving `REFRESH_JOB_ID` first, then falling back to `REFRESH_JOB_NAME` (default `model-lens-refresh`)
- the shared refresh job now uses job-level parameters that are pushed into the wheel task's named arguments, and the app triggers `jobs/run-now` with `job_parameters`, so `catalog`, `schema`, `scope`, and `model_key` overrides now reach the deployed workflow correctly. The workflow entrypoint also upgrades `scope=scheduler` plus a single `model_key` to `bootstrap` as a safety fallback
- when `REFRESH_JOB_NAME` is used, the resolver now falls back to full-workspace exact, suffix, and substring matching, so Databricks Asset Bundles development job names such as `[dev user] model-lens-refresh` still resolve reliably
- `REFRESH_JOB_ID` and `REFRESH_JOB_NAME` are deploy-time app environment variables in `app.yaml`; they are not values the operator edits during onboarding inside the app
- one shared refresh job remains the default deployment shape; a second bootstrap/backfill job is an optional large-tenant extension only
- if you explicitly set `BOOTSTRAP_REFRESH_JOB_ID` or `BOOTSTRAP_REFRESH_JOB_NAME`, direct bootstrap/backfill triggers use that optional second workflow, but readiness and scheduler-only fallback still depend on the main shared workflow

The commands below assume the default bundle variable `app_name=model-lens`.
If you override `app_name`, replace the app name in every `databricks apps ...` command and either:

- set `REFRESH_JOB_ID=<job-id>` before `databricks apps deploy`, or
- set `REFRESH_JOB_NAME=<app-name>-refresh`

Keep the checked-in `app.yaml` template environment-neutral. Set `REFRESH_JOB_ID` in the deployed app source for each workspace, but do not commit a real workspace job ID back into the repo template.

If no shared refresh workflow exists in the workspace yet, onboarding can still save the monitor config, but no scheduled pickup can happen until that shared workflow is created and the app points at it.

## Permission Matrix

Treat permissions as identity-specific:

- Deployer or platform operator:
  deploy apps and workflows, select the SQL warehouse, and provision or approve the control-plane namespace.
- Deployer or platform operator, when adding app resources manually in the Databricks Apps UI:
  `Can manage` on the app and `Can manage` on the resource being attached, such as the SQL warehouse.
- App service principal:
  `CAN_USE` on the SQL warehouse; source data `USE CATALOG`, `USE SCHEMA`, `SELECT`; control plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`.
- App service principal, if you want onboarding to accelerate the first run with `Run now`:
  `CAN MANAGE RUN` on the shared refresh workflow.
- App service principal, if Setup should create missing objects:
  `CREATE TABLE` in the control-plane schema; `CREATE SCHEMA` if the schema may not exist yet; `CREATE CATALOG` only if you intend to use the `Create catalog if missing` toggle.
  If the schema and tables already exist, Setup now reuses them and does not require those create privileges just to pass the control-plane step.
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
- `refresh_node_type_id`

There is no checked-in bundle default for `control_plane_catalog` or `control_plane_schema` anymore. Pass the real namespace for every workspace explicitly so the shared refresh job and the app point at the same control plane.

Example for a Hive Metastore deployment:

```bash
--var "control_plane_catalog=hive_metastore" \
--var "control_plane_schema=model_lens_control_plane"
```

Optional but important when you do not want the default names:

- `app_name`
- `refresh_spark_version`
  default `"15.4.x-scala2.12"`
- `refresh_data_security_mode`
  default `USER_ISOLATION`
- `refresh_num_workers`
  default `4`
- `refresh_timeout_seconds`
  default `14400`

Large-table tuning knobs:

- `REFRESH_SAMPLE_ROWS_PER_DAY`
  default `50000`
- `REFRESH_MAX_ROWS_PER_WINDOW`
  default `250000`
- `FEATURE_DETAIL_SAMPLE_ROWS_PER_DAY`
  default `50000`
- `FEATURE_DETAIL_MAX_ROWS`
  default `200000`
- `REFRESH_STALE_RUN_MINUTES`
  default `75`

These are app and workflow environment variables, not bundle vars. They cap pandas-side window loads so one extremely large monitor does not force a full-table in-memory read, and they control when the shared scheduler automatically marks an abandoned `running` refresh as failed so the monitor can be retried later.
The Spark refresh workflow no longer depends on those caps for core metric generation; they mainly govern bounded pandas fallback reads and detail views. The Spark refresh repository also clamps monitor-level worker fanout to `1` by default so one driver session is not shared across concurrent monitor threads unless you override that behavior intentionally.

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
cd <repo-root>
python3 -m pytest
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .

databricks bundle validate \
  -t warehouse_only \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "refresh_node_type_id=<spark-node-type-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>"
```

The local test suite now includes Spark-refresh regressions. Run it from an environment with the repo dev dependencies installed so `pyspark` is present; those Spark-specific tests still skip automatically when no local Java runtime is available.

Because the shared refresh workflow now runs on a Spark job cluster, `refresh_node_type_id` is mandatory even for `warehouse_only`.
The cluster now also defaults to `refresh_data_security_mode=USER_ISOLATION`, which is required for Unity Catalog access. Override it to `SINGLE_USER` only if your workspace policy requires that UC-capable access mode instead.

Lakebase-enabled:

```bash
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .

databricks bundle validate \
  -t dev \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "refresh_node_type_id=<spark-node-type-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>" \
  --var "lakebase_instance_name=<lakebase-instance-name>" \
  --var "lakebase_database_name=<lakebase-database-name>" \
  --var "lakebase_pguser=<lakebase-db-user>"
```

Expected result:

- tests pass
- bundle validation succeeds
- the wheel build succeeds and the bundle can resolve `../dist/*.whl` plus the Spark job-cluster libraries for the shared refresh workflow

If local bundle validation fails with a repo-local `.databricks/bundle` permission error, fix or remove that local bundle directory and rerun. That is a workstation ownership issue, not a Model Lens bundle contract issue.

Before calling the build broadly customer-ready, run these focused workspace release gates in addition to the local validation above:

- open Overview with at least two active monitors and confirm the bulk latest-quality/latest-drift queries render normally in a real Databricks workspace
- run at least one real Spark bootstrap and one incremental Spark refresh in Databricks; local skipped Spark tests are not enough proof for 20M-100M/day tenants
- force a severe numeric drift case where the latest current window sits fully outside the baseline range and confirm Drift still shows finite PSI / JS / KL values instead of blanks or warnings
- open feature detail and prediction detail on a window whose newest rows land later in the `window_end` day and confirm those same-day rows are still included

If you know the workspace will monitor very large or very wide inference tables, set the large-table caps deliberately before deploy rather than discovering OOM pressure during the first backfill. Lowering the caps is usually safer than immediately scaling compute.

## 3. Deploy The Bundle

This section assumes the bundle is managing creation of the Databricks App resource.
If the app already exists and you want to preserve its current service principal, use [Manual Setup With An Existing Databricks App](./MANUAL_EXISTING_APP_SETUP.md) instead of this section.

Warehouse-only:

```bash
databricks bundle deploy \
  -t warehouse_only \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "refresh_node_type_id=<spark-node-type-id>" \
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
  --var "refresh_node_type_id=<spark-node-type-id>" \
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
- the workflow `<app-name>-refresh` is created on Spark job compute, where `<app-name>` is `model-lens` unless you overrode `app_name`
- the workflow is scheduled hourly by default and acts as the shared pickup path for newly saved monitors
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
- read access to the source data catalog/schema/tables
- read/write access to the control-plane catalog/schema/tables
- if Model Lens should create the control-plane tables itself, `CREATE TABLE` in the control-plane schema

Treat the warehouse grant as a post-deploy check, not a one-time assumption. After every `databricks apps start model-lens` + `databricks apps deploy model-lens ...` cycle, verify the same app identity still has `CAN_USE` on the configured SQL warehouse and regrant it if the app shows warehouse-access errors.
If you want the app to accelerate onboarding with `Run now`, also verify `CAN MANAGE RUN` on the refresh workflow. If that permission is unavailable, the shared hourly job still remains the default pickup path only if that workflow already exists and the app is wired to it through `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`.
If `REFRESH_JOB_ID` is set, use that exact job as the permission target and grant the app service principal `CAN_MANAGE_RUN` on job `<REFRESH_JOB_ID>`.
If you configure `BOOTSTRAP_REFRESH_JOB_ID` for the optional second lane, grant the app service principal `CAN_MANAGE_RUN` on that second job too if you expect direct `Run First Refresh` acceleration through it. Without that grant, onboarding should still work in scheduler-only mode through the main shared workflow.
If you want operators to edit the shared wake interval from `Monitor Settings -> Admin`, also grant the app service principal `CAN_MANAGE` on the shared refresh workflow. Without that grant, the app will show the current schedule read-only and tell the operator to update the job externally.

For large-table customers, also verify that the deployed app and shared refresh workflow configuration include the intended readback/fallback settings:

- `REFRESH_SAMPLE_ROWS_PER_DAY`
- `REFRESH_MAX_ROWS_PER_WINDOW`
- `FEATURE_DETAIL_SAMPLE_ROWS_PER_DAY`
- `FEATURE_DETAIL_MAX_ROWS`

At a minimum, the app identity and the scheduled refresh job identity must be able to do this:

- source data: `USE CATALOG`, `USE SCHEMA`, `SELECT`
- control plane: `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`

Additionally:

- app identity: `CAN MANAGE RUN` on the refresh job only if the app should accelerate onboarding with `Run now`
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

The wizard does not unlock the `Discover` step until setup succeeds for the current control-plane namespace. If setup fails, fix the underlying issue and click `Setup Control Plane` again to retry. After redeploying a newer Model Lens build into an existing workspace, rerun `Setup Control Plane` once so additive schema migrations are applied. That rerun is intentionally idempotent: existing columns/tables are checked before additive `ALTER TABLE` migrations are attempted. Setup also backfills legacy completed `refresh_runs` rows so older bootstrap runs gain `generation_id` and `published_at`, which makes pre-generation monitors visible to the new published-generation readers again. For labeled monitors that already have persisted performance data but predate `daily_label_metrics`, setup now also queues a one-time `performance_repair` so the next scheduled repair run backfills those daily labeled facts automatically. If those facts are still missing, later `performance_repair` runs also bypass the unchanged-label-watermark fast skip until the backfill succeeds.

Then click `Validate Workspace Wiring`.

Expected readiness modes:

- `fully_ready`: the app can save monitors and trigger bootstrap immediately
- `scheduler_only`: the app can save monitors and rely on the scheduled shared workflow pickup path

When the readiness card shows `Immediate Bootstrap: Grant CAN_MANAGE_RUN on job <id>`, the workflow wiring is fine and the missing step is the Databricks job permission grant for the app service principal.
When the readiness card shows `Immediate Bootstrap: Verification unavailable`, the app could not inspect the workflow ACL. Direct trigger may still work, so test `Run First Refresh` before assuming the grant is missing.

If the readiness card stays `not ready`, onboarding is intentionally blocked until the shared workflow wiring is fixed.

Recommended:

- point the app at a pre-created namespace
- leave `Create catalog if missing` off unless you are using an admin identity and intentionally want Model Lens to create the catalog
- keep the app namespace fields aligned with the bundle `control_plane_catalog` / `control_plane_schema` vars so the workflow and the app write to the same place
- for Git/manual app deploys, stamp those same catalog/schema values into the deployed `app.yaml`; the repo template is intentionally blank and should not be treated as a workspace default
- if you want app-side Lakebase reads immediately, fill in `Lakebase Instance Name` and `Lakebase Database Name` first
- prefer `REFRESH_JOB_ID` over `REFRESH_JOB_NAME`; name lookup is still supported but treated as a fallback

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
- `daily_quality_profiles`
- `daily_feature_profiles`
- `performance_metrics`
- `daily_performance_profiles`
- `performance_bin_specs`
- `incidents`
- `incident_history`
- `refresh_runs`
- `comparison_windows`
- `monitor_runtime_state`

## 7. Load Test Data

Run [`examples/scratch_dataset.sql`](../examples/scratch_dataset.sql) in Databricks SQL.

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
   If your inference table already contains the true labels, leave `Optional Labels Table` blank and confirm the inferred `Label Column In Source` instead.
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
   If the same join column name exists in both inference and labels tables, `Entity ID Column` can be left blank and Model Lens will reuse the shared join column on the inference side.
   Hyphenated feature or slice columns such as `us-central1` are supported and should not require renaming before onboarding.
   If you are using labels from the inference table itself, keep `Optional Labels Table` empty and set `Label Column In Source` instead of the external-label fields.
6. In the same `Advanced` section, confirm features:
   - `amount`
   - `velocity_7d`
   - `device_score`
   Model Lens now keeps the full detected numeric feature set on by default. If your real table has dozens of numeric features, the first refresh may still take longer, but the app should not silently trim the contract.
7. Continue to `Activate` and confirm:
   - `Drift / Quality Refresh`: `Every 6 Hours`
   - `Performance Refresh`: `Daily (7-Day Repair)` when labels are enabled
   - `Enable scheduled refreshes for this monitor`: on
   - `Tracked Performance Metrics`: `F1 Score`, `Precision`, `Recall`
   - `Default Performance Metric`: `F1 Score`
8. Click `Save Monitor`

Expected result:

- the config is saved
- the monitor is marked `pending bootstrap` in `monitor_runtime_state`
- the app returns immediately instead of blocking on the refresh computation
- if the app has `CAN MANAGE RUN`, the shared refresh job is triggered asynchronously for bootstrap
- if the app cannot resolve the workflow or lacks `Run now` permission, the monitor is still saved; automatic pickup only happens if the shared hourly workflow already exists and the app is wired to it through `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`
- if the monitor stays `pending bootstrap`, the `Monitor Settings` page exposes `Run First Refresh` so the operator can retry the selected monitor after fixing workflow wiring or permissions
- inside the shared job, monitor refreshes can run concurrently, but only up to the configured `MAX_PARALLEL_REFRESH_WORKERS` cap and never with two scopes for the same model in one scheduler pass
- if one monitor hits an unexpected worker-level exception, that result is recorded as a failed monitor refresh instead of aborting the whole shared batch
- each monitor scope is processed from one bounded projected source-range load, and the workflow derives the persisted window/history rows from the daily profile layer built for that range instead of re-querying every comparison window
- on incremental runs, the workflow also reuses already-persisted daily profile facts for the affected date span before rewriting the touched window/history rows
- when the Spark repository is active, those fact/window/history rewrites are written back through Spark/Delta persistence rather than row-batch warehouse inserts
- `quality_metrics` stays model-wide because the workflow rebuilds that compatibility row from all persisted `daily_quality_profiles`, not just from the bounded incremental slice
- numeric performance buckets stay stable across runs because the workflow stores canonical `performance_bin_specs` per monitor feature and reuses them during later performance repair
- for labels stored directly in the inference table, performance repair compares an opaque label-freshness signature over the repair horizon, so late backfills on older rows still trigger recompute
- the cadence you selected in the review step is stored with the monitor and can be edited later from the `Monitor Settings` page
- the selected built-in performance metrics and default metric are stored with the monitor and drive both refresh persistence and the Performance-tab selector
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
- `refresh_runs` records the refresh mode, status, and window counts for audit/debugging, and each monitor attempt now gets its row before source-range discovery so early skips/failures are auditable too
- `comparison_windows` records one row per logical baseline/current pairing
- `quality_metrics` reflects the full persisted monitor history even after incremental refreshes
- `performance_bin_specs` contains one canonical bucket spec row per numeric feature that participates in performance repair
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
