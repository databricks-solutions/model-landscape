# Manual Setup With An Existing Databricks App

This guide is for the case where you already have a Databricks App and want Model Lens to run on that existing app compute and keep using that existing app service principal.

This path does not use `ai_dev_kit`.
It assumes you will wire the app manually with the Databricks UI and CLI.

## Important Constraint

Databricks Apps create a dedicated service principal per app instance.
You cannot attach an arbitrary different service principal to an app later.

That means:

- if you want to keep the existing app service principal, keep the existing app
- do not delete and recreate the app
- deploy Model Lens into the same app name so the same app identity remains in use

As of `2026-04-01T18:22Z`, this matches the current Databricks app authorization and resource model in the official Databricks docs:

- [Configure authorization in a Databricks app](https://docs.databricks.com/gcp/en/dev-tools/databricks-apps/auth)
- [Add resources to a Databricks app](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/resources)
- [Add a SQL warehouse resource to a Databricks app](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/sql-warehouse)
- [Configure permissions for a Databricks app](https://docs.databricks.com/gcp/en/dev-tools/databricks-apps/permissions)
- [Trigger a single job run](https://docs.databricks.com/aws/en/jobs/run-now)
- [Databricks CLI workspace commands](https://docs.databricks.com/aws/en/dev-tools/cli/reference/workspace-commands)

## When To Use This Path

Use this walkthrough if all of the following are true:

- you already have a Databricks App
- you want to preserve that app's URL, compute, and service principal
- you want to deploy Model Lens source into that existing app
- you already know the SQL warehouse, control-plane catalog, and control-plane schema

If you want Databricks to create a new app for Model Lens automatically, use [Deploy To A Workspace](./DEPLOY_TO_WORKSPACE.md) instead.

If your customer workspace already has an app, an approved SQL warehouse, and either an existing shared refresh job or a platform team that can reset one for you, use the concise runbook first:

- [Constrained Workspace Runbook](./CONSTRAINED_WORKSPACE_RUNBOOK.md)
- [Customer Manual Redeploy Guide](./CUSTOMER_MANUAL_REDEPLOY.md)

## The Two Existing-App Tracks

There are two distinct existing-app paths.
Choose the right one up front.

### Track A: Bundle-Bound Existing App

Use this only if the human operator can manage app resources.

You need:

- `Can manage` on the existing app
- `Can manage` on the SQL warehouse if the bundle or UI needs to add or update the `sql_warehouse` app resource

This track lets you:

- bind the bundle app resource to the existing app
- let the bundle continue managing the refresh workflow
- keep the repo root [app.yaml](../app.yaml) environment-neutral in git, then deploy app source with a literal `SQL_WAREHOUSE_ID` for that workspace

### Track B: Existing App Without App-Resource Management

Use this if either of the following is true:

- `bundle deploy` fails with `User does not have permission to grant permissions for added resource: sql_warehouse`
- the Apps UI fails with `User does not have permission to add resource sql-warehouse ... User needs MANAGE permission on the resource`

This track avoids the blocker entirely.

It does not require the operator to add or update a `sql_warehouse` app resource.
Instead, Model Lens now ships a helper script that:

- prepares a deployable source tree for the existing app
- writes an `app.yaml` with a literal `SQL_WAREHOUSE_ID`
- copies the current wheel into the source tree
- optionally writes a `refresh-job.json` payload for `databricks jobs create`

That script is:

- [prepare_existing_app_source.py](../scripts/prepare_existing_app_source.py)

Track B is the recommended path when the app already exists and the operator cannot manage app resources.

Track B is also the recommended path for the common customer-constrained setup where:

- the existing app must keep its current service principal
- the existing SQL warehouse is already approved
- the shared refresh job already exists, or the platform team wants to reuse the same job ID by resetting it to the current Model Lens contract
- direct `Run now` may or may not be allowed

Important:

- the generated `app.yaml` and the generated `refresh-job.json` must use the same `control_plane_catalog` and `control_plane_schema`
- on Hive Metastore workspaces, that often means `hive_metastore` plus a schema such as `model_lens_control_plane`
- do not rely on any repo default for the control-plane namespace; pass it explicitly every time

## What You Need

Collect these values first:

- existing app name
  for example `ml-drift-monitor`
- existing app resource ID
  from `databricks apps get <existing-app-name> -o json`
- SQL warehouse ID
- control-plane catalog
- control-plane schema
- a workspace path where you will upload the prepared source tree
  for example `/Workspace/Users/<your-email>/model-lens-existing-app`
- if using Lakebase:
  - Lakebase instance name
  - Lakebase database name
  - Lakebase schema

## Operator Permissions Needed

Before you start, the human operator performing the manual steps should have:

- `Can manage` on the existing Databricks App
- permissions to deploy or update the refresh workflow
- permissions to grant the app and workflow identities the required warehouse, Unity Catalog, and job ACLs

Only if you plan to use Track A:

- `Can manage` on the SQL warehouse if the bundle or Apps UI will add or edit the `sql_warehouse` app resource

## Step 1: Verify The Existing App Identity

Inspect the existing app:

```bash
databricks apps get <existing-app-name> -o json
```

Confirm all of the following:

- the app exists
- you intend to keep this exact app name
- you can identify the app object's unique ID
- the app has not been deleted and recreated recently

If you delete the app, Databricks deletes the app service principal too.
If you recreate the app, you get a new service principal and must regrant everything.

## Step 2: Start Or Reuse The Existing App Compute

Bring the existing app compute into `ACTIVE`:

```bash
databricks apps start <existing-app-name>
databricks apps get <existing-app-name>
```

Expected result:

- the app reports `ACTIVE`
- the app can accept a new `apps deploy`

## Step 3: Build And Validate Locally

From the repo root:

```bash
cd <repo-root>
python3 -m pytest
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

Expected result:

- tests pass
- a wheel exists in `dist/`

The manual helper uses that wheel both for the refresh job payload and for a ready-to-import workspace source tree.

## Step 4: Choose The Workflow Creation Path

At this point, choose one of the following.

### Track A: Bind The Existing App, Then Deploy The Bundle

Use this only if the operator can manage app resources.

The bundle app resource key is `control_plane`.
The existing app resource ID is the app object's ID from:

```bash
databricks apps get <existing-app-name> -o json
```

`bundle deployment bind` still needs the same required vars as `bundle deploy`, because the CLI parses the full bundle before binding resources.

Warehouse-only example:

```bash
databricks bundle deployment bind \
  -t warehouse_only \
  --var "app_name=<existing-app-name>" \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "refresh_node_type_id=<spark-node-type-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>" \
  control_plane <existing-app-resource-id>

databricks bundle deploy \
  -t warehouse_only \
  --var "app_name=<existing-app-name>" \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "refresh_node_type_id=<spark-node-type-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>"
```

After deploy:

- the workflow name is `<existing-app-name>-refresh`
- the staged app source is available under the bundle workspace path
- you can skip directly to Step 7 if the app already has a working `sql_warehouse` resource

If this deploy fails with a warehouse-resource permission error, stop using Track A and switch to Track B.

### Track B: Create The Workflow And App Source Without App-Resource Management

Use this when the operator cannot manage the app's `sql_warehouse` resource.

Generate a ready-to-import source tree and a refresh job payload:

```bash
python3 scripts/prepare_existing_app_source.py \
  --app-name <existing-app-name> \
  --sql-warehouse-id <sql-warehouse-id> \
  --control-plane-catalog <control-plane-catalog> \
  --control-plane-schema <control-plane-schema> \
  --output-dir /tmp/model-lens-existing-app \
  --workspace-source-path /Workspace/Users/<your-email>/model-lens-existing-app
```

What this creates under `/tmp/model-lens-existing-app`:

- `app.yaml` with a literal `SQL_WAREHOUSE_ID`
- `src/`
- `requirements.txt`
- `pyproject.toml`
- `dist/<built-wheel>.whl`
- `refresh-job.json`

At this point, the generated `app.yaml` uses:

- `SQL_WAREHOUSE_ID=<literal warehouse id>`
- `REFRESH_JOB_NAME=<existing-app-name>-refresh`

`REFRESH_JOB_ID` and `REFRESH_JOB_NAME` are app environment variables in that generated `app.yaml`. They are not configured from the onboarding wizard. To change refresh-job wiring later, regenerate or edit that `app.yaml`, re-import the prepared source tree, and redeploy the app.
If you want a separate large-tenant bootstrap/backfill lane later, the same generated `app.yaml` can also carry:

- `BOOTSTRAP_REFRESH_JOB_ID`
- `BOOTSTRAP_REFRESH_JOB_NAME`

Those optional values only affect direct bootstrap/backfill triggers. The default product shape remains one shared scheduled job.

The generated shared job payload now declares job-level parameters and pushes them into the wheel task's named arguments. The app triggers `jobs/run-now` with `job_parameters`, which is the override path Databricks currently honors for targeted bootstrap runs.
The generated Spark job cluster also defaults to `data_security_mode=USER_ISOLATION` so the workflow can access Unity Catalog tables. If your workspace policy requires it, change the generated payload to `SINGLE_USER` before `jobs create` or `jobs reset`.
The generated `refresh-job.json` still needs one workspace-specific cluster value before you create or reset the job: set `job_clusters[0].new_cluster.node_type_id` to an approved workspace node type, or preserve the existing job's approved node type when you mirror the payload in the UI.

If the app will monitor very large tables, set these environment variables in the generated `app.yaml` and shared refresh job before deploy:

- `REFRESH_SAMPLE_ROWS_PER_DAY`
- `REFRESH_MAX_ROWS_PER_WINDOW`
- `FEATURE_DETAIL_SAMPLE_ROWS_PER_DAY`
- `FEATURE_DETAIL_MAX_ROWS`
- `REFRESH_STALE_RUN_MINUTES`
- `MAX_PARALLEL_REFRESH_WORKERS`

Defaults are safe for many customers, but lowering them is the first lever to pull when a workspace has exceptionally wide or high-volume tables.
`REFRESH_STALE_RUN_MINUTES` controls when the shared scheduler automatically marks an abandoned `running` row failed so that monitor can be retried later.
`MAX_PARALLEL_REFRESH_WORKERS` controls only monitor-level concurrency in the shared job. Leave it low unless the cluster size and per-monitor source ranges are already known-safe for the tenant, because the scheduler still runs one scope per model at a time and each active worker still drives its own Spark range load plus Delta writes for the affected windows.

That means you can create the shared refresh workflow first and then later switch the app to `REFRESH_JOB_ID` if you want stricter wiring.

## Step 5: Upload The Prepared Source Tree To Workspace

This step is required for Track B.

Create the workspace directory:

```bash
databricks workspace mkdirs /Workspace/Users/<your-email>/model-lens-existing-app
```

Import the prepared local tree:

```bash
databricks workspace import-dir \
  /tmp/model-lens-existing-app \
  /Workspace/Users/<your-email>/model-lens-existing-app \
  --overwrite
```

Verify the workspace path exists:

```bash
databricks workspace get-status /Workspace/Users/<your-email>/model-lens-existing-app
```

## Step 6: Create Or Update The Refresh Workflow

### Track A

If Track A already deployed the bundle successfully, you should already have the refresh workflow.

Record the job ID:

```bash
databricks jobs list
```

Find `<existing-app-name>-refresh` and copy its numeric `job_id`.

### Track B

Create the workflow from the generated JSON payload:

```bash
databricks jobs create --json @/tmp/model-lens-existing-app/refresh-job.json
```

Expected result:

- a workflow named `<existing-app-name>-refresh` is created
- it is scheduled hourly by default as the shared pickup path for saved monitors
- it uses control-plane runtime state and cadence presets, so you do not need one Databricks workflow per model
- it writes one `refresh_runs` audit row per attempted monitor execution and rebuilds `quality_metrics` from persisted `daily_quality_profiles`, so bounded incremental runs do not shrink the monitor summary
- it reuses persisted `performance_bin_specs` for performance repair, so per-feature buckets stay stable across bootstrap and later incremental runs
- the environment dependencies point at the wheel under `/Workspace/Users/<your-email>/model-lens-existing-app/dist/...`

Then fetch the job ID:

```bash
databricks jobs list
```

Find `<existing-app-name>-refresh` and copy its numeric `job_id`.

If the workspace already has a shared refresh job and you want to keep that exact job ID, do not create a second workflow. Inspect the existing job first, compare it to the generated `refresh-job.json`, and use `jobs reset` if it is not already aligned.

The reusable job should match all of the following:

- one Python wheel task
- package name `model_lens`
- entry point `model-lens-refresh`
- task key `refresh_control_plane`
- job parameters for `warehouse_id`, `control_plane_catalog`, `control_plane_schema`, `scope`, and `model_key`
- `scope` defaulting to `scheduler`
- a UC-capable Spark access mode: `USER_ISOLATION` or `SINGLE_USER`
- the workspace wheel path under the generated source tree
- an active hourly schedule if you plan to rely on `scheduler_only`

#### Detailed Track B Refresh-Job Creation Walkthrough

If you want the exact sequence instead of the short form above, use this checklist.

1. Confirm the generated payload exists locally:

```bash
ls -l /tmp/model-lens-existing-app/refresh-job.json
```

2. Inspect the payload before creating anything:

```bash
cat /tmp/model-lens-existing-app/refresh-job.json
```

Verify all of the following in that JSON:

- `"name"` is `<existing-app-name>-refresh`
- `"package_name"` is `model_lens`
- `"entry_point"` is `model-lens-refresh`
- `"job_clusters"[0]."new_cluster"."node_type_id"` has been filled in with an approved workspace node type before create/reset
- `"--warehouse-id"` points at the correct SQL warehouse
- `"--catalog"` points at the correct control-plane catalog
- `"--schema"` points at the correct control-plane schema
- the first environment dependency points at the wheel in your workspace source tree, for example:
  `/Workspace/Users/<your-email>/model-lens-existing-app/dist/model_lens-0.1.0-py3-none-any.whl`

3. Create the job:

```bash
databricks jobs create --json @/tmp/model-lens-existing-app/refresh-job.json
```

The command returns JSON.
Record the numeric `job_id` from that output immediately.

4. Confirm the job now exists:

```bash
databricks jobs list
```

Find `<existing-app-name>-refresh` and verify the displayed `job_id` matches the one from `jobs create`.

5. Inspect the created job in full:

```bash
databricks jobs get <job-id> -o json
```

Verify:

- the task key is `refresh_control_plane`
- the task type is a Python wheel task
- the task parameters still contain the intended warehouse/catalog/schema values
- the environment dependency still points at the expected wheel in workspace storage
- the configured Run as identity is the one you intend to grant data access to

6. If the job already existed and you want to replace its settings instead of creating a second one, do not run `jobs create` again.
Wrap the generated settings into a reset request body:

```json
{
  "job_id": <job-id>,
  "new_settings": {
    ...contents of refresh-job.json...
  }
}
```

Save that as `/tmp/model-lens-existing-app/refresh-job-reset.json`, then run:

```bash
databricks jobs reset --json @/tmp/model-lens-existing-app/refresh-job-reset.json
```

If you want a concrete way to generate that wrapper file from the existing payload, use:

```bash
python3 - <<'PY'
import json
from pathlib import Path

job_id = <job-id>
settings = json.loads(Path("/tmp/model-lens-existing-app/refresh-job.json").read_text())
payload = {"job_id": job_id, "new_settings": settings}
Path("/tmp/model-lens-existing-app/refresh-job-reset.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
```

Then verify again:

```bash
databricks jobs get <job-id> -o json
```

7. Grant the job Run as identity access before testing a run:

- `CAN USE` on the SQL warehouse
- source data `USE CATALOG`, `USE SCHEMA`, `SELECT`
- control-plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`
- if Lakebase sync is enabled, the required Lakebase connection and write privileges

8. Trigger one manual test run before wiring the app to it:

```bash
databricks jobs run-now <job-id> --no-wait
```

Then inspect the run in the Jobs UI or with:

```bash
databricks jobs get-run <run-id> -o json
```

Expected result:

- the wheel environment resolves successfully
- the task reaches your code instead of failing during environment setup
- if no monitors exist yet, the run should still complete cleanly or skip without a packaging/configuration error

9. Once the job is confirmed healthy, rerun the helper so the app uses the numeric job ID:

```bash
python3 scripts/prepare_existing_app_source.py \
  --app-name <existing-app-name> \
  --sql-warehouse-id <sql-warehouse-id> \
  --control-plane-catalog <control-plane-catalog> \
  --control-plane-schema <control-plane-schema> \
  --refresh-job-id <job-id> \
  --output-dir /tmp/model-lens-existing-app \
  --workspace-source-path /Workspace/Users/<your-email>/model-lens-existing-app
```

10. Re-import the generated source tree so the deployed app will use `REFRESH_JOB_ID`:

```bash
databricks workspace import-dir \
  /tmp/model-lens-existing-app \
  /Workspace/Users/<your-email>/model-lens-existing-app \
  --overwrite
```

At that point, the app no longer depends on workflow name lookup.

#### UI Equivalent For Track B

If you prefer creating the job in the Databricks UI instead of `jobs create --json`, mirror the generated payload exactly:

1. Open `Workflows`.
2. Click `Create job`.
3. Set the job name to `<existing-app-name>-refresh`.
4. Add one task with:
   - task key: `refresh_control_plane`
   - task type: `Python wheel`
   - package name: `model_lens`
   - entry point: `model-lens-refresh`
5. Set named task parameters:
   - `warehouse-id`, `<sql-warehouse-id>`
   - `catalog`, `<control-plane-catalog>`
   - `schema`, `<control-plane-schema>`
   - `scope`, `scheduler`
6. Attach the task to one shared Spark job cluster:
   - choose the Spark runtime version that matches the bundle default or your workspace standard
   - set a UC-capable access mode: `USER_ISOLATION` by default, or `SINGLE_USER` if your workspace policy requires it
   - set a workspace-approved node type
   - start with `2` workers unless your platform team requires a different baseline
7. Add task libraries:
   - the wheel path under your workspace source tree
   - `dash>=2.18,<3.0`
   - `dash-bootstrap-components>=1.6,<2.0`
   - `databricks-sdk>=0.81,<1.0`
   - `databricks-sql-connector>=3.0,<4.0`
   - `numpy>=1.26,<3.0`
   - `pandas>=2.2,<3.0`
   - `plotly>=5.24,<6.0`
   - `psycopg[binary]>=3.2,<4.0`
   - `scikit-learn>=1.5,<2.0`
8. Set retries to:
   - max retries: `2`
   - min retry interval millis: `60000`
   - timeout seconds: `3600`
9. Save the job.
10. Record the numeric job ID from the UI.
11. Trigger one manual run from the UI and verify that the cluster starts, the wheel installs, and the task reaches your code.
12. Continue with the `--refresh-job-id <job-id>` regeneration step above so the app points at the exact workflow ID.

### Optional Hardening Step For Both Tracks

Once you know the job ID, regenerate the prepared source tree so the app uses `REFRESH_JOB_ID` explicitly instead of name lookup:

```bash
python3 scripts/prepare_existing_app_source.py \
  --app-name <existing-app-name> \
  --sql-warehouse-id <sql-warehouse-id> \
  --control-plane-catalog <control-plane-catalog> \
  --control-plane-schema <control-plane-schema> \
  --refresh-job-id <job-id> \
  --output-dir /tmp/model-lens-existing-app \
  --workspace-source-path /Workspace/Users/<your-email>/model-lens-existing-app

databricks workspace import-dir \
  /tmp/model-lens-existing-app \
  /Workspace/Users/<your-email>/model-lens-existing-app \
  --overwrite
```

This rewrites `app.yaml` so the onboarding flow triggers the exact workflow ID.

## Step 7: Deploy Model Lens Into The Existing App

### Track A

If you used the bundle, the app source is already staged under the bundle workspace directory:

- warehouse-only:
  `/Workspace/Users/<your-email>/.bundle/model-lens/warehouse_only/files`
- dev:
  `/Workspace/Users/<your-email>/.bundle/model-lens/dev/files`

Deploy from that staged path:

```bash
databricks apps deploy <existing-app-name> \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-lens/warehouse_only/files
```

### Track B

Deploy from the prepared workspace source tree:

```bash
databricks apps deploy <existing-app-name> \
  --source-code-path /Workspace/Users/<your-email>/model-lens-existing-app
```

After deployment:

```bash
databricks apps get <existing-app-name> -o json
```

Confirm:

- the existing app is still the same app instance
- the deployment is not stuck in a failed state
- the same app service principal is still attached

## Step 8: Grant Permissions To The Existing App Service Principal

Get the app identity from:

- the app Authorization tab in the Databricks Apps UI
- or `databricks apps get <existing-app-name> -o json`

Then grant the existing app service principal all of the following:

- `CAN USE` on the SQL warehouse
- source data:
  - `USE CATALOG`
  - `USE SCHEMA`
  - `SELECT`
- control plane:
  - `USE CATALOG`
  - `USE SCHEMA`
  - `SELECT`
  - `MODIFY`

If Setup should create missing objects from the UI, also grant:

- `CREATE TABLE` in the control-plane schema
- `CREATE SCHEMA` if the schema may not exist yet
- `CREATE CATALOG` only if you want the toggle to create the catalog

Important:

- `CAN MANAGE RUN` on the refresh workflow is required only if you want immediate bootstrap from the app UI
- if `CAN MANAGE RUN` is intentionally unavailable, `scheduler_only` remains a supported operating mode as long as the shared job already exists, is scheduled, and the app is wired to it
- if the generated `app.yaml` sets `REFRESH_JOB_ID`, grant `CAN_MANAGE_RUN` on that exact job ID; the Setup readiness card now points to that concrete grant when immediate bootstrap is unavailable
- if the generated `app.yaml` also sets `BOOTSTRAP_REFRESH_JOB_ID`, grant `CAN_MANAGE_RUN` on that second job only if you expect direct `Run First Refresh` acceleration through the optional bootstrap lane
- the app identity triggers the job
- the job's Run as identity performs the actual refresh work

## Step 9: Grant Permissions To The Refresh Job Identity

The refresh workflow must also have data-plane access.

Grant the job owner or Run as identity:

- `CAN USE` on the SQL warehouse
- source data:
  - `USE CATALOG`
  - `USE SCHEMA`
  - `SELECT`
- control plane:
  - `USE CATALOG`
  - `USE SCHEMA`
  - `SELECT`
  - `MODIFY`

If Lakebase sync is enabled, also grant the workflow identity:

- permission to resolve the Lakebase instance
- permission to connect to the Lakebase database
- write access to the target Lakebase schema used for the read model

## Step 10: Open The Existing App And Verify Runtime Wiring

Open the app in the browser.

In the `Monitor Settings` page, verify:

- `SQL_WAREHOUSE_ID` is populated
- `REFRESH_JOB_ID` or `REFRESH_JOB_NAME` is populated
- if you intentionally enabled the optional bootstrap lane, `BOOTSTRAP_REFRESH_JOB_ID` or `BOOTSTRAP_REFRESH_JOB_NAME` is populated too
- the control-plane namespace matches your intended destination
- if you are using `REFRESH_JOB_NAME`, it matches the real workflow you intend to trigger

In the `Setup` step, verify:

- `Control Plane Catalog` matches the value you set
- `Control Plane Schema` matches the value you set
- the permission checklist matches what you actually granted

If `SQL_WAREHOUSE_ID` is blank in Track B, the most common cause is that you deployed from the wrong workspace source path instead of the generated manual tree.

## Step 11: Run Setup And Create The First Monitor

From the app:

1. click `Setup Control Plane`
2. click `Validate Workspace Wiring`
3. confirm `Workspace Readiness` shows either `fully_ready` or `scheduler_only`
4. continue to `Discover`
5. enter the source table
6. optionally enter the labels table and MLflow metadata
7. confirm the inferred draft
8. click `Save Monitor`

Expected result:

- the monitor saves immediately
- the monitor is marked `pending bootstrap` in the control plane
- if the app has `CAN MANAGE RUN`, it reports that the shared refresh job was triggered
- if it does not, the shared hourly job still remains the default pickup path only if that workflow already exists and the app is wired to it through `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`
- if the workflow wiring is missing, ambiguous, paused, or unscheduled, setup now keeps onboarding blocked instead of allowing a dead-end monitor save
- if the monitor stays `pending bootstrap`, the `Monitor Settings` page exposes `Run First Refresh` so the operator can retry the selected monitor after fixing workflow wiring or permissions
- the cadence chosen during activation is stored with the monitor and can be edited later from the `Monitor Settings` page
- the Overview page shows the monitor after the workflow finishes

## Troubleshooting

### `bundle deploy` Fails With `User does not have permission to grant permissions for added resource: sql_warehouse`

You are on the wrong track.

Switch from Track A to Track B.
Do not keep retrying `bundle deploy` for the existing app unless an admin can manage that app resource.

### The Apps UI Says `User does not have permission to add resource sql-warehouse ...`

This is the same blocker as above.

Do not rely on adding the `sql_warehouse` app resource through the UI.
Use Track B and generate a source tree with a literal `SQL_WAREHOUSE_ID`.

### `Save Monitor` Only Saves The Config

Check:

- `REFRESH_JOB_ID` or `REFRESH_JOB_NAME` is set correctly
- the app service principal has `CAN MANAGE RUN` on the refresh job
- if using `REFRESH_JOB_NAME`, the configured value matches the real deployed workflow name for this app
- if you configured `BOOTSTRAP_REFRESH_JOB_ID` or `BOOTSTRAP_REFRESH_JOB_NAME`, verify that optional bootstrap lane separately; otherwise direct bootstrap falls back to the shared job by default

If `REFRESH_JOB_ID` is set and the app still reports scheduler-only mode, the expected operator fix is explicit: grant the app service principal `CAN_MANAGE_RUN` on that job ID.
If `BOOTSTRAP_REFRESH_JOB_ID` is set and `Run First Refresh` still cannot trigger directly, the expected operator fix is explicit: grant the app service principal `CAN_MANAGE_RUN` on that bootstrap job ID, or leave the monitor in `pending bootstrap` and let the shared scheduled job pick it up.
If the Setup card instead shows `Verification unavailable`, the app could not inspect the job ACLs with its current privileges. Direct trigger may still work, so test `Run First Refresh` or `jobs run-now` before assuming the permission grant is missing.

If `CAN MANAGE RUN` is intentionally unavailable, the app can still save the monitor and the shared hourly job can pick it up on its next run, but only if that workflow already exists and the app is wired to it through `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`. In that operating mode, treat the missing `Run now` permission as lost acceleration, not lost functionality.

### The App Opens But Cannot Query The Warehouse

Check:

- the app service principal still has `CAN USE` on the warehouse
- in Track B, the deployed app source path came from the generated manual tree
- the `Monitor Settings` page shows the expected `SQL_WAREHOUSE_ID`

### `apps deploy` Succeeds But The Wrong Source Tree Is Running

Check:

- the workspace path from Step 5 or Step 7 actually exists
- you deployed from the intended path:
  - bundle-staged path for Track A
  - generated manual workspace path for Track B
- if you regenerated the manual source tree with `--refresh-job-id`, you re-imported it before `apps deploy`

## Recommended Minimal Path When You Cannot Manage App Resources

If you want the shortest robust path for the exact constrained-user case:

1. keep the existing app
2. build the wheel locally
3. run [prepare_existing_app_source.py](../scripts/prepare_existing_app_source.py) with:
   - `--app-name`
   - `--sql-warehouse-id`
   - `--control-plane-catalog`
   - `--control-plane-schema`
   - `--output-dir`
   - `--workspace-source-path`
4. upload that directory with `databricks workspace import-dir`
5. create the refresh job from `refresh-job.json`, or reset the existing shared refresh job to that payload if the workspace must keep the same job ID
6. rerun the helper with `--refresh-job-id <job-id>` and re-import the directory
7. deploy the app from that workspace path with `databricks apps deploy`
8. grant the existing app service principal:
   - `CAN USE` on the warehouse
   - source-data `SELECT`
   - control-plane `SELECT` and `MODIFY`
   - `CAN MANAGE RUN` on the job only if you want immediate UI-triggered bootstrap; otherwise leave the workspace in supported `scheduler_only` mode
9. grant the refresh job identity the actual data/control-plane privileges

That keeps the existing app identity, avoids app-resource management entirely, and still gives you the current non-blocking Model Lens onboarding flow.
