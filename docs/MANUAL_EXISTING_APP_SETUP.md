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

As of `2026-04-01T11:04:18-0600`, this matches the current Databricks app authorization model in the official Databricks docs:

- [Configure authorization in a Databricks app](https://docs.databricks.com/gcp/en/dev-tools/databricks-apps/auth)
- [Add resources to a Databricks app](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/resources)
- [Add a SQL warehouse resource to a Databricks app](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/sql-warehouse)
- [Configure permissions for a Databricks app](https://docs.databricks.com/gcp/en/dev-tools/databricks-apps/permissions)
- [Trigger a single job run](https://docs.databricks.com/aws/en/jobs/run-now)

## When To Use This Path

Use this walkthrough if all of the following are true:

- you already have a Databricks App
- you want to preserve that app's URL, compute, and service principal
- you want to deploy Model Lens source into that existing app
- you are comfortable granting warehouse, job, and Unity Catalog permissions manually

If you want Databricks to create a new app for Model Lens automatically, use [Deploy To A Workspace](/Users/volo.vragov/Desktop/work/model-lens/docs/DEPLOY_TO_WORKSPACE.md) instead.

## What You Need

Collect these values first:

- existing app name:
  for example `geocomply-drift-monitor`
- SQL warehouse ID for the app
- control-plane catalog:
  for example `model_observability`
- control-plane schema:
  for example `control_plane`
- refresh job ID or refresh job name
- source data catalog/schema/table names
- if using Lakebase:
  - Lakebase instance name
  - Lakebase database name
  - Lakebase schema

## Overview

The manual path has four separate concerns:

1. keep the existing app and its service principal
2. make sure the refresh workflow exists
3. make sure the app runtime has the right environment variables and app resources
4. grant the app identity and job identity the required permissions

Treat those as separate checks.
The app can start successfully and still fail later if any one of the four is missing.

## Step 1: Verify The Existing App Identity

Inspect the existing app:

```bash
databricks apps get <existing-app-name> -o json
```

Confirm all of the following:

- the app exists
- you intend to keep this exact app name
- the app can be started
- the app has not been deleted and recreated recently

If you delete the app, Databricks deletes the app service principal too.
If you recreate the app, you get a new service principal and must regrant everything.

## Step 2: Start Or Reuse The Existing App Compute

Bring the existing app compute into `ACTIVE`:

```bash
databricks apps start <existing-app-name>
databricks apps get <existing-app-name>
```

Do this before deploying source code.

Expected result:

- the app reports `ACTIVE`
- the app can accept a new `apps deploy`

## Step 3: Build The Model Lens Package

From the repo root:

```bash
cd /Users/volo.vragov/Desktop/work/model-lens
python3 -m pytest
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

Expected result:

- tests pass
- the wheel is built into `dist/`

## Step 4: Make Sure The Refresh Workflow Exists

Model Lens now triggers the initial refresh asynchronously.
That means the app must know which job to call.

You have two supported options:

- set `REFRESH_JOB_ID` directly
- or set `REFRESH_JOB_NAME` and let the app resolve the job by name

### Option A: Reuse An Existing Refresh Job

If you already have the refresh workflow:

- record the job ID
- confirm the job still runs with the intended Run as identity

### Option B: Deploy The Refresh Job With The Bundle

If you want the repo to create or update the workflow, deploy the bundle first.

Warehouse-only example:

```bash
databricks bundle deploy \
  -t warehouse_only \
  --var "app_name=<existing-app-name>" \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>"
```

This is still compatible with the manual app path.
The important part is that you do not delete the existing app instance.

If your existing app name is not `model-lens`, using `app_name=<existing-app-name>` keeps the workflow naming aligned with the app.

## Step 5: Decide How The App Gets `SQL_WAREHOUSE_ID`

The current [app.yaml](/Users/volo.vragov/Desktop/work/model-lens/app.yaml) expects:

```yaml
- name: SQL_WAREHOUSE_ID
  valueFrom: sql_warehouse
```

That means the existing app must expose a SQL warehouse app resource keyed as `sql_warehouse`.

### Preferred Manual Setup

In the Databricks Apps UI for the existing app:

1. open the app
2. go to `Configure`
3. add a `SQL warehouse` resource
4. choose the target warehouse
5. set permission to `CAN USE`
6. set the resource key to exactly `sql_warehouse`

If you do this, the current `app.yaml` can stay as-is.

### Fallback Manual Setup

If you do not want to use an app resource for the warehouse, change [app.yaml](/Users/volo.vragov/Desktop/work/model-lens/app.yaml) before deployment:

```yaml
- name: SQL_WAREHOUSE_ID
  value: "<sql-warehouse-id>"
```

Use this only if you are intentionally hardcoding the warehouse ID.
The resource-based path is cleaner and more portable.

## Step 6: Set The Required App Environment Variables

The existing app must provide these values to Model Lens:

- `CONTROL_PLANE_CATALOG`
- `CONTROL_PLANE_SCHEMA`
- `SQL_WAREHOUSE_ID`
- `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`

Optional but commonly useful:

- `LAKEBASE_INSTANCE_NAME`
- `LAKEBASE_DATABASE_NAME`
- `LAKEBASE_SCHEMA`
- `GENIE_SPACE_ID`

The repo default [app.yaml](/Users/volo.vragov/Desktop/work/model-lens/app.yaml) already contains these environment variables.
For a manual existing-app deployment, update the values before source deploy if needed.

### Recommended Manual Values

For a warehouse-only manual setup:

```yaml
env:
  - name: CONTROL_PLANE_CATALOG
    value: <control-plane-catalog>
  - name: CONTROL_PLANE_SCHEMA
    value: <control-plane-schema>
  - name: SQL_WAREHOUSE_ID
    valueFrom: sql_warehouse
  - name: REFRESH_JOB_ID
    value: "<existing-refresh-job-id>"
  - name: REFRESH_JOB_NAME
    value: ""
```

If you prefer job-name resolution:

```yaml
env:
  - name: REFRESH_JOB_ID
    value: ""
  - name: REFRESH_JOB_NAME
    value: "<existing-refresh-job-name>"
```

Using `REFRESH_JOB_ID` is safer when multiple jobs have similar names.

## Step 7: Stage Source Code For Manual App Deploy

You need a workspace path that contains the Model Lens source code.

The simplest path is to use the bundle output path after `databricks bundle deploy`:

- warehouse-only:
  `/Workspace/Users/<your-email>/.bundle/model-lens/warehouse_only/files`
- dev:
  `/Workspace/Users/<your-email>/.bundle/model-lens/dev/files`

If you are not using the bundle to stage files, you can upload the repo manually to a workspace path, but the path must contain:

- `app.yaml`
- `src/model_lens/...`
- the built wheel in `dist/` if your workflow deployment path still depends on it

The bundle-staged workspace path is usually the least error-prone option.

## Step 8: Deploy Model Lens Into The Existing App

Deploy source into the existing app instance:

```bash
databricks apps deploy <existing-app-name> \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-lens/warehouse_only/files
```

Or, if using the `dev` target:

```bash
databricks apps deploy <existing-app-name> \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-lens/dev/files
```

This updates the existing app compute.
It does not create a new app identity as long as you keep the same app.

After deployment:

```bash
databricks apps get <existing-app-name> -o json
```

Confirm the app still points to the same app instance and is not stuck in a failed deployment state.

## Step 9: Grant Permissions To The Existing App Service Principal

Get the app identity from:

- the app Authorization tab in the Databricks Apps UI
- or `databricks apps get <existing-app-name> -o json`

Then grant the existing app service principal all of the following:

- `CAN USE` on the SQL warehouse
- `CAN MANAGE RUN` on the refresh workflow
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

- `CAN MANAGE RUN` is required because the onboarding flow now triggers the initial refresh asynchronously
- the app identity triggers the job
- the job's Run as identity performs the actual refresh work

## Step 10: Grant Permissions To The Refresh Job Identity

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

## Step 11: Open The Existing App And Verify Runtime Wiring

Open the app in the browser.

In the `Setup` step, verify:

- `Control Plane Catalog` matches the value you set
- `Control Plane Schema` matches the value you set
- if Lakebase is in use, the Lakebase fields are correct
- the permission checklist matches what you actually granted

In the `Reference` page, verify:

- `SQL_WAREHOUSE_ID` is populated
- `REFRESH_JOB_ID` or `REFRESH_JOB_NAME` is populated
- the control-plane namespace matches your intended destination

If `SQL_WAREHOUSE_ID` is blank, the usual cause is:

- the existing app is missing a SQL warehouse resource keyed `sql_warehouse`
- or `app.yaml` was not updated to a literal `value`

If the initial refresh cannot be triggered, the usual cause is:

- missing `CAN MANAGE RUN` on the refresh job
- or missing `REFRESH_JOB_ID` / `REFRESH_JOB_NAME`

## Step 12: Run Setup And Create The First Monitor

From the app:

1. click `Setup Control Plane`
2. continue to `Discover`
3. enter the source table
4. optionally enter the labels table and MLflow metadata
5. confirm the inferred draft
6. click `Save Monitor And Trigger Refresh`

Expected result:

- the monitor saves immediately
- the app reports that the refresh job was triggered
- the Overview page shows the monitor after the workflow finishes

## Troubleshooting

### The Existing App Started But The UI Cannot Query The Warehouse

Check:

- the existing app still has the SQL warehouse resource
- the resource key is `sql_warehouse`
- the app service principal still has `CAN USE`

### The App Opens But `Save Monitor And Trigger Refresh` Only Saves The Config

Check:

- `REFRESH_JOB_ID` or `REFRESH_JOB_NAME` is set correctly
- the app service principal has `CAN MANAGE RUN` on the refresh job

### The App Cannot See Control-Plane Tables

Check:

- `CONTROL_PLANE_CATALOG`
- `CONTROL_PLANE_SCHEMA`
- `USE CATALOG`, `USE SCHEMA`, `SELECT`, and `MODIFY` on the control-plane namespace

### The App Service Principal Changed Unexpectedly

That usually means the app was deleted and recreated.
If that happened:

- treat it as a new app identity
- regrant warehouse, job, and Unity Catalog permissions
- recheck the SQL warehouse app resource binding

## Recommended Minimal Manual Pattern

If you want the shortest robust manual path:

1. keep the existing app
2. add a SQL warehouse app resource keyed `sql_warehouse`
3. deploy or identify the refresh job
4. set `REFRESH_JOB_ID`
5. deploy Model Lens source into the existing app
6. grant the existing app service principal:
   - `CAN USE` on the warehouse
   - `CAN MANAGE RUN` on the job
   - source-data `SELECT`
   - control-plane `SELECT` and `MODIFY`
7. grant the refresh job identity the actual data/control-plane privileges

That gives you the existing app compute, the existing app identity, and the current non-blocking Model Lens onboarding flow without introducing any extra tooling.
