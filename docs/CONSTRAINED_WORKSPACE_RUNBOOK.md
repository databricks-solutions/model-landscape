# Constrained Workspace Runbook

Use this runbook when the customer workspace already has:

- an existing Databricks App that must keep its current URL and app service principal
- an existing SQL warehouse that Model Lens must use
- an existing shared refresh job, or a platform team that can create or reset one for you

This is the recommended operator path for customer workspaces where the deployer cannot freely manage Databricks App resources.

## What This Path Assumes

- you are not letting the bundle create a new app
- you are not relying on Databricks App `sql_warehouse` resource management
- you will deploy app source into the existing app with a literal `SQL_WAREHOUSE_ID`
- you will point the app at one shared refresh workflow by `REFRESH_JOB_ID` whenever possible

## What Is Required Versus Optional

Required:

- keep the existing app
- know the existing SQL warehouse ID
- know the control-plane catalog and schema
- have one shared refresh workflow that matches the current Model Lens job contract
- grant the app service principal `CAN_USE` on the SQL warehouse
- grant the refresh job Run as identity access to the SQL warehouse, source tables, and control-plane tables

Optional:

- `CAN_MANAGE_RUN` on the refresh workflow for the app service principal
  - needed only if you want immediate `Run First Refresh` / activation-time bootstrap from the UI
  - not required if `scheduler_only` mode is acceptable and the shared job is already scheduled

## Fastest Supported Path

1. Build the wheel locally:

```bash
cd <repo-root>
python3 -m pytest
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

2. Generate a deployable source tree with literal workspace values:

```bash
python3 scripts/prepare_existing_app_source.py \
  --app-name <existing-app-name> \
  --sql-warehouse-id <sql-warehouse-id> \
  --control-plane-catalog <control-plane-catalog> \
  --control-plane-schema <control-plane-schema> \
  --output-dir /tmp/model-lens-existing-app \
  --workspace-source-path /Workspace/Users/<your-email>/model-lens-existing-app
```

3. Upload that generated tree:

```bash
databricks workspace mkdirs /Workspace/Users/<your-email>/model-lens-existing-app
databricks workspace import-dir \
  /tmp/model-lens-existing-app \
  /Workspace/Users/<your-email>/model-lens-existing-app \
  --overwrite
```

4. If a shared refresh job already exists, reuse it by resetting it to the generated payload instead of creating a second job.

Before `jobs create` or `jobs reset`, fill in the Spark cluster settings that are intentionally workspace-specific:

- set `job_clusters[0].new_cluster.node_type_id` to an approved workspace node type
- keep or set `job_clusters[0].new_cluster.data_security_mode` to `USER_ISOLATION` or `SINGLE_USER`

The reusable job must match all of the following:

- one Python wheel task
- package name `model_lens`
- entry point `model-lens-refresh`
- task key `refresh_control_plane`
- job parameters for `warehouse_id`, `control_plane_catalog`, `control_plane_schema`, `scope`, and `model_key`
- `scope` defaulting to `scheduler`
- a UC-capable Spark access mode: `USER_ISOLATION` or `SINGLE_USER`
- the workspace wheel path under the generated source tree
- an active hourly schedule if you expect `scheduler_only` pickup

Inspect the existing job:

```bash
databricks jobs get <job-id> -o json
```

If it does not match the generated payload, reset it:

```bash
databricks jobs reset --json @/tmp/model-lens-existing-app/refresh-job-reset.json
```

If no shared job exists yet, create one from the generated payload:

```bash
databricks jobs create --json @/tmp/model-lens-existing-app/refresh-job.json
```

5. Once the job ID is known, regenerate the source tree so the app points at that exact job ID:

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

6. Deploy the generated source tree into the existing app:

```bash
databricks apps start <existing-app-name>
databricks apps deploy <existing-app-name> \
  --source-code-path /Workspace/Users/<your-email>/model-lens-existing-app
```

7. Grant permissions.

App service principal:

- `CAN_USE` on the SQL warehouse
- source data `USE CATALOG`, `USE SCHEMA`, `SELECT`
- control plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`
- `CAN_MANAGE_RUN` on the shared refresh workflow only if you want immediate bootstrap from the UI

Refresh job Run as identity:

- `CAN_USE` on the SQL warehouse
- source data `USE CATALOG`, `USE SCHEMA`, `SELECT`
- control plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`

8. Open the app and verify:

- `SQL_WAREHOUSE_ID` is populated
- `REFRESH_JOB_ID` is populated
- the control-plane namespace matches the intended workspace namespace
- `Validate Workspace Wiring` reports either `fully_ready` or `scheduler_only`

9. Run `Setup Control Plane`, then save a monitor.

Expected outcomes:

- if the app service principal has `CAN_MANAGE_RUN`, the app can trigger the shared refresh job immediately
- if it does not, the app can still save the monitor and leave it in `pending bootstrap`
- in `scheduler_only` mode, the scheduled shared workflow is the supported pickup path

## Scheduler-Only Mode Is Supported

Treat `scheduler_only` as a valid operating mode, not a deployment failure.

It is acceptable when all of the following are true:

- the shared refresh job exists
- the job schedule is enabled
- the app is wired to that job through `REFRESH_JOB_ID` or, less preferably, `REFRESH_JOB_NAME`
- the job Run as identity has the required data-plane permissions

In this mode:

- the onboarding save still succeeds
- the monitor stays `pending bootstrap` until the next shared-job pickup
- `Run First Refresh` acceleration is unavailable until `CAN_MANAGE_RUN` is granted

## The Most Common Failure Modes

### App opens but cannot query data

Usually one of:

- `SQL_WAREHOUSE_ID` is missing from the deployed `app.yaml`
- the app service principal lacks `CAN_USE` on that warehouse
- the deployed source path was not the generated manual tree

### Monitor saves but never refreshes

Check all of:

- `REFRESH_JOB_ID` points at the intended job
- the job schedule is unpaused
- the job still matches the current Model Lens wheel-task contract
- the job Run as identity has data-plane access
- if you expected direct UI bootstrap, the app service principal has `CAN_MANAGE_RUN` on that exact job ID

### Setup says `Verification unavailable`

This means the app could not inspect the workflow ACLs with its current privileges.
It does not prove that `Run now` is impossible.

Next checks:

- test `Run First Refresh` from the app
- or test `databricks jobs run-now <job-id>`

### Existing job is old or drifted from the repo contract

Do not keep the old settings and hope for compatibility.
Reset the job from the generated `refresh-job.json` or mirror that payload in the UI.

## Use This Together With

- [Manual Setup With An Existing Databricks App](./MANUAL_EXISTING_APP_SETUP.md)
- [Customer Manual Redeploy Guide](./CUSTOMER_MANUAL_REDEPLOY.md)
- [Workspace Smoke Test](./WORKSPACE_SMOKE_TEST.md)
