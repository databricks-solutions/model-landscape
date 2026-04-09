# GeoComply Manual Redeploy Guide

This is the recommended manual redeploy path for the current GeoComply workspace.

It assumes all of the following are already true:

- existing app name: `ml-drift-monitor`
- existing SQL warehouse ID: `ad699a620534cfc9`
- control-plane catalog: `gc_prod_mlproduct`
- control-plane schema: `mlp_rsch`
- existing shared refresh job ID: `530788144136151`
- workspace source path: `/Workspace/Users/faramarz.jabbarvaziri@geocomply.com/model-lens-existing-app`

This guide does **not** use `ai_dev_kit`.

## Important For GeoComply

Do **not** create a second refresh job unless your platform team asks for that.

Use the existing shared job:

- `530788144136151`

Also, do **not** assume the generated `new_cluster` section should replace the job's current compute settings.

Why:

- the app uses the SQL warehouse for UI/control-plane queries
- the large bootstrap refresh uses the Databricks refresh workflow
- if that workflow is configured with `new_cluster`, the job's Run as identity must be allowed to launch job compute
- GeoComply already hit `PERMISSION_DENIED: You are not authorized to create clusters`, so compute settings must be preserved or corrected deliberately

## Recommended Path

1. keep the existing app
2. keep the existing shared refresh job ID
3. update the job's Model Lens task/code wiring
4. preserve the existing approved compute settings on that job unless the platform team tells you otherwise
5. point the app at `REFRESH_JOB_ID=530788144136151`

## Step 1: Build The Current Repo

```bash
cd /Users/faramarz.jabbarvaziri/Documents/codebases/model-lens
python3 -m pytest
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

## Step 2: Generate The Deployable Source Tree

```bash
python3 scripts/prepare_existing_app_source.py \
  --app-name ml-drift-monitor \
  --sql-warehouse-id ad699a620534cfc9 \
  --control-plane-catalog gc_prod_mlproduct \
  --control-plane-schema mlp_rsch \
  --refresh-job-id 530788144136151 \
  --output-dir /tmp/model-lens-existing-app \
  --workspace-source-path /Workspace/Users/faramarz.jabbarvaziri@geocomply.com/model-lens-existing-app
```

This generates:

- `app.yaml`
- `dist/<wheel>.whl`
- `refresh-job.json`

## Step 3: Upload The Source Tree

```bash
databricks workspace mkdirs /Workspace/Users/faramarz.jabbarvaziri@geocomply.com/model-lens-existing-app

databricks workspace import-dir \
  /tmp/model-lens-existing-app \
  /Workspace/Users/faramarz.jabbarvaziri@geocomply.com/model-lens-existing-app \
  --overwrite
```

## Step 4: Update The Existing Refresh Job

Inspect the current job first:

```bash
databricks jobs get 530788144136151 -o json
```

For this workspace, the important rule is:

- keep the current approved compute settings unless the platform team explicitly changes them

That means:

- if the job already points at an approved existing cluster, keep that
- if the job already has a policy-approved job-cluster shape, keep that
- do **not** blindly replace the job with the generated `new_cluster` section unless the Run as identity is allowed to launch that compute

What should be updated on the existing job:

- task type: Python wheel task
- task key: `refresh_control_plane`
- package name: `model_lens`
- entry point: `model-lens-refresh`
- task/job parameters for:
  - warehouse ID
  - control-plane catalog
  - control-plane schema
  - `scope`
  - `model_key`
- wheel/library path to the newly uploaded wheel under:
  - `/Workspace/Users/faramarz.jabbarvaziri@geocomply.com/model-lens-existing-app/dist/...`
- hourly schedule, if GeoComply wants `scheduler_only` pickup

If your platform team prefers CLI reset, use it only after preserving the allowed compute section.
If not, updating the job in the Databricks UI is safer for this workspace.

## Step 5: Deploy Into The Existing App

```bash
databricks apps start ml-drift-monitor

databricks apps deploy ml-drift-monitor \
  --source-code-path /Workspace/Users/faramarz.jabbarvaziri@geocomply.com/model-lens-existing-app
```

## Step 6: Permissions

### App service principal

Needs:

- `CAN_USE` on SQL warehouse `ad699a620534cfc9`
- source data `USE CATALOG`, `USE SCHEMA`, `SELECT`
- control plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`

Optional:

- `CAN_MANAGE_RUN` on job `530788144136151`
  - only needed if the app should trigger bootstrap immediately from the UI
  - not required if `scheduler_only` is acceptable

### Refresh job Run as identity

Needs:

- `CAN_USE` on SQL warehouse `ad699a620534cfc9`
- source data `USE CATALOG`, `USE SCHEMA`, `SELECT`
- control plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`
- permission to use the job's configured compute mode

That last point is critical:

- if the job uses `new_cluster`, the Run as identity must be allowed to launch that job cluster
- if the workspace does not allow that, the job must use approved existing compute instead

## Step 7: Open The App And Verify

In the app, verify:

- `SQL_WAREHOUSE_ID` is populated
- `REFRESH_JOB_ID` is `530788144136151`
- `Control Plane Catalog` is `gc_prod_mlproduct`
- `Control Plane Schema` is `mlp_rsch`

Then:

1. click `Setup Control Plane`
2. click `Validate Workspace Wiring`

Valid outcomes:

- `fully_ready`
- `scheduler_only`

## Step 8: Save A Monitor

Expected behavior:

- if the app service principal has `CAN_MANAGE_RUN`, the app can trigger the job immediately
- if not, the monitor can still be saved and left in `pending bootstrap`
- in `scheduler_only` mode, the shared refresh job should pick it up on schedule

## If The Job Fails Again

### Error: `You are not authorized to create clusters`

That means the job is still trying to launch a job cluster that its Run as identity is not allowed to create.

For GeoComply, the next action is:

- do **not** change the app deploy path
- do **not** change the SQL warehouse
- update the existing job's compute configuration with the platform/admin team

The fix is one of:

1. preserve or switch to an approved existing cluster
2. use a policy-approved compute configuration that the Run as identity is allowed to launch
3. grant the Run as identity the required permission to launch the approved job cluster

## Short Version

For GeoComply, the correct shape is:

- existing app
- existing warehouse
- existing job ID `530788144136151`
- updated Model Lens wheel/task config
- preserved approved compute
- app wired by `REFRESH_JOB_ID`

The app deploy is not the current blocker.
The remaining blocker is the refresh job's compute configuration and permissions.
