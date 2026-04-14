# Customer Manual Redeploy Guide

Use this guide when your Databricks workspace already has:

- an existing Databricks App that must keep its current URL and app service principal
- an existing SQL warehouse that Model Lens should use
- an existing shared refresh job, or permission to create or reset one manually

This guide does **not** use `ai_dev_kit`.

Keep customer-specific values out of this shared file.
If you need a workspace-specific copy for one customer, generate it outside the repo or in a private system that is not committed back into git.

## What This Guide Covers

This path keeps the existing app identity and deploys Model Lens into that existing app by:

1. generating a deployable app source tree with literal workspace values
2. creating or resetting one shared refresh job
3. pointing the app at that exact refresh job by `REFRESH_JOB_ID`
4. deploying the generated source tree into the existing app

## What You Need

Please collect these values before starting:

- existing app name
- SQL warehouse ID
- control-plane catalog
- control-plane schema
- approved Spark node type for the refresh job cluster
- workspace path where the generated source tree will be uploaded
  - example: `/Workspace/Users/<your-email>/model-lens-existing-app`

If you already have a shared Model Lens refresh job, also note:

- refresh job ID

## Step 1: Build The Current Repo

From the latest Model Lens repo checkout:

```bash
cd <repo-root>
python3 -m pytest
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

Expected result:

- tests pass
- a wheel is created in `dist/`

## Step 2: Generate A Deployable Source Tree

Generate a manual source tree for the existing app:

```bash
python3 scripts/prepare_existing_app_source.py \
  --app-name <existing-app-name> \
  --sql-warehouse-id <sql-warehouse-id> \
  --control-plane-catalog <control-plane-catalog> \
  --control-plane-schema <control-plane-schema> \
  --output-dir /tmp/model-lens-existing-app \
  --workspace-source-path /Workspace/Users/<your-email>/model-lens-existing-app
```

This creates:

- `app.yaml`
- `dist/<wheel>.whl`
- `refresh-job.json`

## Step 3: Edit The Generated Refresh Job Payload

If you are creating a new shared refresh job, or if your platform team explicitly wants to replace the job's compute settings, open:

- `/tmp/model-lens-existing-app/refresh-job.json`

Update the Spark cluster settings that are always workspace-specific:

- set `job_clusters[0].new_cluster.node_type_id` to an approved workspace node type
- keep `job_clusters[0].new_cluster.data_security_mode` as `USER_ISOLATION`
  - if your workspace policy requires it, change it to `SINGLE_USER`

Do not skip this step. The generated payload does not know your workspace-approved node type.

If you are reusing an existing approved shared job, do **not** blindly overwrite its compute settings.
Instead:

- keep the existing approved compute configuration on that job
- update the Model Lens task, libraries, parameters, and schedule to match the generated payload
- only change the compute section if your platform team tells you to

If the existing job uses `new_cluster`, the job's Run as identity must be allowed to launch that job cluster.
If the workspace does not allow that, the job must instead use an approved existing cluster or a policy-approved compute configuration.

## Step 4: Create Or Reset The Shared Refresh Job

### Option A: Reuse An Existing Shared Refresh Job

If you already have a shared Model Lens refresh job, reuse that job ID instead of creating a second one.

Inspect the job:

```bash
databricks jobs get <job-id> -o json
```

The job should match the generated payload in all important ways:

- one Python wheel task
- package name `model_lens`
- entry point `model-lens-refresh`
- task key `refresh_control_plane`
- job parameters for warehouse/catalog/schema/scope/model_key
- an hourly schedule
- a UC-capable Spark access mode (`USER_ISOLATION` or `SINGLE_USER`)

The safest way to reuse an existing approved job is:

- preserve its approved compute settings
- update only the Model Lens task contract and code payload

That means the existing job should end up with:

- one Python wheel task
- package name `model_lens`
- entry point `model-lens-refresh`
- task key `refresh_control_plane`
- the latest wheel/library path from the generated source tree
- current job parameters for warehouse/catalog/schema/scope/model_key
- an hourly schedule if you want `scheduler_only` pickup

If the job is out of date, reset it.

Before using `jobs reset`, make sure the reset payload preserves the compute settings that are actually allowed in your workspace. Do not assume the generated `new_cluster` section is acceptable for a locked-down workspace.

Create a reset wrapper file:

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

Then reset the job:

```bash
databricks jobs reset --json @/tmp/model-lens-existing-app/refresh-job-reset.json
```

### Option B: Create A New Shared Refresh Job

If no shared Model Lens refresh job exists yet:

```bash
databricks jobs create --json @/tmp/model-lens-existing-app/refresh-job.json
```

After create or reset, record the numeric `job_id`.

## Step 5: Regenerate The App Source Tree With The Exact Job ID

Regenerate the source tree so the app points at the exact refresh job ID:

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

This is recommended over job-name lookup.

## Step 6: Upload The Generated Source Tree

```bash
databricks workspace mkdirs /Workspace/Users/<your-email>/model-lens-existing-app

databricks workspace import-dir \
  /tmp/model-lens-existing-app \
  /Workspace/Users/<your-email>/model-lens-existing-app \
  --overwrite
```

## Step 7: Deploy Into The Existing App

```bash
databricks apps start <existing-app-name>

databricks apps deploy <existing-app-name> \
  --source-code-path /Workspace/Users/<your-email>/model-lens-existing-app
```

## Step 8: Grant Permissions

### App Service Principal

The existing app service principal needs:

- `CAN_USE` on the SQL warehouse
- source data:
  - `USE CATALOG`
  - `USE SCHEMA`
  - `SELECT`
- control plane:
  - `USE CATALOG`
  - `USE SCHEMA`
  - `SELECT`
  - `MODIFY`

Optional:

- `CAN_MANAGE` on the shared refresh job
  - recommended if you want operators to change the shared wake interval from `Monitor Settings -> Admin` and keep full in-app job management available
- `CAN_MANAGE_RUN` on the shared refresh job
  - minimum direct-trigger grant if you only want the app to trigger the first refresh immediately from the UI
  - not required if `scheduler_only` mode is acceptable

### Refresh Job Run As Identity

The refresh job Run as identity needs:

- `CAN_USE` on the SQL warehouse
- source data:
  - `USE CATALOG`
  - `USE SCHEMA`
  - `SELECT`
- control plane:
  - `USE CATALOG`
  - `USE SCHEMA`
  - `SELECT`
  - `MODIFY`

## Step 9: Open The App And Verify Wiring

After deploy, open the app and verify:

- `SQL_WAREHOUSE_ID` is populated
- `REFRESH_JOB_ID` is populated
- the control-plane catalog and schema are correct

Then:

1. click `Setup Control Plane`
2. click `Validate Workspace Wiring`

Valid readiness outcomes:

- `fully_ready`
- `scheduler_only`

Both are supported.

## Step 10: Save The First Monitor

Expected outcomes:

- if the app service principal has `CAN_MANAGE_RUN`, the app can trigger bootstrap immediately
- if it does not, the monitor can still be saved and remain `pending bootstrap`
- in `scheduler_only` mode, the shared scheduled refresh job is the supported pickup path
- for large tables, `pending bootstrap` / `Computing/Pending` can persist until the first refresh job actually finishes; that is expected during the initial run

## Important Notes

- After redeploying a newer Model Lens build into an existing workspace, run `Setup Control Plane` once so additive schema migrations are applied.
- If you reuse an existing shared refresh job, do not assume an older job definition is still compatible. Reset it to the generated `refresh-job.json` contract if there is any doubt.
- Prefer `REFRESH_JOB_ID` over workflow name lookup.
- if the shared refresh job already ran successfully after redeploy, do not manually reinstall the wheel again; do that only when the workspace relies on cluster-scoped libraries or the job cannot resolve the uploaded workspace wheel.
- delete and archive actions in the app operate by `model_key`, so verify the displayed `model_key` before using monitor lifecycle actions.

## Troubleshooting

### The app opens but cannot read data

Usually one of:

- the deployed `app.yaml` does not contain the intended `SQL_WAREHOUSE_ID`
- the app service principal lacks `CAN_USE` on that warehouse
- the app was deployed from the wrong workspace source path

### A monitor saves but never refreshes

Check:

- `REFRESH_JOB_ID` points at the correct job
- the job schedule is enabled
- the refresh job Run as identity has the required data-plane permissions
- the refresh job Run as identity also has the required compute permission for the job's configured compute mode
- the job matches the current generated payload
- if immediate bootstrap was expected, the app service principal has `CAN_MANAGE_RUN` on that exact job

If the run fails with a cluster-preparation or cluster-permission error, that is a job-compute issue, not an app-deploy issue. In that case, keep the existing approved compute on the reused job or ask the platform team to update the job to an approved compute mode.

### The app shows `Verification unavailable`

This means the app could not inspect workflow ACLs with its current privileges.
It does **not** prove that `Run now` is impossible.

Next checks:

- test `Run First Refresh` from the app
- or test `databricks jobs run-now <job-id>`

### The existing job seems old or inconsistent

Reset it from the generated `refresh-job.json` payload instead of keeping the older settings.
