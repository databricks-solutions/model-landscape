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
9. run the focused rollout gates for Overview, severe drift, and same-day detail views

## Prerequisites

You need:

- Databricks CLI auth configured for the target workspace
- a SQL warehouse you can use
- privileges to create or run a Spark-capable Databricks workflow job
- an approved Databricks node type for the shared refresh cluster
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
  `CAN_USE` on the SQL warehouse; source data `USE CATALOG`, `USE SCHEMA`, `SELECT`; control plane `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY`
- app service principal, if you want full in-app job management:
  `CAN MANAGE` on the shared refresh workflow
- app service principal, if you want the app to accelerate onboarding with `Run now`:
  `CAN MANAGE RUN` on the shared refresh workflow is the minimum direct-trigger grant if `CAN MANAGE` is not available
- app service principal, if Setup should create missing objects:
  `CREATE TABLE` in the control-plane schema; `CREATE SCHEMA` if the schema is missing; `CREATE CATALOG` only if you plan to use the toggle
  If the control-plane schema and tables are already present, Setup now checks for them first and can succeed without those create grants.
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
cd <repo-root>
python3 -m pytest
python3 scripts/model_lens_setup.py --help
python3 scripts/model_lens_refresh.py --help
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
databricks bundle validate \
  -t warehouse_only \
  --var "sql_warehouse_id=<sql-warehouse-id>" \
  --var "refresh_node_type_id=<spark-node-type-id>" \
  --var "control_plane_catalog=<control-plane-catalog>" \
  --var "control_plane_schema=<control-plane-schema>"
```

Expected result:

- tests pass
- both wrapper scripts parse
- wheel build succeeds
- bundle validation passes

Use the real workspace namespace every time. There is no repo-wide default control-plane catalog/schema in the bundle now. For example, a Hive Metastore test workspace often uses:

```bash
--var "control_plane_catalog=hive_metastore" \
--var "control_plane_schema=model_lens_control_plane"
```

Note: the local test suite now includes Spark-refresh regressions. Run it from an environment with the repo dev dependencies installed so `pyspark` is available; the Spark-specific tests still skip automatically when no local Java runtime is present.
For large-tenant rollout, treat those local skipped tests as insufficient proof by themselves; the real release gate is successful Databricks Spark execution for bootstrap and incremental runs.

## Step 2: Create Scratch Data

Open a Databricks SQL editor and run [`examples/scratch_dataset.sql`](../examples/scratch_dataset.sql).

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
  --var "refresh_node_type_id=<spark-node-type-id>" \
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
- the workflow `<app-name>-refresh` exists on Spark job compute, where `<app-name>` is `model-lens` unless you overrode `app_name`
- the app service principal shown in `databricks apps get model-lens -o json` still has `CAN_USE` on the SQL warehouse
- the same app identity has `CAN MANAGE RUN` on the refresh workflow

If you are reusing an existing app and cannot manage its SQL warehouse app resource, do not use this smoke-test deploy path. Use the generated manual existing-app path in [Manual Setup With An Existing Databricks App](./MANUAL_EXISTING_APP_SETUP.md) instead.
If the customer workspace must also reuse an existing shared refresh job and an existing approved SQL warehouse, use [Constrained Workspace Runbook](./CONSTRAINED_WORKSPACE_RUNBOOK.md) as the primary test path instead of this bundle-created-app path.

## Step 4: Open The App

Open the `model-lens` Databricks App.

Check immediately:

- the app title reads `Model Lens`
- if no monitors exist yet, `Overview` renders a clean empty state that says `No monitors onboarded yet. Go to Onboarding to add your first model.` instead of returning a 500
- `SQL_WAREHOUSE_ID` is populated
- `REFRESH_JOB_ID` is populated or `REFRESH_JOB_NAME` matches the deployed workflow name
- if `REFRESH_JOB_ID` is blank, confirm you intentionally rely on name lookup; `REFRESH_JOB_ID` is safer for repeated customer deployments
- if `REFRESH_JOB_ID` is populated and the setup card shows `Grant CAN_MANAGE_RUN on job <id>`, grant the app service principal `CAN_MANAGE_RUN` on that job before expecting immediate bootstrap from the UI
- if the setup card shows `Immediate Bootstrap: Verification unavailable`, the app could not inspect workflow ACLs; direct trigger may still work, so test `Run First Refresh` before assuming the grant is missing
- if you intentionally enabled the optional bootstrap lane, verify `BOOTSTRAP_REFRESH_JOB_ID` or `BOOTSTRAP_REFRESH_JOB_NAME` is populated too; otherwise the app should show that bootstrap uses the shared refresh workflow by default
- `USE_LAKEBASE_READ_MODEL` is `false`
- `Control Plane Catalog` and `Control Plane Schema` show the namespace you want to use
- if Lakebase exists in the workspace and is visible to the app identity, an informational banner recommends Lakebase
- optional `Lakebase Instance Name` / `Lakebase Database Name` inputs are visible for session-level acceleration
- there is no pre-rename product naming anywhere

## Step 5: Initialize The Control Plane

In the app `Setup` step, click `Setup Control Plane`.

The wizard should keep `Continue to Discover` disabled until setup succeeds for the current control-plane namespace. If setup fails, fix the underlying issue and click `Setup Control Plane` again to retry. After redeploying a newer Model Lens build into an existing workspace, rerun `Setup Control Plane` once so additive schema migrations are applied before smoke testing. That rerun should now be safe on already-migrated workspaces because the setup path checks existing columns/tables before issuing additive `ALTER TABLE` migrations. It also backfills legacy completed `refresh_runs` rows so monitors bootstrapped before published generations existed stop showing `Computing/Pending` forever. For labeled monitors that already had performance history before `daily_label_metrics` existed, setup should also queue a one-time `performance_repair` so the next scheduled repair run backfills the missing daily labeled facts. If the table is still empty, subsequent `performance_repair` runs should ignore the unchanged-watermark optimization until those daily labeled facts exist.

Then click `Validate Workspace Wiring`.

Expected readiness modes:

- `fully_ready`: the app can trigger bootstrap immediately
- `scheduler_only`: the app can onboard monitors and rely on the scheduled shared workflow

Optional extension:

- if a separate bootstrap workflow is configured, the readiness card should show it as an optional bootstrap lane
- if no separate bootstrap workflow is configured, the readiness card should explicitly say bootstrap uses the shared refresh workflow by default
- a missing or ungranted optional bootstrap lane should warn, not block onboarding, because scheduled pickup still belongs to the main shared workflow

Expected blocked state:

- `not_ready`: the card should show the exact missing workflow, warehouse, or permission-adjacent wiring issue and `Continue to Discover` should stay disabled

Recommended:

- pre-create the target namespace outside the app
- leave `Create catalog if missing` off unless you are testing with an admin identity
- keep the app namespace fields aligned with the bundle vars so in-app refreshes and workflow refreshes hit the same control plane
- if you want to test fast app reads, enter `Lakebase Instance Name` and `Lakebase Database Name` before clicking setup

Expected result:

- success banner
- control-plane tables are created under your chosen `<control-plane-catalog>.<control-plane-schema>`
- `Workspace Readiness` resolves to either `fully_ready` or `scheduler_only`
- if you intentionally test a missing or wrong workflow config, the readiness card should stay `not ready` and block onboarding

Verify in SQL:

```sql
SHOW TABLES IN <control-plane-catalog>.<control-plane-schema>;
```

Expected tables:

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
   Shared-table example:
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
   Table-scoped example:
   - if your inference table already represents exactly one model and has no real model-id column, leave `Model ID Column` and `Monitored Model ID Value` blank on purpose
   - discovery should keep that draft as a normal table-scoped monitor instead of downgrading it just because `model_id` is absent
   If your real tables join on the same shared key name in both tables, `Entity ID Column` can stay blank and Model Lens should still validate the join against that shared column.
   Hyphenated feature names such as `us-central1` should also save successfully without manual renaming.
   If your real inference table already includes true labels, leave the labels-table inputs blank and confirm `Label Column In Source` instead.
7. In the same `Advanced` section, confirm feature selection:
   - `amount`
   - `velocity_7d`
   - `device_score`
   For wider real-world tables, expect all numeric features to remain selected by default unless you intentionally narrow them.
8. Optional slice check after the happy path:
   - confirm `region`
   - confirm `merchant_segment` if you added it to the dataset
9. Continue to the `Activate` step and confirm the cadence section:
   - `Drift / Quality Refresh`: `Every 6 Hours`
   - `Performance Refresh`: `Daily (7-Day Repair)` when labels are enabled
   - `Enable scheduled refreshes for this monitor`: on
   - `Tracked Performance Metrics`: `F1 Score`, `Precision`, `Recall`
   - `Default Performance Metric`: `F1 Score`
10. Save the monitor:

- click `Save Monitor`

Expected result:

- success banner
- the monitor is marked `pending bootstrap` in `monitor_runtime_state`
- if the app has `CAN MANAGE RUN`, the success banner says the monitor was saved and the shared refresh job was triggered asynchronously
- if the app cannot resolve the workflow or lacks `Run now` permission, the monitor is still saved; automatic pickup only happens if the shared hourly workflow already exists and the app is wired to it through `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`
- if the monitor remains `pending bootstrap`, the `Monitor Settings` page shows `Run First Refresh` for that selected monitor
- if a scheduled or direct bootstrap run still logs `refresh-control-plane complete: scope=bootstrap mode=auto models=0`, the same job log now includes `refresh-control-plane selection:` lines for every active monitor considered, including reasons such as `bootstrap_already_completed`, `bootstrap_run_already_running`, `recent_failure_backoff`, or `max_bootstraps_per_run_reached`
- if the shared refresh cluster cannot read Unity Catalog tables, verify the job cluster access mode is UC-capable (`USER_ISOLATION` by default in the bundle/manual payload, or `SINGLE_USER` if your workspace policy requires it)
- the monitor appears on the overview page
- the new `Incidents` page loads without errors, even before any incidents exist
- the `Monitor Settings` page shows the saved cadence, runtime state, and recent refresh-run history for the selected monitor
- `Monitor Settings` is now split into `Contract`, `Settings`, and `Admin` tabs; contract/summary data should be in `Contract`, cadence/diagnostics in `Settings`, and lifecycle/runtime wiring in `Admin`
- the `Refresh Diagnostics` section in `Monitor Settings` classifies recent runs as `Source Scan Bound`, `Daily Profiles Bound`, `Derivation Bound`, `Persistence Bound`, or `Mixed` once enough successful timed runs exist
- the `Refresh Diagnostics` recommendations match the recorded timings rather than a generic fixed banner
- the `Monitor Settings` page also shows recent incident lifecycle rows for that monitor when drift/performance incidents have been opened, escalated, or recovered
- the `Monitor Settings` page also shows an `Active` / `Archived` / `All` filter plus lifecycle controls; by default it follows the sidebar-selected monitor, old success banners clear when you switch monitors or revisit the page, active monitors show `Archive Monitor`, archived monitors show `Restore Monitor`, and `Delete Monitor And History` remains available in both states
- once incidents exist, the `Incidents` page shows cross-monitor open incidents and recent lifecycle rows, and its monitor/severity/status/metric filters all work without reloading the app
- after the workflow finishes, Drift and Performance should already show historical windows rather than a single snapshot
- after the workflow finishes, Data Quality should show window-history charts instead of only the latest summary row
- Drift should expose `Start Date`, `End Date`, `Class Basis`, and `Class Value` controls; for binary classification monitors, selecting a class filter after at least one new refresh should change the heatmap/top-feature set without rescanning the raw source table
- changing Drift date/class/granularity/top-N controls alone should not query immediately; the expensive Drift refresh should happen only after you click `Apply Drift Filters`
- Data Quality should expose the same date-range and binary-class controls; after at least one new refresh, class-filtered quality history should render instead of silently falling back to the unfiltered monitor-wide history
- changing Data Quality date/class/threshold controls alone should not query immediately; the expensive quality refresh should happen only after you click `Apply Quality Filters`
- for binary classification monitors, Data Quality class filters should either render the requested slice or show `No rows matched the selected class filter in this date range.`; they should not fall back to a generic “wait for refresh” message when source-derived data is available
- if you apply a class filter before the workspace has run a refresh on the new build, the page should now distinguish between:
  - `No rows matched the selected class filter in this date range`
  - `Filtered ... history is not available for the full range yet. Select a date range or refresh ...`
  - the older unavailable-after-refresh message when class-aware daily facts are genuinely missing
- Drift chart titles should include the active granularity and any active filters, threshold guides should stay hidden until you enable `Show Threshold Guides`, and very small drift values should switch to scientific notation instead of collapsing into unreadable `0.0000` labels
- `Drift Analysis` should now expose an inline `Thresholds` accordion for `PSI`, `Jensen-Shannon`, `KL Divergence`, and `Null Rate (%)`; saving there should persist the same per-monitor overrides used by Overview/Data Quality, and `Reset to Defaults` should clear them
- `Monitor Settings -> Settings` should now expose per-monitor threshold overrides for `PSI`, `Jensen-Shannon Divergence`, `KL Divergence`, and `Null Rate (%)`; saving those values should change Overview/Drift/Data Quality severity semantics for future refreshes without rewriting historical incident history
- Performance should prefer raw daily metrics from persisted `daily_label_metrics`; if a day has undefined `precision`, `recall`, or `f1`, the timeline should show a gap rather than a forced zero. If the workspace has not backfilled `daily_label_metrics` yet, the page should fall back to persisted weighted daily/window performance history with an explanatory warning instead of showing an empty chart
- the Performance page should now expose a `Tracked Drift Features` multi-select and a drift-threshold toggle; monitors with a small tracked-feature set should default to showing all of them on the lower drift chart, and that lower drift chart should respect the selected metric's threshold guides
- the Performance feature-impact chart and the latest-bin table should now use `Weighted Contribution` and `Current Window Share (%)` wording, with help popovers explaining that long red bars hurt the selected metric most and long green bars help it most
- the old lower Performance bin-detail plot should be gone; instead, the selected feature should offer a button that opens Feature Deep Dive with that feature prefilled
- Feature Deep Dive should now show `Binning Mode`, optional custom edges, both `Percentile Clip` / `IQR Fence` outlier controls, and an `Apply Distribution Controls` button; feature/dimension changes rerender immediately, but heavier distribution-control changes should only apply after clicking that button
- the Feature Deep Dive dimension breakdown should explicitly describe Average vs Median (P50) and show P25 / P50 / P75 / Rows in hover text
- when those exact-sample modes require raw values, the page should either use a bounded raw-window read or explain that exact-sample detail is unavailable
- Data Quality's lower-right chart should now be `Latest Window Performance Snapshot` for binary classification monitors instead of the old prediction-distribution chart
- `Monitor Settings -> Admin -> Shared Workflow Schedule` should show the detected shared-job wake interval; if the app service principal has `CAN_MANAGE` on the shared job, the interval should be editable in-app, otherwise the card should stay read-only with explicit guidance
- `Monitor Settings -> Settings -> Compute Guidance` should now explain the current shared wake interval, per-monitor cadence, and recent compute-footprint tier (`Low`, `Elevated`, `High`, or `No compute footprint data yet`)
- with the scratch dataset and `Baseline Days = 7`, you should have 8 daily comparison windows immediately
- for very large real-world tables, the first run should use one exact bounded Spark source-range load per monitor scope, persist daily facts and affected derived windows through Spark/Delta writes, and reserve the configured sampling caps for UI/detail fallbacks rather than core refresh correctness
- for multi-monitor tenants, the shared job should still avoid two scopes for the same model in one scheduler pass; the Spark refresh repository now defaults to serial monitor execution inside the driver unless you deliberately override the worker cap for that workspace
- for very wide monitors, watch runtime and cluster pressure carefully because the remaining main scale cost is per-feature Spark work, not pandas raw-frame loading
- `quality_metrics` should remain model-wide because the workflow rebuilds it from all persisted `daily_quality_profiles`, not only from the bounded refresh slice
- `performance_bin_specs` should be created for numeric performance features so later repair runs reuse the same bucket edges
- if labels come from the inference table itself, performance repair should track an opaque label-freshness signature instead of only the max event timestamp
- if the repository falls back to an unbounded raw current-window load, rows later in the `window_end` day should still appear in prediction and dimension detail views
- the remaining historical hardening work is tracked in [Historical Backfill Plan](./HISTORICAL_BACKFILL_PLAN.md)

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
SELECT model_key, COUNT(*) AS daily_quality_rows
FROM <control-plane-catalog>.<control-plane-schema>.daily_quality_profiles
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- one row
- at least one persisted daily-quality profile row

```sql
SELECT model_key, COUNT(*) AS daily_feature_rows
FROM <control-plane-catalog>.<control-plane-schema>.daily_feature_profiles
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- one row
- at least one persisted daily-feature profile row

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
SELECT DISTINCT metric_name
FROM <control-plane-catalog>.<control-plane-schema>.performance_metrics
WHERE model_key = 'fraud_model_demo'
ORDER BY metric_name;
```

Expected for the default classification path:

- `f1`
- `precision`
- `recall`

```sql
SELECT model_key, COUNT(*) AS daily_perf_rows
FROM <control-plane-catalog>.<control-plane-schema>.daily_performance_profiles
WHERE model_key = 'fraud_model_demo'
GROUP BY 1;
```

Expected:

- at least one row when labels are available

```sql
SELECT model_key, feature_name, edges_json
FROM <control-plane-catalog>.<control-plane-schema>.performance_bin_specs
WHERE model_key = 'fraud_model_demo'
ORDER BY feature_name;
```

Expected:

- at least one row when labels and numeric performance features are available
- `edges_json` stays stable across later incremental repair runs for the same monitor

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
SELECT model_key, requested_mode, run_kind, status, window_count, data_min_date, data_max_date, range_start, range_end, rows_scanned
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
- `range_start` / `range_end` are populated for scheduled shared-job runs
- `rows_scanned` reflects the SQL-side bounded refresh range, not an unbounded app-side full-frame read
- the row exists even for early skipped or failed monitor attempts, because the workflow now creates it before source-range discovery
- if a later bootstrap/backfill run fails before completion, the previously published monitor generation should still remain visible in the app instead of disappearing mid-run
- the resulting `comparison_windows`, `drift_metrics`, `quality_history`, and `performance_metrics` rows were derived from the daily profile layer built for that bounded range inside the same refresh run
- on later incremental runs, those derived rows can also reuse already-persisted daily profile facts for the affected span instead of depending only on the current run’s bounded load

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
- the quality page shows `Rows Per Comparison Window`, `Null Rate Trends`, and `Prediction Average Over Time`
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

## Step 11: Focused Rollout Gates

Run these three checks before broader customer rollout:

### Overview with multiple active monitors

- onboard or restore a second monitor
- return to Overview

Expected:

- both monitor cards render
- the page loads successfully using the bulk latest-quality plus historical-drift-summary read path in a real Databricks workspace
- if one monitor had severe drift earlier but the newest window recovered, Overview still ranks that monitor by the historical max PSI instead of showing it as fully healthy
- a newly created or not-yet-derived monitor appears as `Computing/Pending`, not as a healthy zero-PSI card
- the page shows a loading spinner instead of blank content while the warehouse read is running

### Severe numeric drift

- use a scratch variant or real table whose latest current-window values sit fully outside the baseline range
- run the shared refresh workflow
- open Drift

Expected:

- PSI / JS / KL stay finite
- the page shows a real severe-drift signal instead of blank output or warning-driven gaps
- the Top N feature ranking reflects the highest historical drift across the stored windows, not only the newest window
- the heatmap title reflects the active granularity and only includes the ranked Top N features instead of every feature in the monitor

### Same-day current-window detail

- use a source table whose newest rows land later in the day on `window_end`
- open Prediction Distribution and one Dimension Breakdown view for that monitor

Expected:

- those same-day rows are included when the repository supports bounded current-window reads
- if the repository cannot serve a bounded raw read, the deep-dive page shows an explicit unavailable state rather than silently scanning the full source table
- the feature context card shows the active baseline/current dates plus whether the distribution came from persisted profile data or a bounded source-window read

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
- Overview renders with 2+ active monitors in a real workspace
- severe out-of-range drift still produces finite PSI / JS / KL values
- same-day current-window detail views include rows later on `window_end`
- there are no old names left in the product surface
