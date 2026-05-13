# mlflow-lens SDK Design

## Overview

mlflow-lens is a thin, opinionated Python SDK that enriches MLflow experiment tracking with structured summaries, training-run drift detection, visualization panels, cost attribution, and Delta export. It composes with `mlflow.start_run()` -- never replaces it.

> MLflow logs everything. mlflow-lens explains it.

mlflow-lens turns:
- **metrics -> meaning**
- **runs -> stories**
- **changes -> explanations**
- **costs -> attribution**

## Philosophy

1. **MLflow is the source of truth** -- no duplication of experiment storage, no competing abstractions. Everything derives from MLflow runs, LoggedModels, artifacts, and Databricks system tables.

2. **Thin overlay, not a platform** -- no orchestration, no pipeline system. The SDK composes with `mlflow.start_run()`, it does not wrap or replace it.

3. **Zero-friction adoption** -- the Model Landscape app works on raw MLflow experiments immediately (no SDK required). SDK enrichment is additive with graceful degradation when custom artifacts are absent.

4. **Structured artifacts over HTML** -- store data and schema as JSON artifacts, render in the app UI. Why: composability, queryability, cross-run consistency.

5. **Cross-run understanding > single-run inspection** -- primary value is comparison, timelines, drift across training runs. Not isolated single-run debugging (MLflow native UI handles this).

6. **Narrative-first UX** -- every experiment view answers: What happened? What changed? Should I trust this? Has the data shifted? What did it cost?

7. **Version everything** -- all Lens artifacts include `lens_version` and `schema_version`. All tags use a `lens.` prefix. Forward compatibility: unknown fields are preserved, not dropped.

8. **MLflow 3 native** -- support LoggedModel as a first-class entity. Require `mlflow>=2.10` for broad compatibility, optimize for MLflow 3.x. Never depend on removed APIs.

## Modules

### `experiment`

Context enrichment and structured logging on active MLflow runs.

- **`auto_log_context()`** -- logs workspace URL, cluster ID, runtime version, notebook path, git SHA as `lens.*` tags
- **`log(data: dict)`** -- flattens nested dicts to dot-notation MLflow params

```python
with mlflow.start_run():
    experiment.auto_log_context()
    experiment.log({
        "data": {"source": "credit_v3", "rows": 50000},
        "preprocessing": {"scaler": "standard"}
    })
```

### `summary`

Lightweight structured run representations for fast ranking and comparison.

- **`log(task, primary_metric, score, ...)`** -- writes `lens/summary.json` artifact
- **`build(experiment_ids, tags)`** -- paginates all runs (defeats the 1000-run cap), builds append-only index
- **`cache(index, path)`** / **`load(path)`** -- parquet persistence with JSON column serialization

### `drift`

Training-run drift detection using the shared analytics engine.

- **`log_drift(reference_run_id, features, predictions)`** -- computes PSI + KL + JS divergence per feature, saves `lens/drift.json` and `lens/drift_snapshot.json` for chained comparisons
- **`compute_psi(reference, current, n_bins)`** -- delegates to `mlflow_lens.analytics.drift` (stable histogram edges, 20 bins)
- **`classify_drift(psi)`** -- categorizes: low (< 0.1), moderate (0.1--0.25), high (>= 0.25)

### `panels`

Declarative visualization artifacts rendered by the app.

- **`log_panel(panel_type, data, ...)`** -- writes structured JSON to `lens/panels/`
- Supported types: `confusion_matrix`, `roc_curve`, `feature_importance`, `learning_curve`

### `cost`

Tags runs with cluster metadata for system table cost attribution.

- **`log_cost_context()`** -- captures `lens.cluster_id` and `lens.cost_eligible` on Databricks. No-op locally.

### `export`

Bridges enriched indexes to Delta tables for BI integration.

- **`to_delta(experiment_ids, table_name)`** -- writes index to Unity Catalog via Spark. Requires Databricks runtime.

### `analytics.drift`

Canonical drift math shared by both mlflow-lens (training) and Model Landscape (production).

- `compute_psi`, `compute_kl`, `compute_js` -- numeric feature drift with stable histogram edges (20 bins, epsilon smoothing)
- `compute_feature_drift` -- batch computation across all features
- `compute_categorical_distribution_metrics` -- PSI/KL/JS for categorical features

## Artifact Schema

All artifacts stored under `lens/` in MLflow runs:

```
artifacts/
├── lens/
│   ├── summary.json              # run summary for ranking
│   ├── drift.json                # feature PSI/KL/JS + prediction shift
│   ├── drift_snapshot.json       # feature/prediction stats for chaining
│   └── panels/
│       ├── confusion_matrix.json
│       ├── roc_curve.json
│       ├── feature_importance.json
│       └── learning_curve.json
```

### Run Envelope (Internal Schema)

```json
{
    "lens_version": "0.1.0",
    "schema_version": "2",
    "run_id": "abc123",
    "experiment_id": "1",
    "run_name": "regularized-xgb",
    "status": "FINISHED",
    "start_time": 1711540800000,
    "end_time": 1711541142000,
    "duration_s": 342,
    "indexed_at": "2026-03-27T12:00:00Z",
    "has_lens_artifacts": true,
    "metrics": {"auc": 0.91, "f1": 0.87},
    "params": {"max_depth": 6, "learning_rate": 0.1},
    "tags": {"lens.cluster_id": "0312-..."},
    "summary": {"task": "classification", "primary_metric": "auc", "score": 0.91},
    "logged_model_id": "m-abc123",
    "compute": {
        "cluster_type": "i3.xlarge",
        "runtime": "16.2 ML",
        "dbu_cost": 4.82,
        "wall_clock_s": 342
    }
}
```

## Roadmap

### Implemented
- `experiment` module (context + metadata)
- `summary` module (log, build, cache, load with pagination)
- `drift` module (PSI + KL + JS with stable edges, snapshot chaining)
- `panels` module (4 panel types)
- `cost` module (Databricks tagging)
- `export` module (Delta export skeleton)
- `analytics.drift` (shared engine)

### Next
- App-side panel rendering with cross-run comparison
- Cost view: system table enrichment for per-run compute cost
- Timeline view: run sequence with performance trends
- LoggedModel integration for MLflow 3.x

### Future
- System table intelligence (cluster utilization, job lineage, audit)
- Unity Catalog lineage tracking
- Advanced panels (calibration, embeddings, correlations)
- Training-time alerts and thresholds

## Non-Goals

1. **Production inference monitoring** -- Model Landscape app handles this
2. **LLM tracing and evaluation** -- MLflow 3 handles this natively
3. **Pipeline orchestration** -- use Databricks Workflows
4. **Model serving or deployment** -- use Databricks Model Serving
5. **Data versioning** -- use Delta time travel
6. **Replacing MLflow's native UI** -- mlflow-lens complements it
