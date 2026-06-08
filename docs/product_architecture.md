# Product architecture and package boundaries

Model Landscape is the product. It has two packages with separate jobs:

| Package | Role | Runs where |
| --- | --- | --- |
| `mlflow-lens` | Training-time SDK for MLflow run context, summaries, SHAP/feature-importance panels, confusion/ROC/PR panels, and training-time drift artifacts. | Notebooks, training jobs, and MLflow experiments. |
| `model_landscape` | Databricks App and control plane for production monitoring, refresh workflows, incidents, drift, performance, data quality, and Lakebase/warehouse reads. | Databricks Apps, Databricks Workflows, and SQL warehouses. |

The packages are intentionally not aliases of each other. There is no
`model_lens` compatibility import and no legacy `model-lens` app entrypoint.
Old Model Lens development continues here as Model Landscape. Model Lens is
historical provenance, not a parallel maintained product.

## Shared concepts

- **Drift math:** both packages use the same concepts: PSI, Jensen-Shannon
  divergence, KL divergence, and feature-level distribution comparisons.
- **MLflow lineage:** the app stores linked experiment/run/model-version
  lineage in monitor configs so production incidents can be interpreted next
  to training context.
- **Tutorial flow:** `mlflow-lens` enriches training runs; Model Landscape
  monitors the inference table produced by those models.
- **Production monitoring contract:** the app still requires an inference
  table with timestamps, predictions, feature columns, and optional labels.
  SDK artifacts add context but are never required for production monitoring.

## SDK artifact contract consumed by the app

The app discovers SDK artifacts through MLflow run tags. It does not need to
list the artifact tree during onboarding.

| Contract item | Path or tag | Meaning |
| --- | --- | --- |
| SDK version tag | `lens.version` | Version that wrote the SDK artifacts. |
| Summary tag | `lens.has_summary=true` | Run contains `lens/summary.json`. |
| Drift tag | `lens.has_drift=true` | Run contains `lens/drift.json`. |
| Panel tag | `lens.panel.<panel_type>=true` | Run contains `lens/panels/<panel_type>.json`. |
| Summary artifact | `lens/summary.json` | Run task, primary metric, score, dataset, notes, and extra fields. |
| Drift artifact | `lens/drift.json` | Reference run, feature drift rows, and prediction-shift payload. |
| Drift snapshot artifact | `lens/drift_snapshot.json` | Lightweight drift snapshot used by SDK workflows. |
| Panel artifact | `lens/panels/<panel_type>.json` | Raw panel payload for app rendering or comparison. |

Supported panel types currently mirror `mlflow_lens.panels.PANEL_TYPES`:

```text
confusion_matrix, roc_curve, feature_importance, learning_curve,
classification_report, precision_recall_curve, class_prediction_error,
discrimination_threshold, prediction_error, residuals, alpha_selection,
validation_curve, cv_scores
```

The parser helpers in `model_landscape.services.mlflow_artifacts` are
permissive by design: unknown fields are preserved, unknown panel types are
accepted, and missing optional fields do not block monitor discovery.

## UX direction

Training context should be shown where it explains production behavior:

- onboarding should tell users when linked MLflow runs have SDK artifacts;
- Monitor Settings can show the linked lineage and artifact availability;
- future app sections can render feature importance, confusion matrix, ROC/PR,
  run summary, and training drift from those artifacts;
- production pages remain unchanged: Overview, Drift, Performance, Data
  Quality, Incidents, Feature Deep Dive, and Monitor Settings.
