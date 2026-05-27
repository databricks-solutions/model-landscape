# Changelog

All notable changes to **mlflow-lens** are tracked here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Plotly-native panel quick functions** modeled on Yellowbrick.
  - Classifier: `roc_auc`, `confusion_matrix`, `precision_recall`,
    `classification_report`, `class_prediction_error`,
    `discrimination_threshold`.
  - Regressor: `prediction_error`, `residuals`, `alpha_selection`.
  - Model selection: `learning_curve`, `validation_curve`,
    `feature_importances`, `cv_scores`.
- Each quick function:
  - returns a `plotly.graph_objects.Figure`,
  - exposes a pre-computed entry point (`.from_scores`,
    `.from_predictions`, or `.from_values`),
  - logs both `lens/panels/{type}.json` and `lens/panels/{type}.html`
    to the active MLflow run when `log=True`.
- Shared Plotly theme in `mlflow_lens._plotly_theme` (`lens_layout`).
- `mlflow-lens` deps now include `plotly>=5.24` and `scikit-learn>=1.5`.
- MkDocs Material documentation site with an interactive Plotly gallery.

### Changed

- `mlflow_lens.panels.PANEL_TYPES` extended from 4 to 13 types.
- `mlflow_lens._artifacts.log_json_artifact` now serializes numpy scalars
  and arrays as native Python types.

## [0.1.0] — 2026-05-27

### Added

- Initial public SDK release.
- Structured run summaries (`summary.log`, `summary.build`, `summary.cache`).
- Training-time drift detection (`drift.log_drift`).
- Cost-attribution tags (`cost.log_cost_context`).
- Databricks workspace context (`experiment.auto_log_context`).
- Low-level panel artifact API (`panels.log_panel`).
- 4 original panel types: `confusion_matrix`, `roc_curve`,
  `feature_importance`, `learning_curve`.
- Delta-table export (`export.to_delta`).
