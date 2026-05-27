# mlflow-lens

**Plotly-native panels and run-summary helpers for MLflow.**

`mlflow-lens` is a small, focused SDK that lives next to your MLflow runs. It
gives you Yellowbrick-style diagnostic plots — confusion matrices, ROC curves,
precision-recall curves, residuals, learning curves, and more — rendered as
interactive Plotly figures and logged alongside the run as both raw data
(`lens/panels/*.json`) and self-contained HTML (`lens/panels/*.html`).

```python
import mlflow
from mlflow_lens.classifier import roc_auc

with mlflow.start_run():
    fig = roc_auc(model, X_test, y_test, log=True)
    fig.show()
```

## Why mlflow-lens?

- **Plotly-only.** Every figure is interactive in notebooks and in the MLflow
  artifact viewer. No matplotlib in the dep tree.
- **Yellowbrick-shaped API.** Quick functions you call like
  `roc_auc(model, X, y)` — no class hierarchy to learn.
- **JSON + HTML side-by-side.** The raw data lives in `lens/panels/*.json`
  so dashboards can render compactly; the interactive HTML is right there
  in the MLflow UI for one-click inspection.
- **Single source of truth in your run.** Tags like `lens.panel.roc_curve=true`
  and `lens.figure.roc_curve=true` make panels discoverable.

## What's in the box

<div class="grid cards" markdown>

- :material-chart-scatter-plot: __Classification__

    ---

    ROC/AUC, confusion matrix, precision-recall, classification report,
    class prediction error, discrimination threshold.

    [→ Classifier panels](api/classifier.md)

- :material-chart-line: __Regression__

    ---

    Prediction error, residuals, alpha selection.

    [→ Regressor panels](api/regressor.md)

- :material-tune-variant: __Model selection__

    ---

    Learning curve, validation curve, feature importances, CV scores.

    [→ Model selection panels](api/model_selection.md)

- :material-database-arrow-right: __Run helpers__

    ---

    Training-time drift, structured run summaries, cost tagging,
    Databricks workspace context.

    [→ Drift](api/drift.md) · [→ Summary](api/summary.md)

</div>

## Status

`mlflow-lens` is in **Alpha** (0.1.x). The panel JSON schemas are
versioned (`schema_version: "1"`) and considered stable; the Python API
may still evolve before 1.0.

See the [changelog](changelog.md) for what shipped in each release.
