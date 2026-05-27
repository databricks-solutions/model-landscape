# Concepts

## Two layers: data + figure

Every panel in mlflow-lens is logged as **two artifacts** in the same run:

| Artifact | Path | Consumer |
| --- | --- | --- |
| Raw data (JSON) | `lens/panels/{type}.json` | Lens warehouse app, comparison dashboards, programmatic post-processing |
| Interactive figure (HTML) | `lens/panels/{type}.html` | MLflow UI artifact viewer, one-click inspection in a browser |

The data side is the long-lived contract; the HTML is a convenience for
human eyeballs.

## Quick functions vs. low-level `log_panel`

There are two ways to log a panel:

### Quick function (recommended)

```python
from mlflow_lens.classifier import roc_auc

fig = roc_auc(model, X, y, log=True)
```

The quick function does three things:

1. Computes the panel's data (e.g. per-class FPR/TPR/AUC).
2. Builds a Plotly figure with the shared `lens_layout` theme.
3. If `log=True`, writes both the JSON data via
   :func:`mlflow_lens.panels.log_panel` and the HTML figure via
   `mlflow.log_figure`.

It returns the Plotly `Figure` so you can `fig.show()` it in a notebook.

### Low-level `log_panel`

```python
from mlflow_lens.panels import log_panel

log_panel("roc_curve", {"classes": [...], "n_classes": 2})
```

Use this when:

- You already have panel data computed by some other tool and just want
  to attach it to a run.
- You're writing a custom panel renderer and want the artifact pipeline
  without the figure.

## Panel types

The panel-type registry lives at :data:`mlflow_lens.panels.PANEL_TYPES`.
The full list:

| Category | Type | Quick function |
| --- | --- | --- |
| Classification | `confusion_matrix` | `classifier.confusion_matrix` |
| Classification | `roc_curve` | `classifier.roc_auc` |
| Classification | `precision_recall_curve` | `classifier.precision_recall` |
| Classification | `classification_report` | `classifier.classification_report` |
| Classification | `class_prediction_error` | `classifier.class_prediction_error` |
| Classification | `discrimination_threshold` | `classifier.discrimination_threshold` |
| Regression | `prediction_error` | `regressor.prediction_error` |
| Regression | `residuals` | `regressor.residuals` |
| Regression | `alpha_selection` | `regressor.alpha_selection` |
| Model selection | `learning_curve` | `model_selection.learning_curve` |
| Model selection | `validation_curve` | `model_selection.validation_curve` |
| Model selection | `feature_importance` | `model_selection.feature_importances` |
| Model selection | `cv_scores` | `model_selection.cv_scores` |

## Panel JSON shape

Every payload has this top-level envelope:

```json
{
  "lens_version": "0.1.0",
  "schema_version": "1",
  "type": "<panel_type>",
  "data": { ... or [...] }
}
```

Plus optional `top_n`, `labels`, and arbitrary keys merged from `extra`.

The shape of `data` is type-specific:

- **`roc_curve`** — `{ "classes": [{label, fpr, tpr, auc}, ...], "n_classes": int, "macro_auc": float }`
- **`confusion_matrix`** — `{ "matrix": [[int, ...], ...], "normalize": null | "true" | "pred" | "all" }`
- **`feature_importance`** — `[ { "feature": str, "importance": float }, ... ]` (sorted desc)
- **`residuals`** — `{ "test": [{y_pred, residual}, ...], "train": [...], "r2": float }`

…and so on. The mkdocstrings-generated [API pages](api/index.md) document
each panel's exact return data.

## Tags

When `log=True`, mlflow-lens sets a few tags on the run so panels are
discoverable without listing the full artifact tree:

```
lens.version             # SDK version (e.g. "0.1.0")
lens.panel.<type>        # "true" if JSON panel artifact exists
lens.figure.<type>       # "true" if HTML figure artifact exists
```

Filter runs by tag in the MLflow UI or programmatically via
`mlflow.search_runs(filter_string="tags.\"lens.figure.roc_curve\" = 'true'")`.
