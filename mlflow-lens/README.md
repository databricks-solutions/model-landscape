# mlflow-lens

MLflow enrichment SDK for training-time observability. Adds structured run
summaries, training-time drift detection, cost attribution, and standard
classification/regression panels (confusion matrix, ROC, feature importance)
to any MLflow tracking run.

## Designed for Serverless env v5 (ML)

`mlflow-lens` targets the **Databricks Serverless environment version 5 ML**
base environment (`workspace-base-environments/databricks_ml_v5`) and the
equivalent Databricks Runtime for ML. Its dependencies are pinned with
floor-only bounds against exactly what env v5 ships, so it installs without
downgrading anything the runtime already provides:

| Dependency      | Env v5 ships | mlflow-lens requires |
| --------------- | ------------ | -------------------- |
| Python          | 3.12.3       | `>=3.10`             |
| mlflow-skinny   | 3.8.1        | `>=2.20`             |
| numpy           | 2.1.3        | `>=1.26`             |
| pandas          | 2.2.3        | `>=2.2`              |
| plotly          | 5.24.1       | `>=5.24`             |
| scikit-learn    | 1.6.1        | `>=1.5`              |

Because every runtime dependency is already present in env v5, installing with
`--no-deps` (see below) is the recommended path there — pip then adds only
`mlflow-lens` itself and leaves the runtime's MLflow 3.x stack untouched. The
package also remains compatible with older environments (e.g. env v4 /
MLflow 2.x) down to the floors above.

## Install

### From source (works today)

Until a release wheel is published, install straight from this repo. The
package lives in the `mlflow-lens/` subdirectory:

```bash
pip install "git+https://github.com/databricks-solutions/model-landscape.git@dev#subdirectory=mlflow-lens"
```

On a Databricks runtime that already provides the dependencies (Serverless
env v5, DBR ML), add `--no-deps` so pip never touches the runtime's packages:

```bash
pip install --no-deps "git+https://github.com/databricks-solutions/model-landscape.git@dev#subdirectory=mlflow-lens"
```

On Serverless env v5, reference it from your job's environment spec — use a
base environment **or** `environment_version`, not both:

```yaml
environments:
  - environment_key: mlv5
    spec:
      base_environment: workspace-base-environments/databricks_ml_v5
      dependencies:
        - "git+https://github.com/databricks-solutions/model-landscape.git@dev#subdirectory=mlflow-lens"
```

### From a GitHub Release

Once a release is cut, `mlflow-lens` is distributed through GitHub Releases on
this repo (not PyPI). Pin to a specific version:

```bash
pip install "https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.2.0/mlflow_lens-0.2.0-py3-none-any.whl"
```

Find the latest version at
[/releases](https://github.com/databricks-solutions/model-landscape/releases?q=mlflow-lens)
and substitute the tag/version above.

In `pyproject.toml`:

```toml
dependencies = [
  "mlflow-lens @ https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.2.0/mlflow_lens-0.2.0-py3-none-any.whl",
]
```

For the Spark `export` module (`mlflow_lens.export.to_delta`), append the
`[spark]` extra:

```bash
pip install "https://.../mlflow_lens-0.2.0-py3-none-any.whl[spark]"
```

Each release attaches a `SHA256SUMS` file. To verify the wheel before installing:

```bash
curl -LO https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.2.0/SHA256SUMS
curl -LO https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.2.0/mlflow_lens-0.2.0-py3-none-any.whl
sha256sum -c SHA256SUMS
```

## Usage

```python
import mlflow
import mlflow_lens

with mlflow.start_run() as run:
    # ... train your model ...

    # Workspace + git context, dataset signatures, env capture
    mlflow_lens.experiment.log_context()

    # Structured summary artifact
    mlflow_lens.summary.log_summary(model=model, metrics={"f1": 0.92})

    # Training-time feature/prediction drift vs a reference run
    mlflow_lens.drift.log_drift(
        reference_run_id="abc123",
        features=X_test,
        predictions=y_pred,
    )

    # Confusion / ROC / feature-importance panels (see "Visualization panels")
    from mlflow_lens.classifier import confusion_matrix, roc_auc
    confusion_matrix(model, X_test, y_test, log=True)
    roc_auc(model, X_test, y_test, log=True)

    # Compute cost attribution
    mlflow_lens.cost.log_cost_context()
```

## Visualization panels

Panels live in three packages — `mlflow_lens.classifier`, `mlflow_lens.regressor`,
and `mlflow_lens.model_selection`. Each panel has **two ways to call it**:

* **Model-based** — pass a fitted model and data, e.g.
  `confusion_matrix(model, X, y, log=True)`. Uses scikit-learn conventions
  (`predict`, `predict_proba`/`decision_function`, `classes_`,
  `feature_importances_`/`coef_`).
* **Framework-agnostic** — a `.from_*` classmethod that takes plain arrays you
  computed yourself, e.g. `confusion_matrix.from_predictions(y_true, y_pred, log=True)`.
  Use this for any model that isn't a scikit-learn estimator.

```python
import torch
from mlflow_lens.classifier import roc_auc, confusion_matrix

# A PyTorch / native-booster model: compute predictions yourself, then pass arrays.
probs = torch.softmax(net(X_test), dim=1).detach().cpu().numpy()
preds = probs.argmax(axis=1)

roc_auc.from_scores(y_test, probs, log=True)
confusion_matrix.from_predictions(y_test, preds, log=True)
```

### Framework support

| Panel(s) | scikit-learn | XGBoost / LightGBM (sklearn API, e.g. `XGBClassifier`) | XGBoost / LightGBM native `Booster` | PyTorch |
| --- | --- | --- | --- | --- |
| `confusion_matrix`, `class_prediction_error`, `classification_report`, `prediction_error`, `residuals` | direct | direct | `.from_predictions` | `.from_predictions` |
| `roc_auc`, `precision_recall`, `discrimination_threshold` | direct | direct | `.from_scores` | `.from_scores` |
| `feature_importances` | direct | direct | `.from_values` | `.from_values` |
| `feature_importance.shap_importance` | direct (Tree/Kernel) | direct (Tree) | direct (Tree) | `.from_shap_values`² |
| `cv_scores` | direct | direct | `.from_scores` | `.from_scores` |
| `learning_curve`, `validation_curve`, `alpha_selection` | direct | direct | not supported¹ | not supported¹ |

*"direct" = pass the fitted model. Otherwise use the listed `.from_*` classmethod with pre-computed arrays.*

¹ These panels re-fit the estimator across folds / hyperparameters via
scikit-learn, so they require a scikit-learn-compatible estimator
(`get_params`/`set_params`/`fit`). The XGBoost and LightGBM **sklearn wrappers**
qualify; native `Booster` objects and PyTorch modules do not. Pre-compute
fold scores yourself and use `cv_scores.from_scores(...)` instead.

² `shap_importance` auto-uses `TreeExplainer` for tree models (including native
boosters) and `KernelExplainer` for any model exposing `predict`/`predict_proba`.
For PyTorch, compute SHAP values with `shap.DeepExplainer`/`GradientExplainer`
and pass them to `shap_importance.from_shap_values(...)`.

> **Native boosters:** `Booster.predict` returns probabilities, not class
> labels, so use `.from_predictions` / `.from_scores` rather than the
> model-based entry point even though the booster has a `.predict` method.

### SHAP feature importance

`mlflow_lens.feature_importance.shap_importance` computes global importance as
the mean absolute SHAP value per feature and emits the same
`feature_importance` panel type as `model_selection.feature_importances`. The
explainer is auto-selected: `shap.TreeExplainer` for tree models (scikit-learn
ensembles, XGBoost, LightGBM, CatBoost), `shap.KernelExplainer` for everything
else.

Requires the `[shap]` extra (`shap` is already present in the Serverless env v5
ML base environment, so `--no-deps` installs need nothing extra there):

```python
from mlflow_lens.feature_importance import shap_importance

# Auto-detects TreeExplainer; X may be a DataFrame (names inferred) or array.
shap_importance(model, X_test, top_n=20, log=True)

# KernelExplainer is bounded for cost: max_samples rows explained,
# background_samples for the background, nsamples coalitions per row ("trials").
shap_importance(model, X_test, max_samples=100, background_samples=100, nsamples="auto")

# Pre-computed SHAP values (e.g. shap.DeepExplainer / GradientExplainer for PyTorch):
shap_importance.from_shap_values(feature_names, shap_values, top_n=20, log=True)
```

## Modules

| Module | Purpose |
|---|---|
| `mlflow_lens.experiment` | Workspace, git, env, and dataset context logging |
| `mlflow_lens.summary` | Structured run summaries as JSON artifacts |
| `mlflow_lens.drift` | PSI / KL / JS drift between training runs |
| `mlflow_lens.classifier` | Confusion matrix, ROC, PR, classification report, threshold panels |
| `mlflow_lens.regressor` | Prediction-error, residuals, alpha-selection panels |
| `mlflow_lens.model_selection` | Learning/validation curves, feature importances, CV scores |
| `mlflow_lens.feature_importance` | SHAP feature importance (auto Tree/Kernel explainer; requires `[shap]`) |
| `mlflow_lens.panels` | Low-level `log_panel(panel_type, data, ...)` JSON writer |
| `mlflow_lens.cost` | Compute cost attribution per run |
| `mlflow_lens.export` | Export run summaries to Delta (requires `[spark]` extra) |

## License

Licensed under the Databricks License. See [LICENSE.md](LICENSE.md).

## Source

This SDK is developed in
[databricks-solutions/model-landscape](https://github.com/databricks-solutions/model-landscape)
alongside the Model Landscape Databricks App that consumes the same drift
and metric definitions. See [RELEASING.md](RELEASING.md) for the release
process.
