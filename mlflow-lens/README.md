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
pip install "https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.1.0/mlflow_lens-0.1.0-py3-none-any.whl"
```

Find the latest version at
[/releases](https://github.com/databricks-solutions/model-landscape/releases?q=mlflow-lens)
and substitute the tag/version above.

In `pyproject.toml`:

```toml
dependencies = [
  "mlflow-lens @ https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.1.0/mlflow_lens-0.1.0-py3-none-any.whl",
]
```

For the Spark `export` module (`mlflow_lens.export.to_delta`), append the
`[spark]` extra:

```bash
pip install "https://.../mlflow_lens-0.1.0-py3-none-any.whl[spark]"
```

Each release attaches a `SHA256SUMS` file. To verify the wheel before installing:

```bash
curl -LO https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.1.0/SHA256SUMS
curl -LO https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.1.0/mlflow_lens-0.1.0-py3-none-any.whl
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

    # Confusion / ROC / feature importance
    mlflow_lens.panels.log_panel("confusion_matrix", y_true=y_test, y_pred=y_pred)

    # Compute cost attribution
    mlflow_lens.cost.log_cost_context()
```

## Modules

| Module | Purpose |
|---|---|
| `mlflow_lens.experiment` | Workspace, git, env, and dataset context logging |
| `mlflow_lens.summary` | Structured run summaries as JSON artifacts |
| `mlflow_lens.drift` | PSI / KL / JS drift between training runs |
| `mlflow_lens.panels` | Confusion matrix, ROC, feature importance figures |
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
