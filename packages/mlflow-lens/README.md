# mlflow-lens

MLflow enrichment SDK for training-time observability. Adds structured run
summaries, training-time drift detection, cost attribution, and standard
classification/regression panels (confusion matrix, ROC, feature importance)
to any MLflow tracking run.

## Install

```bash
pip install mlflow-lens
```

For the Spark `export` module (`mlflow_lens.export.to_delta`):

```bash
pip install mlflow-lens[spark]
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
and metric definitions.
