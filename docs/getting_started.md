# Getting started

## Install

`mlflow-lens` is published as a wheel attached to each
[GitHub Release](https://github.com/databricks-solutions/model-landscape/releases).

```bash
pip install "https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.1.0/mlflow_lens-0.1.0-py3-none-any.whl"
```

On Databricks Serverless v5+, mlflow-lens's runtime dependencies
(`mlflow-skinny`, `numpy`, `pandas`, `plotly`, `scikit-learn`) are provided
out of the box.

## Log your first panel

```python
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlflow_lens.classifier import roc_auc, confusion_matrix

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

model = LogisticRegression(max_iter=500).fit(X_tr, y_tr)

mlflow.set_experiment("/Users/me@databricks.com/mlflow-lens-demo")
with mlflow.start_run():
    roc_auc(model, X_te, y_te, log=True)
    confusion_matrix(model, X_te, y_te, log=True)
```

What this logs to the run:

```text
lens/panels/
├── roc_curve.json          # raw fpr/tpr/auc per class
├── roc_curve.html          # interactive Plotly figure
├── confusion_matrix.json   # raw matrix
└── confusion_matrix.html
```

And these tags:

```text
lens.version            = 0.1.0
lens.panel.roc_curve    = true
lens.figure.roc_curve   = true
lens.panel.confusion_matrix  = true
lens.figure.confusion_matrix = true
```

Open the run in the MLflow UI: the `.html` files render inline as
interactive Plotly figures; the `.json` files are consumed by the Lens
warehouse app for cross-run comparison.

## Without a model object

Every quick function also exposes a pre-computed entry point — use it
when you already have predictions or scores and don't need to pass a
fitted estimator.

```python
from mlflow_lens.classifier import roc_auc, confusion_matrix

# Pre-computed positive-class scores
roc_auc.from_scores(y_true, y_score, log=True)

# Pre-computed predictions
confusion_matrix.from_predictions(y_true, y_pred, labels=[0, 1], log=True)
```

## Without an active run

Pass an explicit `run_id=` to log into a specific run:

```python
roc_auc(model, X_te, y_te, log=True, run_id="abc123def456")
```

Calling with `log=True` outside an active run and without `run_id=`
raises a clear error.
