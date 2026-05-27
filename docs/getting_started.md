# Get started

Two paths. Pick one — they don't depend on each other.

## Deploy the whole stack

You want the app + the SDK + the tutorial running in your Databricks
workspace, end-to-end.

### Prerequisites

- A Databricks workspace with Unity Catalog and Serverless compute enabled.
- A running SQL warehouse (you'll pass its ID below).
- An approved Spark node type for the refresh job cluster.
- UC privileges to `CREATE CATALOG` and `CREATE SCHEMA`, or pre-created
  control-plane namespace with `MODIFY` grants.
- Databricks CLI authenticated against the workspace.

### Deploy the bundle

```bash
git clone https://github.com/databricks-solutions/model-landscape.git
cd model-landscape
uv build --wheel --out-dir dist

databricks bundle deploy -t warehouse_only \
  --var "sql_warehouse_id=<your-warehouse-id>" \
  --var "refresh_node_type_id=<your-node-type>" \
  --var "control_plane_catalog=<catalog>" \
  --var "control_plane_schema=model_landscape_control_plane"
```

The bundle creates:

- The **Databricks App** (`model-landscape`) with the warehouse app source.
- The **refresh workflow** (`model-landscape-refresh`) that recomputes drift
  / performance / incidents on a schedule.
- The **tutorial job** (`model-landscape-tutorial`) — two notebooks that
  populate demo data.

### Start the app

```bash
databricks apps start model-landscape
databricks apps deploy model-landscape \
  --source-code-path /Workspace/Users/<your-email>/.bundle/model-landscape/warehouse_only/files
```

### Run the tutorial

```bash
databricks bundle run tutorial_mlops
```

This runs two notebooks back-to-back:

1. **`01_train_and_ship`** — generates 100k synthetic fraud transactions,
   trains three candidate models, attaches mlflow-lens panels (ROC,
   confusion matrix, PR curve, classification report, feature importances,
   learning curve) to the best, registers it in Unity Catalog as Champion.
2. **`02_observe_and_recover`** — scores Champion, fast-forwards 60 days of
   inference with a designed drift narrative (a holiday transaction-mix
   shift on day 26 + a broken pipeline null-spike on day 30), stands up
   monitors, runs the bootstrap refresh, then retrains on recent data and
   ships v2.

Open the app at `https://<workspace>/apps/model-landscape`. You should see
two monitors, several open incidents from the designed drift, and the F1
recovery after v2 ships.

For the full deploy reference (variables, targets, troubleshooting),
see [Deploy to a workspace](deploy.md). If you're installing into a
pre-existing Databricks App (constrained workspaces), see
[Deploy into an existing app](existing_app.md).

---

## Just the SDK

You want to enrich your existing MLflow runs without standing up the app.
mlflow-lens is published as a wheel attached to each
[GitHub Release](https://github.com/databricks-solutions/model-landscape/releases).

```bash
pip install "https://github.com/databricks-solutions/model-landscape/releases/download/mlflow-lens-v0.1.0/mlflow_lens-0.1.0-py3-none-any.whl"
```

On Databricks Serverless v5+, all runtime dependencies (`mlflow-skinny`,
`numpy`, `pandas`, `plotly`, `scikit-learn`) are provided.

### Log your first panel

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

The run gets four artifacts:

```text
lens/panels/
├── roc_curve.json         # raw fpr/tpr/auc per class
├── roc_curve.html         # interactive Plotly figure
├── confusion_matrix.json
└── confusion_matrix.html
```

Plus tags `lens.panel.roc_curve=true`, `lens.figure.roc_curve=true`, and
similar for the confusion matrix. Open the run in MLflow → Artifacts →
`lens/panels/`. The `.html` files render inline as interactive figures.

### Without a fitted model

Every quick function has a pre-computed entry point — `.from_scores`,
`.from_predictions`, or `.from_values`:

```python
roc_auc.from_scores(y_true, y_score, log=True)
confusion_matrix.from_predictions(y_true, y_pred, labels=[0, 1], log=True)
```

### What else is there

- [SDK concepts](concepts.md) — the data-plus-figure model, panel
  registry, tag conventions.
- [API reference](api/index.md) — every quick function, every parameter.
- [Panel gallery](gallery/index.md) — one interactive Plotly figure per
  quick function so you can see what each one produces.
