---
title: Model Landscape
hide:
  - navigation
---

# Catch model drift before it costs you.

**Model Landscape** is a Databricks-native observability stack for production
ML. It's two things that work together:

- **mlflow-lens** — a small SDK that enriches your MLflow runs with
  interactive Plotly panels, structured summaries, training-time drift, and
  workspace context. One line per panel, two artifacts per panel
  (`lens/panels/<type>.json` for the app to read, `<type>.html` for you to
  click in the MLflow UI).
- **The warehouse app** — a Databricks App that monitors Unity Catalog
  inference tables for drift, performance degradation, data-quality issues,
  and incidents. The data never leaves your workspace.

Together they answer the question every ML team eventually has to: **is the
model still working, and if not, what changed?**

[Check out the intro deck](intro-deck.html){ .md-button .md-button--primary }
[Get started](getting_started.md){ .md-button }

---

## Two halves of one product

<div class="grid cards" markdown>

- **The warehouse app**

    ---

    Point it at any Unity Catalog inference table. It computes daily drift
    (PSI / JS / KL), per-bin performance, data-quality signals, and opens
    incidents when thresholds are breached. Designed for fleets, not single
    models.

    [Architecture](architecture.md) · [Deploy](deploy.md) · [Existing-app install](existing_app.md)

- **The mlflow-lens SDK**

    ---

    Yellowbrick-style quick functions, Plotly-only, that log both the raw
    data and an interactive figure to your MLflow run. Classification,
    regression, and model-selection panels in one import.

    [SDK concepts](concepts.md) · [API reference](api/index.md) · [Panel gallery](gallery/index.md)

</div>

## What you do with it

```python
import mlflow
from mlflow_lens.classifier import roc_auc, confusion_matrix

with mlflow.start_run():
    roc_auc(model, X_test, y_test, log=True)
    confusion_matrix(model, X_test, y_test, log=True)
```

Then point the warehouse app at the inference table this model is scoring
in production. When the holiday season ships a 25-point shift in
`merchant_category`, the app opens an incident on day 1 — not day 14.

[See the end-to-end tutorial →](getting_started.md#run-the-tutorial)

---

## Status

`v0.2.0` — alpha. Panel JSON schemas are versioned (`schema_version: "1"`)
and considered stable; the Python API may evolve before 1.0. See the
[changelog](changelog.md) for what shipped in each release.
