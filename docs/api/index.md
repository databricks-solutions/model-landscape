# API Reference

The mlflow-lens Python API is organized into:

- **Panel quick functions** — the user-facing one-liners that produce
  Plotly figures and log them to MLflow.
    - [Classifier panels](classifier.md)
    - [Regressor panels](regressor.md)
    - [Model-selection panels](model_selection.md)
- **Low-level panel artifact API** — `log_panel`, `PANEL_TYPES`.
    - [Panels](panels.md)
- **Run helpers** — drift, summary, cost, experiment context, export.
    - [Drift](drift.md)
    - [Summary](summary.md)
