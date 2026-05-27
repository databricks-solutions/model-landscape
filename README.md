# Model Landscape

Databricks-native model observability. Two halves of one product:

- **mlflow-lens** — a small SDK that enriches MLflow runs with interactive
  Plotly panels (classification, regression, model selection), training-time
  drift, structured summaries, and workspace context.
- **The warehouse app** — a Databricks App that monitors any Unity Catalog
  inference table for drift, performance degradation, data-quality
  regressions, and incidents. Data never leaves the workspace.

## Docs

The full documentation site is the canonical source — install guide,
architecture, SDK reference, panel gallery, and the intro deck all live
there:

→ **https://databricks-solutions.github.io/model-landscape/**

Run it locally:

```bash
uv pip install mkdocs-material 'mkdocstrings[python]>=0.26' \
               mkdocs-include-markdown-plugin
uv run python docs/_build_gallery.py
uv run mkdocs serve
```

## Quick deploy

```bash
git clone https://github.com/databricks-solutions/model-landscape.git
cd model-landscape
uv build --wheel --out-dir dist

databricks bundle deploy -t warehouse_only \
  --var "sql_warehouse_id=<id>" \
  --var "refresh_node_type_id=<node-type>" \
  --var "control_plane_catalog=<catalog>" \
  --var "control_plane_schema=model_landscape_control_plane"

databricks apps start model-landscape
databricks apps deploy model-landscape \
  --source-code-path /Workspace/Users/<email>/.bundle/model-landscape/warehouse_only/files
databricks bundle run tutorial_mlops
```

Full first-time-setup guide: [Get started](https://databricks-solutions.github.io/model-landscape/getting_started/).

## Local development

```bash
uv sync --extra dev               # install workspace + dev deps
uv run pytest -q                  # run tests
uv build --wheel --out-dir dist   # build SDK + app wheel
uv run python -m model_landscape.app  # run app locally
```

## License

See [LICENSE.md](LICENSE.md). Third-party notices in [NOTICE.txt](NOTICE.txt).
