<p align="center">
  <a href="https://github.com/databricks-solutions/model-landscape">
    <img src="docs/model-lens-logo.svg" width="220" height="220" alt="Model Landscape logo" />
  </a>
</p>

<p align="center">
  <b>Databricks-native model observability — catch drift before it costs you.</b>
</p>

<div align="center">

![Databricks](https://img.shields.io/badge/Databricks-FF3621?logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-008DE4?logo=plotly&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)
[![Docs](https://img.shields.io/badge/docs-databricks--solutions.github.io-FF3621)](https://databricks-solutions.github.io/model-landscape/)

</div>

---

Model Landscape is two halves of one product:

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

`databricks bundle deploy` rebuilds the wheel automatically — no manual
`uv build` needed.

```bash
git clone https://github.com/databricks-solutions/model-landscape.git
cd model-landscape

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
uv sync --extra dev                       # install workspace + dev deps
uv run model-landscape-init-hooks         # one-time: wire .githooks/pre-commit
uv run pytest -q                          # run tests
uv build --wheel --out-dir dist           # build SDK + app wheel
uv run python -m model_landscape.app      # run app locally
```

`model-landscape-init-hooks` points git at `.githooks/`, where the
checked-in pre-commit hook runs `ruff check` and `ruff format --check`
on staged Python files before each commit. Run it once per fresh clone.

## License

See [LICENSE.md](LICENSE.md). Third-party notices in [NOTICE.txt](NOTICE.txt).
