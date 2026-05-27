# Development notes

Working notes for the people shipping Model Landscape. Keep it short —
when something becomes long-lived, move it into the docs site
(`docs/`) or `mlflow-lens/RELEASING.md`.

## Current state

| Branch | Head | Status |
|---|---|---|
| `main` | published code (`_version.py` = `0.1.0`) | stable, no `mlflow-lens-v*` release tag yet |
| `dev` | integration branch | tracks merged feature work |
| `mlflow-lens-evolution` | this branch | doc polish + apx patterns + version bump to `0.2.0`; **not yet merged** |

No GitHub Releases exist yet. The release workflow
(`.github/workflows/mlflow-lens-release.yml`) **only fires on
`mlflow-lens-v*` tag push** — merging a branch into `main` doesn't ship
anything. Shipping is always a deliberate tag-and-push step.

## Path to the next release (mlflow-lens-v0.2.0)

The full runbook is in [`mlflow-lens/RELEASING.md`](mlflow-lens/RELEASING.md).
The concrete next steps for this product cycle:

1. Open a PR from `mlflow-lens-evolution` into `dev` (or directly to
   `main` if skipping the staging step). It carries the
   `_version.py` bump from `0.1.0` → `0.2.0` along with the panel
   work, docs site, apx pattern adoption, and the ruff format pass.
2. Land the PR through to `main`.
3. From `main`:

   ```bash
   git checkout main && git pull
   git tag mlflow-lens-v0.2.0
   git push origin mlflow-lens-v0.2.0
   ```

4. The release workflow runs the parity check (passes — `_version.py`
   is `0.2.0`), builds the wheel + sdist, `twine check`s, smoke-installs
   into a clean venv, and publishes the GitHub Release with the wheel,
   sdist, and `SHA256SUMS` attached.

If the parity check fails because `_version.py` and the tag suffix
don't match, the recovery is: delete the bad tag
(`git push --delete origin mlflow-lens-v0.2.0`), fix the version
file, re-tag.

## In flight

- Tutorial smoke-test in a real Databricks workspace
  (`databricks bundle deploy -t warehouse_only` + `databricks bundle run tutorial_mlops`).
- Visual review of the published docs site once the GitHub Pages
  workflow lands the build on `main`.

## Conventions worth keeping

- One coherent stream of work = one feature branch = one PR. Use
  commits within the PR for logical separation (see the
  `feature/split-mlflow-lens` PR shape).
- `databricks bundle deploy` rebuilds the wheel via the `artifacts`
  block — don't run `uv build` manually before deploying.
- The pre-commit hook (`.githooks/pre-commit`) runs ruff against
  staged Python files. Run `uv run model-landscape-init-hooks` once
  per fresh clone to wire it.
- Don't add Databricks-notebook code to `model_landscape/` or
  `mlflow-lens/src/` — those are pure Python packages that ruff lints
  in CI. Notebooks (`tutorial/`, `notebooks/`) are excluded from lint
  because of `%run` and implicit globals.
