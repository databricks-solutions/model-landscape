# Releasing mlflow-lens

`mlflow-lens` ships through GitHub Releases. Builds happen in CI on tag
push (`.github/workflows/mlflow-lens-release.yml`) — there is no manual
upload step.

## Cut a release

1. **Decide the version** following [SemVer](https://semver.org/). Patch
   bumps for bugfixes, minor for backward-compatible features, major for
   breaking changes.

2. **Bump the version** in `packages/mlflow-lens/src/mlflow_lens/_version.py`
   on a feature branch:

   ```python
   __version__ = "0.2.0"
   ```

   Open a PR into `dev`, get review, merge.

3. **Merge `dev` into `main`** via PR when the release is ready to cut.

4. **Tag `main`** with the prefixed version:

   ```bash
   git checkout main && git pull
   git tag mlflow-lens-v0.2.0
   git push origin mlflow-lens-v0.2.0
   ```

5. CI builds the wheel and sdist, runs `twine check`, smoke-installs into a
   clean venv, and publishes a GitHub Release with:
   - `mlflow_lens-<version>-py3-none-any.whl`
   - `mlflow_lens-<version>.tar.gz`
   - `SHA256SUMS`
   - Auto-generated release notes from merged PRs.

6. **Verify the release**: open
   [/releases](https://github.com/databricks-solutions/model-landscape/releases?q=mlflow-lens),
   click the new release, and confirm the three artifacts are attached and
   the install line in the body matches the tag.

## Pre-flight checks

The workflow also runs on any PR that touches `packages/mlflow-lens/` or
the workflow file. That build does everything except the GitHub Release
step, so packaging regressions are caught before merge.

## If something goes wrong

- **Tag/version mismatch**: the workflow fails the build step if the tag
  (`mlflow-lens-vX.Y.Z`) doesn't match `_version.py`. Delete the bad tag
  (`git push --delete origin mlflow-lens-vX.Y.Z`), fix `_version.py`, and
  re-tag.

- **Release published with broken artifacts**: GitHub Releases are
  immutable in spirit but mutable in practice. Edit the release on GitHub
  to mark it as a draft or pre-release, fix the code, bump to the next
  patch version, and re-release. Don't reuse the same version number.

- **Need to yank a release**: edit the release on GitHub and tick "Mark as
  pre-release" + add a `YANKED:` note to the body. Users with the URL
  pinned will still install it; downstream PRs should bump past it.
