# Contributing to Model Landscape

This repository is maintained by Databricks and intended for contributions from Databricks Field Engineers. While the repository is public and meant to accelerate machine learning projects that use Databricks, external contributions are not currently accepted. Feel free to open an issue with requests or suggestions.

## Code Standards

- **Python**: Follow PEP 8 conventions
- **Documentation**: Add to docs/ folder
- **Type hints**: Include type annotations for public functions
- **Naming**: Use lowercase with hyphens for directories (e.g., `databricks-tools-core`)

## Linting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run these before submitting a PR:

```bash
# Check for linting errors
uvx ruff@0.11.0 check \
  --select=E,F,B,PIE \
  --ignore=E401,E402,F401,F403,B017,B904,ANN,TCH \
  --line-length=120 \
  --target-version=py311 \
  databricks-tools-core/ databricks-mcp-server/

# Auto-fix linting errors where possible
uvx ruff@0.11.0 check --fix \
  --select=E,F,B,PIE \
  --ignore=E401,E402,F401,F403,B017,B904,ANN,TCH \
  --line-length=120 \
  --target-version=py311 \
  databricks-tools-core/ databricks-mcp-server/

# Check formatting
uvx ruff@0.11.0 format --check \
  --line-length=120 \
  --target-version=py311 \
  databricks-tools-core/ databricks-mcp-server/

# Auto-format code
uvx ruff@0.11.0 format \
  --line-length=120 \
  --target-version=py311 \
  databricks-tools-core/ databricks-mcp-server/
```

## Testing

Run unit and integration tests before submitting changes:

```bash
cd model-landscape
uv run pytest tests/ -v
```

Ensure your changes work with a live Databricks workspace.

## Pull Request Process

1. Create a feature branch from `main` (fork repo is necessary)
2. Make your changes with clear, descriptive commits
3. Test your changes against a Databricks workspace
4. Open a PR with:
   - Brief description of the change
   - Any relevant context or motivation
   - Testing performed
5. Address review feedback

## Security

- Never commit credentials, tokens, or sensitive data
- Use synthetic data for examples and tests
- Review changes for potential security issues before submitting

## License

By submitting a contribution, you agree that your contributions will be licensed under the same terms as the project (see [LICENSE.md](LICENSE.md)).

You certify that:
- You have the right to submit the contribution
- Your contribution does not include confidential or proprietary information
- You grant Databricks the right to use, modify, and distribute your contribution