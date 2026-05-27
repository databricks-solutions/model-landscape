# Residuals

```python
from mlflow_lens.regressor import residuals
fig = residuals(model, X_test, y_test, X_train=X_train, y_train=y_train, log=True)
```

<iframe
    src="_figures/residuals.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/regressor.md#mlflow_lens.regressor.residuals) for the full signature.
