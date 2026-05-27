# Feature Importances

```python
from mlflow_lens.model_selection import feature_importances
fig = feature_importances(model, feature_names, top_n=15, log=True)
```

<iframe
    src="_figures/feature_importances.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/model_selection.md#mlflow_lens.model_selection.feature_importances) for the full signature.
