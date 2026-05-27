# Learning Curve

```python
from mlflow_lens.model_selection import learning_curve
fig = learning_curve(model, X, y, cv=5, log=True)
```

<iframe
    src="_figures/learning_curve.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/model_selection.md#mlflow_lens.model_selection.learning_curve) for the full signature.
