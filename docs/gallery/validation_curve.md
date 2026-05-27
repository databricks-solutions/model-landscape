# Validation Curve

```python
from mlflow_lens.model_selection import validation_curve
fig = validation_curve(
    model, X, y,
    param_name="max_depth", param_range=[2, 4, 8, 12, 16],
    log=True,
)
```

<iframe
    src="_figures/validation_curve.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/model_selection.md#mlflow_lens.model_selection.validation_curve) for the full signature.
