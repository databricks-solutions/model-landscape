# Classification Report

```python
from mlflow_lens.classifier import classification_report
fig = classification_report(model, X_test, y_test, log=True)
```

<iframe
    src="_figures/classification_report.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/classifier.md#mlflow_lens.classifier.classification_report) for the full signature.
