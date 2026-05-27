# Confusion Matrix

```python
from mlflow_lens.classifier import confusion_matrix
fig = confusion_matrix(model, X_test, y_test, normalize="true", log=True)
```

<iframe
    src="_figures/confusion_matrix.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/classifier.md#mlflow_lens.classifier.confusion_matrix) for the full signature.
