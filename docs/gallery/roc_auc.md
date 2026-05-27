# ROC / AUC

```python
from mlflow_lens.classifier import roc_auc
fig = roc_auc(model, X_test, y_test, log=True)
```

<iframe
    src="_figures/roc_auc.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/classifier.md#mlflow_lens.classifier.roc_auc) for the full signature.
