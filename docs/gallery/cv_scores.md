# CV Scores

```python
from mlflow_lens.model_selection import cv_scores
fig = cv_scores(model, X, y, cv=5, log=True)
```

<iframe
    src="_figures/cv_scores.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/model_selection.md#mlflow_lens.model_selection.cv_scores) for the full signature.
