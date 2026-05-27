# Alpha Selection

```python
import numpy as np
from sklearn.linear_model import Ridge
from mlflow_lens.regressor import alpha_selection

fig = alpha_selection(Ridge(), X, y, alphas=np.logspace(-3, 3, 13), log=True)
```

<iframe
    src="_figures/alpha_selection.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference](../api/regressor.md#mlflow_lens.regressor.alpha_selection) for the full signature.
