"""Generate the gallery: trains tiny sklearn models, renders one Plotly HTML
figure per quick function, and writes a Markdown page for each one that
embeds the figure via an ``<iframe>``.

Outputs:
    docs/gallery/_figures/<panel>.html
    docs/gallery/<panel>.md

Re-run any time the panel API or the example datasets change.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
from mlflow_lens.classifier import (
    class_prediction_error,
    classification_report,
    confusion_matrix,
    discrimination_threshold,
    precision_recall,
    roc_auc,
)
from mlflow_lens.model_selection import (
    cv_scores,
    feature_importances,
    learning_curve,
    validation_curve,
)
from mlflow_lens.regressor import alpha_selection, prediction_error, residuals
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

DOCS_DIR = Path(__file__).resolve().parent
GALLERY_DIR = DOCS_DIR / "gallery"
FIG_DIR = GALLERY_DIR / "_figures"

PAGE_TEMPLATE = """\
# {title}

```python
{snippet}
```

<iframe
    src="_figures/{name}.html"
    width="100%"
    height="560"
    frameborder="0"
    style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"
></iframe>

→ See the [API reference]({api_link}) for the full signature.
"""


def write_page(name: str, title: str, snippet: str, api_link: str) -> None:
    page = GALLERY_DIR / f"{name}.md"
    page.write_text(
        PAGE_TEMPLATE.format(
            title=title,
            name=name,
            snippet=textwrap.dedent(snippet).strip(),
            api_link=api_link,
        )
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- classification ----------
    Xb, yb = load_breast_cancer(return_X_y=True)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, random_state=0, test_size=0.3)
    clf_bin = LogisticRegression(max_iter=500).fit(Xb_tr, yb_tr)

    Xi, yi = load_iris(return_X_y=True)
    Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, random_state=0, test_size=0.3)
    clf_mc = LogisticRegression(max_iter=500).fit(Xi_tr, yi_tr)

    panels_cls = [
        (
            "roc_auc",
            "ROC / AUC",
            "../api/classifier.md#mlflow_lens.classifier.roc_auc",
            roc_auc(clf_mc, Xi_te, yi_te),
            """\
            from mlflow_lens.classifier import roc_auc
            fig = roc_auc(model, X_test, y_test, log=True)
            """,
        ),
        (
            "confusion_matrix",
            "Confusion Matrix",
            "../api/classifier.md#mlflow_lens.classifier.confusion_matrix",
            confusion_matrix(clf_mc, Xi_te, yi_te, normalize="true"),
            """\
            from mlflow_lens.classifier import confusion_matrix
            fig = confusion_matrix(model, X_test, y_test, normalize="true", log=True)
            """,
        ),
        (
            "precision_recall",
            "Precision-Recall",
            "../api/classifier.md#mlflow_lens.classifier.precision_recall",
            precision_recall(clf_mc, Xi_te, yi_te),
            """\
            from mlflow_lens.classifier import precision_recall
            fig = precision_recall(model, X_test, y_test, log=True)
            """,
        ),
        (
            "classification_report",
            "Classification Report",
            "../api/classifier.md#mlflow_lens.classifier.classification_report",
            classification_report(clf_mc, Xi_te, yi_te),
            """\
            from mlflow_lens.classifier import classification_report
            fig = classification_report(model, X_test, y_test, log=True)
            """,
        ),
        (
            "class_prediction_error",
            "Class Prediction Error",
            "../api/classifier.md#mlflow_lens.classifier.class_prediction_error",
            class_prediction_error(clf_mc, Xi_te, yi_te),
            """\
            from mlflow_lens.classifier import class_prediction_error
            fig = class_prediction_error(model, X_test, y_test, log=True)
            """,
        ),
        (
            "discrimination_threshold",
            "Discrimination Threshold",
            "../api/classifier.md#mlflow_lens.classifier.discrimination_threshold",
            discrimination_threshold(clf_bin, Xb_te, yb_te, n_thresholds=50),
            """\
            from mlflow_lens.classifier import discrimination_threshold
            fig = discrimination_threshold(model, X_test, y_test, log=True)
            """,
        ),
    ]

    # ---------- regression ----------
    Xd, yd = load_diabetes(return_X_y=True)
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, random_state=0, test_size=0.3)
    reg = LinearRegression().fit(Xd_tr, yd_tr)

    panels_reg = [
        (
            "prediction_error",
            "Prediction Error",
            "../api/regressor.md#mlflow_lens.regressor.prediction_error",
            prediction_error(reg, Xd_te, yd_te),
            """\
            from mlflow_lens.regressor import prediction_error
            fig = prediction_error(model, X_test, y_test, log=True)
            """,
        ),
        (
            "residuals",
            "Residuals",
            "../api/regressor.md#mlflow_lens.regressor.residuals",
            residuals(reg, Xd_te, yd_te, X_train=Xd_tr, y_train=yd_tr),
            """\
            from mlflow_lens.regressor import residuals
            fig = residuals(model, X_test, y_test, X_train=X_train, y_train=y_train, log=True)
            """,
        ),
        (
            "alpha_selection",
            "Alpha Selection",
            "../api/regressor.md#mlflow_lens.regressor.alpha_selection",
            alpha_selection(Ridge(), Xd, yd, alphas=np.logspace(-3, 3, 13), cv=5),
            """\
            import numpy as np
            from sklearn.linear_model import Ridge
            from mlflow_lens.regressor import alpha_selection

            fig = alpha_selection(Ridge(), X, y, alphas=np.logspace(-3, 3, 13), log=True)
            """,
        ),
    ]

    # ---------- model selection ----------
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(Xb, yb)
    feature_names = load_breast_cancer().feature_names.tolist()

    panels_ms = [
        (
            "learning_curve",
            "Learning Curve",
            "../api/model_selection.md#mlflow_lens.model_selection.learning_curve",
            learning_curve(
                LogisticRegression(max_iter=500), Xb, yb, cv=3, train_sizes=np.linspace(0.2, 1.0, 5)
            ),
            """\
            from mlflow_lens.model_selection import learning_curve
            fig = learning_curve(model, X, y, cv=5, log=True)
            """,
        ),
        (
            "validation_curve",
            "Validation Curve",
            "../api/model_selection.md#mlflow_lens.model_selection.validation_curve",
            validation_curve(
                RandomForestClassifier(n_estimators=25, random_state=0),
                Xb,
                yb,
                param_name="max_depth",
                param_range=[2, 4, 8, 12, 16],
                cv=3,
            ),
            """\
            from mlflow_lens.model_selection import validation_curve
            fig = validation_curve(
                model, X, y,
                param_name="max_depth", param_range=[2, 4, 8, 12, 16],
                log=True,
            )
            """,
        ),
        (
            "feature_importances",
            "Feature Importances",
            "../api/model_selection.md#mlflow_lens.model_selection.feature_importances",
            feature_importances(rf, feature_names, top_n=15),
            """\
            from mlflow_lens.model_selection import feature_importances
            fig = feature_importances(model, feature_names, top_n=15, log=True)
            """,
        ),
        (
            "cv_scores",
            "CV Scores",
            "../api/model_selection.md#mlflow_lens.model_selection.cv_scores",
            cv_scores(LogisticRegression(max_iter=500), Xb, yb, cv=5),
            """\
            from mlflow_lens.model_selection import cv_scores
            fig = cv_scores(model, X, y, cv=5, log=True)
            """,
        ),
    ]

    all_panels = panels_cls + panels_reg + panels_ms

    for name, title, api_link, fig, snippet in all_panels:
        out = FIG_DIR / f"{name}.html"
        fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
        write_page(name, title, snippet, api_link)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
