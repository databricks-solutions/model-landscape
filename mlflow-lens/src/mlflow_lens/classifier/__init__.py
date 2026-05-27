"""Classifier panels: ROC/AUC, confusion matrix, precision-recall, etc."""

from mlflow_lens.classifier.class_prediction_error import class_prediction_error
from mlflow_lens.classifier.classification_report import classification_report
from mlflow_lens.classifier.confusion_matrix import confusion_matrix
from mlflow_lens.classifier.discrimination_threshold import discrimination_threshold
from mlflow_lens.classifier.precision_recall import precision_recall
from mlflow_lens.classifier.roc_auc import roc_auc

__all__ = [
    "class_prediction_error",
    "classification_report",
    "confusion_matrix",
    "discrimination_threshold",
    "precision_recall",
    "roc_auc",
]
