import numpy as np

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions
        )
    }


def find_best_threshold(y_true, scores, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = None
    best_f1 = -1

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold