"""Classification metrics from true labels and predicted probabilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Return accuracy, AUC, precision, recall, F1 and confusion-matrix counts.

    - AUC uses the raw probabilities: it measures how well the model *ranks*
      tumour above normal patches, independent of any threshold.
    - The other metrics use hard predictions (prob >= threshold).
    - Precision/recall/F1 treat tumour (label 1) as the positive class.
      Recall = fraction of tumour patches we catch (sensitivity).
    """
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    # AUC is undefined if only one class is present (can happen on tiny subsets)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else None

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": None if auc is None else float(auc),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(labels)),
    }
