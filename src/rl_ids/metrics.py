"""Pure, dependency-light classification metrics for the holdout evaluation.

Kept free of sklearn/torch so they are trivially unit-testable and reusable by
the report writer.  All functions take plain numpy/sequence inputs.

Definitions used throughout the IDS evaluation:

    precision_c = TP_c / (TP_c + FP_c)      # of things we flagged c, how many were c
    recall_c    = TP_c / (TP_c + FN_c)      # of the true c, how many we caught
    F1_c        = 2 P R / (P + R)
    FPR_c       = FP_c / (FP_c + TN_c)      # one-vs-rest false-positive rate
"""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred, labels) -> np.ndarray:
    """Confusion matrix, rows = true label, cols = predicted label."""
    labels = list(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        matrix[index[t], index[p]] += 1
    return matrix


def per_class_metrics(y_true, y_pred, labels) -> dict[str, dict[str, float]]:
    """One-vs-rest precision / recall / F1 / FPR + support for each label."""
    labels = list(labels)
    matrix = confusion(y_true, y_pred, labels)
    total = int(matrix.sum())
    out: dict[str, dict[str, float]] = {}
    for i, lab in enumerate(labels):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        out[lab] = {
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "support": int(matrix[i, :].sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    return out


def binary_metrics(y_true_pos, y_pred_pos) -> dict[str, float]:
    """Precision / recall / F1 / FPR for a single positive class."""
    y_true_pos = np.asarray(y_true_pos, dtype=bool)
    y_pred_pos = np.asarray(y_pred_pos, dtype=bool)
    tp = int(np.sum(y_true_pos & y_pred_pos))
    fp = int(np.sum(~y_true_pos & y_pred_pos))
    fn = int(np.sum(y_true_pos & ~y_pred_pos))
    tn = int(np.sum(~y_true_pos & ~y_pred_pos))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": int(y_true_pos.sum()),
    }


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def macro_average(report: dict[str, dict[str, float]], metric: str) -> float:
    """Unweighted mean of ``metric`` across labels (macro)."""
    vals = [v[metric] for v in report.values()]
    return float(np.mean(vals)) if vals else 0.0


def weighted_average(report: dict[str, dict[str, float]], metric: str) -> float:
    """Support-weighted mean of ``metric`` across labels."""
    supports = np.asarray([v["support"] for v in report.values()], dtype=float)
    vals = np.asarray([v[metric] for v in report.values()], dtype=float)
    total = supports.sum()
    return float((vals * supports).sum() / total) if total else 0.0
