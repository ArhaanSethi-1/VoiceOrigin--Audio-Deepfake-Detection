"""Metrics, threshold tuning, and plots for binary spoof detection."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def eer_score(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i])


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores, pos_label=1)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def calculate_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    eer, eer_threshold = eer_score(y_true, scores)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_spoof": float(precision_score(y_true, pred, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, pred, zero_division=0)),
        "specificity_genuine": float(specificity),
        "false_positive_rate_genuine_flagged_spoof": float(1.0 - specificity),
        "false_negative_rate_spoof_missed": float(fn / max(fn + tp, 1)),
        "f1_spoof": float(f1_score(y_true, pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc": auc, "eer": eer, "eer_threshold": eer_threshold,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["genuine", "spoof"],
    }


def save_plots(y_true: np.ndarray, scores: np.ndarray, threshold: float, outdir: Path, prefix: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, scores >= threshold, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Genuine", "Spoof"], yticklabels=["Genuine", "Spoof"],
           xlabel="Predicted", ylabel="Actual", title=f"{prefix}: confusion matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_confusion.png", dpi=150)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC {roc_auc_score(y_true, scores):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{prefix}: ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_roc.png", dpi=150)
    plt.close(fig)
