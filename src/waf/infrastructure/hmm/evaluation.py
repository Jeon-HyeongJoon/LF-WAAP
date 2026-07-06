"""IDS-oriented evaluation metrics.

Per the spec: AIC/BIC are only advisory for model selection. What matters for an
intrusion detector is detection quality at operationally relevant false-positive
rates — ROC-AUC, PR-AUC, TPR @ fixed FPR, and false positives per time unit.

Convention: a NormalHMM `score` is length-normalized log-likelihood where HIGHER
means more normal. So the anomaly score is `-score`, and the positive class is
"attack". `normal_scores` / `attack_scores` are sequences of model scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    roc_auc: float
    pr_auc: float
    partial_auc: float
    tpr_at_fpr: float
    target_fpr: float
    threshold_at_fpr: float
    n_normal: int
    n_attack: int

    def __str__(self) -> str:
        return (
            f"ROC-AUC={self.roc_auc:.3f}  PR-AUC={self.pr_auc:.3f}  "
            f"AUCp@{self.target_fpr:.0%}={self.partial_auc:.3f}  "
            f"TPR@{self.target_fpr:.0%}FPR={self.tpr_at_fpr:.3f}  "
            f"thr={self.threshold_at_fpr:.3f}  (n_normal={self.n_normal}, n_attack={self.n_attack})"
        )


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Operating-point classification quality. Positive class = attack."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def __str__(self) -> str:
        return (
            f"Accuracy={self.accuracy:.3f}  Precision={self.precision:.3f}  "
            f"Recall={self.recall:.3f}  F1={self.f1:.3f}  "
            f"(TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn})"
        )


def classification_metrics(tp: int, fp: int, fn: int, tn: int) -> ClassificationMetrics:
    """Accuracy/precision/recall/F1 from a confusion matrix (positive class = attack).

    Denominators that can be zero (no predicted positives, no actual positives) yield
    0.0 for that metric instead of raising — matching sklearn's zero_division=0.
    """
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ClassificationMetrics(accuracy, precision, recall, f1, tp, fp, fn, tn)


def _labels_and_anomaly_scores(
    normal_scores: Sequence[float], attack_scores: Sequence[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(attack_scores))])
    # anomaly score = -normality score (higher => more likely attack)
    y_anomaly = -np.concatenate([np.asarray(normal_scores), np.asarray(attack_scores)])
    return y_true, y_anomaly


def tpr_at_fpr(
    normal_scores: Sequence[float],
    attack_scores: Sequence[float],
    target_fpr: float = 0.01,
) -> tuple[float, float]:
    """Return (TPR, anomaly-score threshold) at the largest FPR <= target_fpr."""
    y_true, y_anomaly = _labels_and_anomaly_scores(normal_scores, attack_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_anomaly)
    eligible = np.where(fpr <= target_fpr)[0]
    idx = eligible[-1] if eligible.size else 0
    return float(tpr[idx]), float(thresholds[idx])


def partial_auc(
    normal_scores: Sequence[float],
    attack_scores: Sequence[float],
    max_fpr: float = 0.1,
) -> float:
    """AUCp: area under the ROC restricted to FPR in [0, max_fpr], normalized to [0, 1].

    HMMPayl argues that for IDS only the low-false-positive region matters, so the
    ROC is integrated up to max_fpr (default 0.1) and divided by max_fpr.

    Note the normalization: a perfect classifier scores 1.0, but a *random* one
    scores ~max_fpr/2 (≈0.05 at max_fpr=0.1), not 0.5 — this is raw-area/max_fpr,
    not the McClish-standardized partial AUC.
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    y_true, y_anomaly = _labels_and_anomaly_scores(normal_scores, attack_scores)
    fpr, tpr, _ = roc_curve(y_true, y_anomaly)
    if max_fpr >= 1.0:
        return float(auc(fpr, tpr))
    # include a point exactly at max_fpr by interpolating TPR there
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    fpr_p = np.append(fpr[:stop], max_fpr)
    tpr_p = np.append(tpr[:stop], np.interp(max_fpr, fpr, tpr))
    return float(auc(fpr_p, tpr_p) / max_fpr)


def false_positives_per_period(
    normal_scores: Sequence[float],
    threshold_normality: float,
    requests_per_period: float,
) -> float:
    """Expected false positives per period, given observed normal request rate.

    `threshold_normality` is the NormalHMM threshold (anomaly iff score < it).
    """
    scores = np.asarray(normal_scores)
    if scores.size == 0:
        return 0.0
    fp_rate = float(np.mean(scores < threshold_normality))
    return fp_rate * requests_per_period


def evaluate(
    normal_scores: Sequence[float],
    attack_scores: Sequence[float],
    target_fpr: float = 0.01,
) -> EvaluationReport:
    y_true, y_anomaly = _labels_and_anomaly_scores(normal_scores, attack_scores)
    tpr, thr = tpr_at_fpr(normal_scores, attack_scores, target_fpr)
    return EvaluationReport(
        roc_auc=float(roc_auc_score(y_true, y_anomaly)),
        pr_auc=float(average_precision_score(y_true, y_anomaly)),
        partial_auc=partial_auc(normal_scores, attack_scores, max_fpr=target_fpr),
        tpr_at_fpr=tpr,
        target_fpr=target_fpr,
        threshold_at_fpr=thr,
        n_normal=len(normal_scores),
        n_attack=len(attack_scores),
    )
