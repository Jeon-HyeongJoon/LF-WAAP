"""classification_metrics — accuracy/precision/recall/F1 from confusion counts.

For the report (content.md "보고서에 포함할 수치"): the deployed model decides per
the per-partition calibrated threshold, so the report's classification metrics come
from the realized confusion matrix, not a re-thresholding of the score arrays.
Positive class = attack.
"""

import pytest

from waf.infrastructure.hmm.evaluation import ClassificationMetrics, classification_metrics


def test_perfect_classifier() -> None:
    m = classification_metrics(tp=10, fp=0, fn=0, tn=10)
    assert m.accuracy == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_known_confusion_matrix() -> None:
    # TP=8, FP=2, FN=4, TN=6  -> from a 20-sample run
    m = classification_metrics(tp=8, fp=2, fn=4, tn=6)
    assert m.accuracy == pytest.approx((8 + 6) / 20)
    assert m.precision == pytest.approx(8 / 10)
    assert m.recall == pytest.approx(8 / 12)
    assert m.f1 == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))


def test_no_predicted_positives_does_not_divide_by_zero() -> None:
    m = classification_metrics(tp=0, fp=0, fn=5, tn=5)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.accuracy == pytest.approx(0.5)


def test_str_lists_metrics_and_confusion() -> None:
    m = classification_metrics(tp=8, fp=2, fn=4, tn=6)
    text = str(m)
    for token in ("Accuracy", "Precision", "Recall", "F1", "TP=8", "FP=2", "FN=4", "TN=6"):
        assert token in text


def test_returns_dataclass_with_counts() -> None:
    m = classification_metrics(tp=8, fp=2, fn=4, tn=6)
    assert isinstance(m, ClassificationMetrics)
    assert (m.tp, m.fp, m.fn, m.tn) == (8, 2, 4, 6)
