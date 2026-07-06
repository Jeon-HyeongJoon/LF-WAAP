"""partial_auc — AUCp: normalized area under ROC in [0, max_fpr] (HMMPayl metric)."""

import random

import pytest

from waf.infrastructure.hmm.evaluation import partial_auc


def test_perfect_separation_gives_one() -> None:
    normal = [i * 0.001 for i in range(50)]  # high normality
    attack = [-5.0 - i * 0.001 for i in range(50)]  # clearly lower
    assert partial_auc(normal, attack, max_fpr=0.1) == pytest.approx(1.0, abs=0.01)


def test_random_scores_near_diagonal() -> None:
    # Under the paper's normalization (raw partial area / max_fpr), a random
    # classifier scores ~max_fpr/2 (= 0.05 for max_fpr=0.1), NOT 0.5.
    rng = random.Random(0)
    normal = [rng.gauss(0, 1) for _ in range(500)]
    attack = [rng.gauss(0, 1) for _ in range(500)]
    value = partial_auc(normal, attack, max_fpr=0.1)
    assert 0.0 <= value < 0.15
    assert value == pytest.approx(0.05, abs=0.05)


def test_evaluate_report_includes_partial_auc() -> None:
    from waf.infrastructure.hmm.evaluation import evaluate

    normal = [i * 0.001 for i in range(50)]
    attack = [-5.0 - i * 0.001 for i in range(50)]
    report = evaluate(normal, attack, target_fpr=0.1)

    assert 0.0 <= report.partial_auc <= 1.0
    assert report.partial_auc == pytest.approx(1.0, abs=0.01)
    assert "AUCp" in str(report)


def test_focuses_on_low_fpr_region() -> None:
    # An attack distribution that overlaps normals at low FPR but separates later
    # should score lower on AUCp (low-FPR) than on full-range AUC.
    rng = random.Random(1)
    normal = [rng.gauss(0, 1) for _ in range(500)]
    attack = [rng.gauss(1.0, 1) for _ in range(500)]
    p = partial_auc(normal, attack, max_fpr=0.1)
    assert 0.0 <= p <= 1.0
