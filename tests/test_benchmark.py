"""ModelBenchmark — 여러 모델 버전을 '같은 데이터'로 학습·평가해 공정 비교한다."""

import pytest

pytest.importorskip("hmmlearn")

from waf.infrastructure.hmm.benchmark import (  # noqa: E402
    ModelBenchmark,
    VariantResult,
    render_comparison,
)
from waf.infrastructure.hmm.evaluation import (  # noqa: E402
    EvaluationReport,
    classification_metrics,
)
from waf.infrastructure.hmm.model_variant import ModelVariant  # noqa: E402

from synthetic import attack_flow, batch, normal_flow  # noqa: E402


def _result(name: str, roc: float, aucp: float, **variant_kw: object) -> VariantResult:
    """학습 없이 만든 합성 결과 — render_comparison(순수 함수) 테스트용."""
    variant = ModelVariant(name=name, **variant_kw)  # type: ignore[arg-type]
    report = EvaluationReport(
        roc_auc=roc, pr_auc=0.5, partial_auc=aucp, tpr_at_fpr=0.3,
        target_fpr=0.05, threshold_at_fpr=0.0, n_normal=100, n_attack=100,
    )
    clf = classification_metrics(tp=30, fp=5, fn=70, tn=95)
    return VariantResult(variant, report, clf, 100, 50, 200, 0, 100, 100)


def _bench() -> ModelBenchmark:
    normal = batch(normal_flow, 60, seed=1)
    validation = batch(normal_flow, 30, seed=2)
    labeled = [(f, False) for f in batch(normal_flow, 40, seed=3)] + [
        (f, True) for f in batch(attack_flow, 40, seed=4)
    ]
    return ModelBenchmark(normal, validation, labeled)


def test_evaluate_variant_trains_and_scores() -> None:
    bench = _bench()
    variant = ModelVariant(
        name="baseline", n_states=4, n_iter=30, target_fpr=0.1, min_train_sequences=5
    )

    result = bench.evaluate_variant(variant)

    assert isinstance(result, VariantResult)
    assert result.variant is variant
    assert result.train_count == 60
    assert result.validation_count == 30
    assert result.report is not None
    assert 0.0 <= result.report.roc_auc <= 1.0
    assert result.classification is not None


def test_run_returns_one_result_per_variant_in_order() -> None:
    bench = _bench()
    variants = [
        ModelVariant(name="baseline", n_states=4, n_iter=30, target_fpr=0.1, min_train_sequences=5),
        ModelVariant(
            name="hmmpayl", n_states=3, window=4, ensemble_size=2,
            n_iter=30, target_fpr=0.1, min_train_sequences=5, max_fit_windows=1000,
        ),
    ]

    results = bench.run(variants)

    assert [r.variant.name for r in results] == ["baseline", "hmmpayl"]
    assert len(results) == 2


def test_render_lists_each_variant_and_metric_labels() -> None:
    table = render_comparison(
        [_result("baseline", 0.6, 0.10), _result("hmmpayl", 0.7, 0.20, window=4)]
    )
    assert "baseline" in table
    assert "hmmpayl" in table
    assert "ROC" in table
    assert "AUCp" in table


def test_render_sorts_by_partial_auc_descending() -> None:
    # AUCp(HMMPayl 권장 지표)가 높은 변형이 표에서 먼저 나와야 한다.
    table = render_comparison(
        [_result("worse", 0.60, 0.10), _result("better", 0.65, 0.30, window=4)]
    )
    assert table.index("better") < table.index("worse")
