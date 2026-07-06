"""CLI pipeline integration test on a tiny CSV — does not read the real dataset."""

import pytest

pytest.importorskip("hmmlearn")

import calibrate_cli as cli  # noqa: E402

HEADER = "classification,Method,host,content,URL"


def _normal(i: int) -> str:
    return f"0,GET,localhost:8080,,http://localhost:8080/p{i} HTTP/1.1"


def _anomalous(i: int) -> str:
    return f"1,GET,localhost:8080,,http://localhost:8080/q{i} HTTP/1.1"


def _write(path, rows: list[str]) -> None:
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="latin-1")


def test_train_writes_model_and_reports_counts(tmp_path) -> None:
    normal_csv = tmp_path / "normal.csv"
    _write(normal_csv, [_normal(i) for i in range(50)])
    model = tmp_path / "model.pkl"

    report = cli.train_model(
        normal_csv, model, validation_ratio=0.2, n_states=4, n_iter=40, target_fpr=0.1
    )

    assert model.exists()
    assert report.normal_count == 40
    assert report.validation_count == 10


def test_train_uses_chosen_extractor(tmp_path) -> None:
    from waf.infrastructure.adapter.persistence import PickleModelRepository

    normal_csv = tmp_path / "normal.csv"
    _write(normal_csv, [_normal(i) for i in range(50)])
    model = tmp_path / "model.pkl"

    cli.train_model(
        normal_csv,
        model,
        extractor="raw_byte",
        validation_ratio=0.2,
        n_states=4,
        n_iter=30,
        target_fpr=0.1,
    )

    detector = PickleModelRepository(model).load()
    assert detector.extractor_name == "raw_byte"


def test_evaluate_loads_model_and_computes_metrics(tmp_path) -> None:
    normal_csv = tmp_path / "normal.csv"
    _write(normal_csv, [_normal(i) for i in range(50)])
    model = tmp_path / "model.pkl"
    cli.train_model(normal_csv, model, validation_ratio=0.2, n_states=4, n_iter=40, target_fpr=0.1)

    eval_csv = tmp_path / "eval.csv"
    _write(eval_csv, [_normal(100 + i) for i in range(20)] + [_anomalous(i) for i in range(20)])

    outcome = cli.evaluate_model(eval_csv, model, target_fpr=0.1)

    assert outcome.covered > 0
    assert outcome.report is not None
    assert outcome.report.n_normal == 20
    assert outcome.report.n_attack == 20
    assert 0.0 <= outcome.report.roc_auc <= 1.0

    # report card values are populated end to end
    assert outcome.classification is not None
    cm = outcome.classification
    assert cm.tp + cm.fp + cm.fn + cm.tn == outcome.covered
    assert outcome.normal_stats.count == 20
    assert outcome.attack_stats.count == 20
    # structure values needed by the report come straight off the loaded model
    assert outcome.detector.n_states == 4
    assert outcome.detector.n_symbols == 8  # byte_class default
    # train-time counts were captured and survived the pickle round-trip
    assert outcome.detector.train_count == 40
    assert outcome.detector.validation_count == 10


def test_compare_runs_multiple_variants_on_same_data(tmp_path) -> None:
    from waf.infrastructure.hmm.model_variant import ModelVariant

    normal_csv = tmp_path / "normal.csv"
    _write(normal_csv, [_normal(i) for i in range(50)])
    eval_csv = tmp_path / "eval.csv"
    _write(eval_csv, [_normal(100 + i) for i in range(20)] + [_anomalous(i) for i in range(20)])

    variants = [
        ModelVariant(name="baseline", n_states=4, n_iter=30, target_fpr=0.1, min_train_sequences=5),
        ModelVariant(
            name="window", n_states=3, window=4, n_iter=30,
            target_fpr=0.1, min_train_sequences=5, max_fit_windows=500,
        ),
    ]

    results = cli.compare_models(normal_csv, eval_csv, variants, validation_ratio=0.2)

    assert [r.variant.name for r in results] == ["baseline", "window"]
    # 모든 변형이 같은 학습 데이터(정상 40개)를 봤다
    assert all(r.train_count == 40 for r in results)


def test_limit_balances_classes_so_both_are_present(tmp_path) -> None:
    from waf.infrastructure.hmm.model_variant import ModelVariant

    normal_csv = tmp_path / "normal.csv"
    _write(normal_csv, [_normal(i) for i in range(50)])
    # CSIC처럼 '정상 먼저, 공격 나중' 순서로 작성 — 앞에서 자르면 공격이 0개가 되는 함정.
    eval_csv = tmp_path / "eval.csv"
    _write(eval_csv, [_normal(100 + i) for i in range(30)] + [_anomalous(i) for i in range(30)])

    variants = [
        ModelVariant(name="baseline", n_states=4, n_iter=30, target_fpr=0.1, min_train_sequences=5)
    ]
    results = cli.compare_models(normal_csv, eval_csv, variants, validation_ratio=0.2, limit=20)

    result = results[0]
    assert result.eval_attack > 0  # 수정 전엔 0(선두 정상만 잘림) → 지표 n/a였다
    assert result.eval_normal > 0
    assert result.report is not None
