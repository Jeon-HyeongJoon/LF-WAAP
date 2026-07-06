"""ModelVariant — 재사용 가능한 'HMM 모델 버전' 정의.

하나의 변형(variant)은 이름 + 하이퍼파라미터 묶음이며, build_detector()로 동일 설정의
HmmDetector를 몇 번이고 다시 만들 수 있다(학습/평가에서 재사용).
"""

from waf.infrastructure.hmm.model_variant import ABLATION_MATRIX, ModelVariant


def test_build_detector_applies_config() -> None:
    variant = ModelVariant(name="x", extractor="raw_byte", n_states=5, window=4, ensemble_size=2)

    detector = variant.build_detector()

    assert detector.extractor_name == "raw_byte"
    assert detector.n_states == 5
    assert detector.window == 4
    assert detector.ensemble_size == 2


def test_uses_hmmpayl_flags_window_or_ensemble() -> None:
    plain = ModelVariant(name="baseline")  # window=0, ensemble=1
    windowed = ModelVariant(name="w", window=6)
    ensembled = ModelVariant(name="e", ensemble_size=3)
    both = ModelVariant(name="both", window=6, ensemble_size=3)

    assert plain.uses_hmmpayl is False
    assert windowed.uses_hmmpayl is True
    assert ensembled.uses_hmmpayl is True
    assert both.uses_hmmpayl is True


def test_ablation_matrix_covers_all_idea_combinations() -> None:
    names = [v.name for v in ABLATION_MATRIX]
    assert len(names) == len(set(names))  # 이름 중복 없음

    # (윈도우 on?, 앙상블 on?) 네 조합이 모두 들어 있어 각 아이디어 기여를 분리할 수 있다.
    combos = {(v.window > 0, v.ensemble_size > 1) for v in ABLATION_MATRIX}
    assert (False, False) in combos  # baseline (HMMPayl 미적용)
    assert (True, False) in combos  # 윈도우만
    assert (False, True) in combos  # 앙상블만
    assert (True, True) in combos  # 둘 다

    # HMMPayl 미적용 베이스라인이 최소 하나 존재
    assert any(not v.uses_hmmpayl for v in ABLATION_MATRIX)


def test_ablation_presets_are_interactive_friendly() -> None:
    # 비교(compare)가 분 단위로 끝나도록 프리셋은 가벼운 학습 비용을 쓴다.
    # (무거운 n_iter/윈도우 상한은 raw_byte+윈도우+앙상블에서 비현실적으로 느렸음)
    for variant in ABLATION_MATRIX:
        assert variant.n_iter <= 40
        if variant.window > 0:
            assert variant.max_fit_windows is not None
            assert variant.max_fit_windows <= 10_000
