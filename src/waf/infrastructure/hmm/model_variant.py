"""ModelVariant — 재사용 가능한 'HMM 모델 버전' 정의.

POC에서 여러 모델 구성을 나란히 비교하려면, 각 구성을 '이름 붙은 하이퍼파라미터 묶음'
으로 한 곳에 선언해 두고 필요할 때마다 동일 설정의 탐지기를 찍어내는 편이 깔끔하다.
HMMPayl이 제안한 세 아이디어 중 ①슬라이딩 윈도우(window)·②K-앙상블(ensemble_size)이
바로 이 변형 파라미터로 표현되고, ③부분 AUC(AUCp)는 평가 단계의 지표로 다룬다.
"""

from __future__ import annotations

from dataclasses import dataclass

from waf.infrastructure.hmm.features import extractor_by_name
from waf.infrastructure.hmm.hmm_detector import HmmDetector


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """하나의 모델 버전. frozen이라 그 자체로 안전하게 공유·재사용된다."""

    name: str
    extractor: str = "byte_class"  # 관측 심볼화 방식 (byte_class / raw_byte / window_token)
    n_states: int = 8  # HMM 숨은 상태(잠재 국면) 수
    window: int = 0  # HMMPayl 아이디어 ① 슬라이딩 윈도우 폭 (0=off)
    ensemble_size: int = 1  # HMMPayl 아이디어 ② 랜덤 재시작 K-앙상블 (1=off)
    target_fpr: float = 0.05  # 임계치 보정·지표 산출의 목표 오탐률
    n_iter: int = 100  # Baum–Welch 반복 횟수
    max_fit_windows: int | None = None  # 학습 윈도우 상한(샘플링, 비용 제어)
    min_train_sequences: int = 10  # 파티션 학습 최소 표본 수(미만이면 해당 파티션 건너뜀)

    @property
    def uses_hmmpayl(self) -> bool:
        """HMMPayl 아이디어(①슬라이딩 윈도우 또는 ②앙상블)가 켜져 있으면 True.

        둘 다 꺼진(window=0, ensemble=1) 구성이 곧 '순수 one-class HMM' 베이스라인이다.
        """
        return self.window > 0 or self.ensemble_size > 1

    def build_detector(self) -> HmmDetector:
        """이 변형 설정 그대로 새 HmmDetector 인스턴스를 만든다(매번 독립 인스턴스)."""
        return HmmDetector(
            extractor=extractor_by_name(self.extractor),
            n_states=self.n_states,
            window=self.window,
            ensemble_size=self.ensemble_size,
            max_fit_windows=self.max_fit_windows,
            target_fpr=self.target_fpr,
            min_train_sequences=self.min_train_sequences,
            n_iter=self.n_iter,
        )


# 전체 ablation 매트릭스 프리셋.
# byte_class 추출기를 고정한 2×2(윈도우 on/off × 앙상블 on/off)로 두 아이디어의 '순수 기여'를
# 분리하고, raw_byte 양 끝점(baseline·full)을 더해 추출기 효과까지 한 표에서 비교한다.
# AUCp 비교가 목적이라 target_fpr=0.05(저FPR 운영점), 윈도우 변형은 비용 제어로 상한을 둔다.
#
# 비용 주의: 비교(compare)가 분 단위로 끝나도록 프리셋은 가벼운 학습 비용을 쓴다.
# (n_iter=100 + max_fit_windows=50000 조합은 raw_byte+윈도우+앙상블에서 비현실적으로 느렸음.
#  더 정밀한 학습이 필요하면 ModelVariant를 직접 만들어 n_iter/상한을 올리면 된다.)
_W = 6  # 슬라이딩 윈도우 폭
_K = 3  # 앙상블 크기
_CAP = 5_000  # 윈도우 학습 상한(비용 제어)
_ITER = 30  # Baum–Welch 반복(인터랙티브 속도 우선)

ABLATION_MATRIX: tuple[ModelVariant, ...] = (
    # ── byte_class 2×2 (아이디어 기여 분리) ──
    ModelVariant("bc_baseline", extractor="byte_class", n_iter=_ITER),
    ModelVariant("bc_window", extractor="byte_class", window=_W, max_fit_windows=_CAP, n_iter=_ITER),
    ModelVariant("bc_ensemble", extractor="byte_class", ensemble_size=_K, n_iter=_ITER),
    ModelVariant("bc_hmmpayl", extractor="byte_class", window=_W, ensemble_size=_K, max_fit_windows=_CAP, n_iter=_ITER),
    # ── raw_byte 양 끝점 (추출기 효과 비교) ──
    ModelVariant("raw_baseline", extractor="raw_byte", n_iter=_ITER),
    ModelVariant("raw_hmmpayl", extractor="raw_byte", window=_W, ensemble_size=_K, max_fit_windows=_CAP, n_iter=_ITER),
)
