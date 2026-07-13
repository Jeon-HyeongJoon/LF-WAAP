import pytest

pytest.importorskip("hmmlearn")

from waf.domain.model.flow import Direction  # noqa: E402
from waf.domain.model.flow import Flow  # noqa: E402
from waf.domain.model.http_request import HttpRequest  # noqa: E402
from waf.infrastructure.flow.http_flow import http_request_to_flow  # noqa: E402
from waf.infrastructure.flow.partition import PartitionKey  # noqa: E402
from waf.infrastructure.hmm import HmmDetector  # noqa: E402
from waf.infrastructure.hmm.partitioned_hmm import Assessment  # noqa: E402
from waf.infrastructure.hybrid import HybridFlowDetector  # noqa: E402
from waf.infrastructure.markov.flow_model import (  # noqa: E402
    START_STATE,
    MarkovAssessment,
    MarkovFlowModel,
)


def _flow(method: str, path: str, query: str = "") -> Flow:
    return http_request_to_flow(
        HttpRequest(method, path, query=query, headers={"Host": "shop.example.com"})
    )


def _normal_sequence(i: int) -> list[Flow]:
    return [
        _flow("GET", "/"),
        _flow("GET", "/login"),
        _flow("POST", "/login"),
        _flow("GET", f"/products/{i}", "ref=home"),
        _flow("POST", "/cart"),
        _flow("POST", "/checkout"),
    ]


def _hmm_assessment(*, blocked: bool = False) -> Assessment:
    return Assessment(
        blocked,
        -2.0 if blocked else -0.5,
        PartitionKey("http", 80, Direction.INBOUND, "65-256"),
        "anomaly: score -2.000 < threshold -1.000" if blocked else "normal",
    )


class _StaticMarkov:
    is_trained = True

    def __init__(self, assessment: MarkovAssessment, *, threshold: float = -1.0) -> None:
        self._assessment = assessment
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def fit_from_flows(
        self,
        normal_sequences: object,
        validation_sequences: object,
        extractor: object,
    ) -> None:
        return None

    def assess_transition(self, previous_state: str, current_state: str) -> MarkovAssessment:
        return self._assessment


class _StaticHmm:
    is_trained = True

    def __init__(self, assessment: Assessment) -> None:
        self._assessment = assessment

    def train(self, normal_flows: object, validation_flows: object) -> None:
        return None

    def assess_flow(self, flow: Flow) -> Assessment:
        return self._assessment


def test_hybrid_detector_allows_weak_isolated_markov_violation() -> None:
    detector = HybridFlowDetector(
        markov_model=_StaticMarkov(
            MarkovAssessment(
                True,
                -1.5,
                START_STATE,
                "GET /rare",
                "markov anomaly: weak isolated transition",
            )
        ),
        hmm_detector=_StaticHmm(_hmm_assessment()),
    )

    result = detector.assess_flow(_flow("GET", "/rare"), session_id="u1")

    assert not result.blocked
    assert result.markov.blocked
    assert not result.hmm.blocked
    assert result.score == pytest.approx(0.5)
    assert "hybrid allowed: isolated markov anomaly" in result.reason


def test_hybrid_detector_blocks_strong_isolated_markov_violation() -> None:
    detector = HybridFlowDetector(
        markov_model=_StaticMarkov(
            MarkovAssessment(
                True,
                -2.25,
                START_STATE,
                "POST /admin/delete",
                "markov anomaly: severe transition",
            )
        ),
        hmm_detector=_StaticHmm(_hmm_assessment()),
    )

    result = detector.assess_flow(_flow("POST", "/admin/delete"), session_id="u1")

    assert result.blocked
    assert result.markov.blocked
    assert not result.hmm.blocked


def test_hybrid_detector_blocks_hmm_anomaly_without_markov_confirmation() -> None:
    detector = HybridFlowDetector(
        markov_model=_StaticMarkov(
            MarkovAssessment(
                False,
                -0.25,
                START_STATE,
                "GET /",
                "markov normal: common transition",
            )
        ),
        hmm_detector=_StaticHmm(_hmm_assessment(blocked=True)),
    )

    result = detector.assess_flow(_flow("GET", "/"), session_id="u1")

    assert result.blocked
    assert not result.markov.blocked
    assert result.hmm.blocked


def test_hybrid_detector_uses_markov_flow_violation_signal() -> None:
    detector = HybridFlowDetector(
        markov_model=MarkovFlowModel(target_fpr=0.1),
        hmm_detector=HmmDetector(n_states=3, min_train_sequences=3, n_iter=10, target_fpr=0.1),
    )
    detector.train_sequences(
        [_normal_sequence(i) for i in range(12)],
        [_normal_sequence(i) for i in range(12, 16)],
    )

    assert not detector.assess_flow(_flow("GET", "/"), session_id="u1").blocked
    result = detector.assess_flow(_flow("POST", "/admin/delete"), session_id="u1")

    assert result.blocked
    assert result.markov.blocked
    assert "markov anomaly" in result.reason
