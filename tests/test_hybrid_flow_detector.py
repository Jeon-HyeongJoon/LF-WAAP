import pytest

pytest.importorskip("hmmlearn")

from waf.domain.model.http_request import HttpRequest  # noqa: E402
from waf.domain.model.flow import Flow  # noqa: E402
from waf.infrastructure.behavior import HybridFlowDetector  # noqa: E402
from waf.infrastructure.behavior.markov_flow import MarkovFlowModel  # noqa: E402
from waf.infrastructure.hmm import HmmDetector  # noqa: E402
from waf.infrastructure.hmm.http_flow import http_request_to_flow  # noqa: E402


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
