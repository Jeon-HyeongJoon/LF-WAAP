from waf.domain.model.detection import DetectionSignal, SignalAction
from waf.domain.model.http_request import HttpRequest
from waf.domain.model.verdict import Decision
from waf.domain.service.block_decision_service import BlockDecisionService


class _StaticDetector:
    def __init__(self, signal: DetectionSignal) -> None:
        self._signal = signal

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        return self._signal


def test_block_decision_service_returns_alert_verdict_for_alert_signal() -> None:
    decider = BlockDecisionService(
        (
            _StaticDetector(
                DetectionSignal(
                    "workflow",
                    blocked=False,
                    reason="transition anomaly",
                    score=2.0,
                    action=SignalAction.ALERT,
                )
            ),
        )
    )

    verdict = decider.decide(HttpRequest("GET", "/"))

    assert verdict.decision is Decision.ALERT
    assert not verdict.is_blocked
    assert "transition anomaly" in verdict.reason


def test_block_decision_service_prioritizes_challenge_over_alert() -> None:
    decider = BlockDecisionService(
        (
            _StaticDetector(
                DetectionSignal("workflow", False, "low probability", action=SignalAction.ALERT)
            ),
            _StaticDetector(
                DetectionSignal("context", False, "sensitive flow", action=SignalAction.CHALLENGE)
            ),
        )
    )

    verdict = decider.decide(HttpRequest("GET", "/"))

    assert verdict.decision is Decision.CHALLENGE
    assert "sensitive flow" in verdict.reason
