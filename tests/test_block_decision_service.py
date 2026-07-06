from waf.application import InspectRequest
from waf.domain.model.detection import DetectionSignal
from waf.domain.model.http_request import HttpRequest
from waf.domain.service import BlockDecisionService, BlockPolicy


class StubDetector:
    def __init__(self, name: str, blocked: bool) -> None:
        self._name = name
        self._blocked = blocked

    @property
    def name(self) -> str:
        return self._name

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        return DetectionSignal(self._name, self._blocked, reason="stub")


REQ = HttpRequest(method="GET", path="/")


def test_any_policy_blocks_when_one_detector_blocks() -> None:
    service = BlockDecisionService(
        [StubDetector("a", False), StubDetector("b", True)],
        policy=BlockPolicy.ANY,
    )
    verdict = service.decide(REQ)
    assert verdict.is_blocked
    assert verdict.blocking_signals[0].detector == "b"


def test_any_policy_allows_when_none_block() -> None:
    service = BlockDecisionService([StubDetector("a", False), StubDetector("b", False)])
    assert not service.decide(REQ).is_blocked


def test_all_policy_blocks_only_when_every_detector_blocks() -> None:
    mixed = BlockDecisionService(
        [StubDetector("a", True), StubDetector("b", False)],
        policy=BlockPolicy.ALL,
    )
    assert not mixed.decide(REQ).is_blocked

    unanimous = BlockDecisionService(
        [StubDetector("a", True), StubDetector("b", True)],
        policy=BlockPolicy.ALL,
    )
    assert unanimous.decide(REQ).is_blocked


def test_use_case_delegates_to_service() -> None:
    service = BlockDecisionService([StubDetector("a", True)])
    inspect = InspectRequest(service)
    assert inspect(REQ).is_blocked
