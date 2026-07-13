"""BlockDecisionService — the domain service that decides ALLOW vs BLOCK.

It orchestrates an ordered set of detectors and combines their signals under a
configurable policy. This is the "차단할지 말지 결정하는 함수" from the plan,
expressed as a first-class domain service so it stays testable and replaceable.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto

from waf.domain.model.detection import DetectionSignal, SignalAction
from waf.domain.model.http_request import HttpRequest
from waf.domain.model.verdict import Verdict
from waf.domain.port.detector import Detector


class BlockPolicy(Enum):
    """How detector signals combine into a final decision."""

    ANY = auto()  # block if any detector blocks (default WAF semantics)
    ALL = auto()  # block only if every detector blocks (high precision)


class BlockDecisionService:
    def __init__(
        self,
        detectors: Sequence[Detector],
        policy: BlockPolicy = BlockPolicy.ANY,
    ) -> None:
        if not detectors:
            raise ValueError("BlockDecisionService requires at least one detector")
        self._detectors = tuple(detectors)
        self._policy = policy

    def decide(self, request: HttpRequest) -> Verdict:
        signals = tuple(d.inspect(request) for d in self._detectors)
        blocking = tuple(s for s in signals if s.blocked)

        if self._policy is BlockPolicy.ANY:
            should_block = len(blocking) > 0
        else:  # BlockPolicy.ALL
            should_block = len(blocking) == len(signals)

        if should_block:
            return Verdict.block(signals)
        return _non_blocking_verdict(signals)


def _non_blocking_verdict(signals: tuple[DetectionSignal, ...]) -> Verdict:
    if any(signal.action is SignalAction.CHALLENGE for signal in signals):
        return Verdict.challenge(signals)
    if any(signal.action is SignalAction.ALERT for signal in signals):
        return Verdict.alert(signals)
    return Verdict.allow(signals)
