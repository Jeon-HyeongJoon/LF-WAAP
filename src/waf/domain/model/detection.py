"""DetectionSignal — the output of a single Detector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalAction(str, Enum):
    ALLOW = "ALLOW"
    ALERT = "ALERT"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class DetectionSignal:
    """One detector's opinion about a request.

    Attributes:
        detector: name of the detector that produced this signal.
        blocked: True if this detector considers the request malicious.
        reason: human-readable explanation (matched rule, anomaly score, ...).
        score: detector-specific confidence/anomaly score. Higher = more suspicious.
                Rule-set detectors typically emit 0.0/1.0; HMM emits a continuous score.
    """

    detector: str
    blocked: bool
    reason: str
    score: float = 0.0
    action: SignalAction = SignalAction.ALLOW

    def __post_init__(self) -> None:
        if self.blocked and self.action is SignalAction.ALLOW:
            object.__setattr__(self, "action", SignalAction.BLOCK)
