"""Verdict — the final block/allow decision for a request."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from waf.domain.model.detection import DetectionSignal, SignalAction


class Decision(str, Enum):
    ALLOW = "ALLOW"
    ALERT = "ALERT"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class Verdict:
    decision: Decision
    signals: tuple[DetectionSignal, ...]

    @property
    def is_blocked(self) -> bool:
        return self.decision is Decision.BLOCK

    @property
    def blocking_signals(self) -> tuple[DetectionSignal, ...]:
        return tuple(s for s in self.signals if s.blocked)

    @property
    def actionable_signals(self) -> tuple[DetectionSignal, ...]:
        return tuple(s for s in self.signals if s.action is not SignalAction.ALLOW)

    @property
    def reason(self) -> str:
        if self.decision is Decision.ALLOW:
            return "allowed"
        return "; ".join(f"[{s.detector}] {s.reason}" for s in self.actionable_signals)

    @classmethod
    def allow(cls, signals: tuple[DetectionSignal, ...]) -> "Verdict":
        return cls(Decision.ALLOW, signals)

    @classmethod
    def alert(cls, signals: tuple[DetectionSignal, ...]) -> "Verdict":
        return cls(Decision.ALERT, signals)

    @classmethod
    def challenge(cls, signals: tuple[DetectionSignal, ...]) -> "Verdict":
        return cls(Decision.CHALLENGE, signals)

    @classmethod
    def block(cls, signals: tuple[DetectionSignal, ...]) -> "Verdict":
        return cls(Decision.BLOCK, signals)
