"""Detector — the port (interface) every blocking strategy implements.

Both the rule-set engine and the HMM model are Detectors. The domain depends
only on this abstraction; concrete implementations live in `infrastructure`.
This is what lets the WAF grow new strategies without touching decision logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.http_request import HttpRequest


@runtime_checkable
class Detector(Protocol):
    @property
    def name(self) -> str: ...

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        """Examine a request and return this detector's signal."""
        ...
