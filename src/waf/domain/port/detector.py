"""Detector — the port every blocking strategy implements.

Both the rule-set engine and anomaly models are Detectors. The domain depends
only on this abstraction; concrete implementations live in infrastructure.
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
