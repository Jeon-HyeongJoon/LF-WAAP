"""CalibratableDetector — a Detector that can be trained/calibrated on traffic.

This is the abstraction CalibrateModel depends on, so the use case never imports
a concrete model (HmmDetector / hmmlearn). The detection technique stays in
infrastructure; the domain only knows "something that trains and inspects".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.flow import Flow
from waf.domain.model.http_request import HttpRequest


@runtime_checkable
class CalibratableDetector(Protocol):
    @property
    def name(self) -> str: ...

    def inspect(self, request: HttpRequest) -> DetectionSignal: ...

    def train(
        self,
        normal_flows: Sequence[Flow],
        validation_flows: Sequence[Flow],
    ) -> None: ...

    @property
    def is_trained(self) -> bool: ...
