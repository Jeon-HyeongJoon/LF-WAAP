"""ModelRepository — port for persisting/retrieving a trained detector.

Typed against the domain `Detector` abstraction so persistence (pickle, blob
store, ...) stays swappable without leaking infrastructure types into the use case.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from waf.domain.detector.detector import Detector


@runtime_checkable
class ModelRepository(Protocol):
    def save(self, detector: Detector) -> None: ...

    def load(self) -> Detector: ...
