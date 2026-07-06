"""TrafficSource — port supplying training/validation traffic to the use case.

The source decides how normal vs validation samples are partitioned (e.g. by
separate files/directories); the use case stays agnostic to where data lives.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from waf.domain.model.flow import Flow


@runtime_checkable
class TrafficSource(Protocol):
    def normal(self) -> Iterable[Flow]:
        """Benign flows used to fit the model."""
        ...

    def validation(self) -> Iterable[Flow]:
        """Held-out benign flows used to calibrate the decision threshold."""
        ...
