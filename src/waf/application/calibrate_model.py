"""CalibrateModel — use case that calibrates a detector from a traffic source.

Pure orchestration: pull normal + validation traffic from the source, hand it to
the detector to train/calibrate, persist the result. No detection math, no I/O
details — those live in the domain model and infrastructure adapters respectively.
"""

from __future__ import annotations

from dataclasses import dataclass

from waf.domain.port.calibratable_detector import CalibratableDetector
from waf.domain.port.model_repository import ModelRepository
from waf.domain.port.traffic_source import TrafficSource


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    normal_count: int
    validation_count: int


class CalibrateModel:
    def __init__(self, source: TrafficSource, repository: ModelRepository) -> None:
        self._source = source
        self._repository = repository

    def __call__(self, detector: CalibratableDetector) -> CalibrationReport:
        normal = list(self._source.normal())
        validation = list(self._source.validation())
        detector.train(normal, validation)
        self._repository.save(detector)
        return CalibrationReport(normal_count=len(normal), validation_count=len(validation))
