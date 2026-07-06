"""CalibrateModel use case — orchestrates: load traffic -> train -> persist.

Tested through stub ports only (no file I/O, no real HMM), so these tests pin the
use case's behavior, not any implementation detail of the source/repo/detector.
"""

from waf.application.calibrate_model import CalibrateModel
from waf.domain.model.detection import DetectionSignal
from waf.domain.model.flow import Direction, Flow
from waf.domain.model.http_request import HttpRequest


def _flow(payload: bytes) -> Flow:
    return Flow("http", 80, Direction.INBOUND, payload)


class StubSource:
    def __init__(self, normal: list[Flow], validation: list[Flow]) -> None:
        self._normal = normal
        self._validation = validation

    def normal(self):
        return iter(self._normal)

    def validation(self):
        return iter(self._validation)


class StubRepository:
    def __init__(self) -> None:
        self.saved = None

    def save(self, detector) -> None:
        self.saved = detector

    def load(self):
        return self.saved


class StubDetector:
    name = "stub"

    def __init__(self) -> None:
        self.trained_with: tuple[list[Flow], list[Flow]] | None = None
        self._trained = False

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        return DetectionSignal(self.name, blocked=False, reason="stub")

    def train(self, normal_flows, validation_flows) -> None:
        self.trained_with = (list(normal_flows), list(validation_flows))
        self._trained = True

    @property
    def is_trained(self) -> bool:
        return self._trained


def test_calibrate_trains_on_source_data_then_persists() -> None:
    normal = [_flow(b"aaa"), _flow(b"bbb")]
    validation = [_flow(b"ccc")]
    source = StubSource(normal, validation)
    repo = StubRepository()
    detector = StubDetector()

    report = CalibrateModel(source, repo)(detector)

    # trained on exactly the source's normal + validation flows
    assert detector.trained_with == (normal, validation)
    # the trained detector was persisted
    assert repo.saved is detector
    assert detector.is_trained
    # observable summary
    assert report.normal_count == 2
    assert report.validation_count == 1
