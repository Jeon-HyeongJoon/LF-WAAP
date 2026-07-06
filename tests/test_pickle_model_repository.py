"""PickleModelRepository — persists/loads a trained Detector via pickle.

Tested through the ModelRepository port contract. A module-level FakeDetector keeps
the core tests fast and dependency-free; one test exercises a real HmmDetector.
"""

from dataclasses import dataclass

import pytest

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.http_request import HttpRequest
from waf.domain.port.model_repository import ModelRepository
from waf.infrastructure.adapter.persistence import PickleModelRepository


@dataclass
class FakeDetector:
    name: str = "fake"
    trained: bool = True
    tag: str = "v1"

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        return DetectionSignal(self.name, blocked=True, reason=self.tag)

    @property
    def is_trained(self) -> bool:
        return self.trained


def test_save_then_load_round_trips_detector(tmp_path) -> None:
    repo = PickleModelRepository(tmp_path / "model.pkl")
    repo.save(FakeDetector(tag="trained-model"))

    loaded = repo.load()

    assert loaded.is_trained
    assert loaded.inspect(HttpRequest("GET", "/")).reason == "trained-model"


def test_load_before_save_raises(tmp_path) -> None:
    repo = PickleModelRepository(tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError):
        repo.load()


def test_save_creates_missing_parent_directories(tmp_path) -> None:
    repo = PickleModelRepository(tmp_path / "nested" / "dir" / "model.pkl")
    repo.save(FakeDetector())
    assert repo.load().is_trained


def test_satisfies_model_repository_port(tmp_path) -> None:
    assert isinstance(PickleModelRepository(tmp_path / "m.pkl"), ModelRepository)


def test_real_hmm_detector_round_trip_preserves_verdict(tmp_path) -> None:
    pytest.importorskip("hmmlearn")
    import random

    from synthetic import attack_flow, batch, normal_flow

    from waf.infrastructure.hmm import HmmDetector

    detector = HmmDetector(n_states=4, target_fpr=0.1, n_iter=40)
    detector.train(batch(normal_flow, 60, seed=1), batch(normal_flow, 30, seed=2))

    repo = PickleModelRepository(tmp_path / "hmm.pkl")
    repo.save(detector)
    loaded = repo.load()

    assert loaded.is_trained
    attack = attack_flow(random.Random(5))
    # loaded model produces the same decision as the original
    assert loaded.assess_flow(attack).blocked == detector.assess_flow(attack).blocked
