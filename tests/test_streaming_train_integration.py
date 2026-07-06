"""파이프 학습 통합 — 토픽에 정상/검증 Flow를 넣으면 기존 CalibrateModel이 그대로 학습한다.

핵심 검증: 학습 코드(CalibrateModel)는 한 줄도 안 바뀌고, TrafficSource 어댑터만
StreamingTrafficSource로 갈아끼웠을 뿐인데 '파이프 → 학습 → 모델'이 성립한다.
"""

import pytest

pytest.importorskip("hmmlearn")

from waf.application.calibrate_model import CalibrateModel  # noqa: E402
from waf.infrastructure.adapter.persistence import PickleModelRepository  # noqa: E402
from waf.infrastructure.hmm import HmmDetector  # noqa: E402
from waf.infrastructure.streaming.codecs import encode_flow  # noqa: E402
from waf.infrastructure.streaming.memory import InMemoryChannel  # noqa: E402
from waf.infrastructure.streaming.traffic_source import StreamingTrafficSource  # noqa: E402

from synthetic import batch, normal_flow  # noqa: E402


def test_training_consumes_pipe_and_produces_trained_model(tmp_path) -> None:
    channel = InMemoryChannel()
    for flow in batch(normal_flow, 60, seed=1):
        channel.publish("waf.train.normal", encode_flow(flow))
    for flow in batch(normal_flow, 30, seed=2):
        channel.publish("waf.train.validation", encode_flow(flow))

    source = StreamingTrafficSource(channel)
    repository = PickleModelRepository(tmp_path / "model.pkl")
    detector = HmmDetector(n_states=4, target_fpr=0.1, min_train_sequences=5, n_iter=30)

    report = CalibrateModel(source, repository)(detector)

    assert report.normal_count == 60
    assert report.validation_count == 30
    assert detector.is_trained
    assert (tmp_path / "model.pkl").exists()
