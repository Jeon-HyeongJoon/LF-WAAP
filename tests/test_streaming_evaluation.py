"""StreamingLabeledSource — 평가 토픽에서 (Flow, is_attack) 쌍을 읽는다.

'모델 테스트용 데이터를 파이프에 추가'를 구현하는 소비측. 학습 소스와 대칭이다.
"""

import pytest

from waf.domain.model.flow import Direction, Flow
from waf.infrastructure.streaming.codecs import encode_labeled_flow
from waf.infrastructure.streaming.evaluation import StreamingLabeledSource, evaluate_stream
from waf.infrastructure.streaming.memory import InMemoryChannel


def test_labeled_source_reads_flow_label_pairs_in_order() -> None:
    channel = InMemoryChannel()
    normal = Flow("http", 80, Direction.INBOUND, b"GET /home")
    attack = Flow("http", 80, Direction.INBOUND, b"' OR 1=1--")
    channel.publish("waf.eval.labeled", encode_labeled_flow(normal, is_attack=False))
    channel.publish("waf.eval.labeled", encode_labeled_flow(attack, is_attack=True))

    source = StreamingLabeledSource(channel)

    assert list(source.labeled()) == [(normal, False), (attack, True)]


def test_evaluate_stream_scores_a_trained_model() -> None:
    pytest.importorskip("hmmlearn")
    from waf.infrastructure.hmm import HmmDetector

    from synthetic import attack_flow, batch, normal_flow

    detector = HmmDetector(n_states=4, target_fpr=0.1, min_train_sequences=5, n_iter=30)
    detector.train(batch(normal_flow, 60, seed=1), batch(normal_flow, 30, seed=2))

    channel = InMemoryChannel()
    for flow in batch(normal_flow, 30, seed=3):
        channel.publish("waf.eval.labeled", encode_labeled_flow(flow, is_attack=False))
    for flow in batch(attack_flow, 30, seed=4):
        channel.publish("waf.eval.labeled", encode_labeled_flow(flow, is_attack=True))

    metrics = evaluate_stream(detector, StreamingLabeledSource(channel), target_fpr=0.1)

    assert metrics.report is not None
    assert 0.0 <= metrics.report.roc_auc <= 1.0
    assert metrics.classification is not None
    assert metrics.eval_normal == 30
    assert metrics.eval_attack == 30
