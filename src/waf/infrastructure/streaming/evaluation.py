"""파이프 기반 모델 테스트(평가) — 평가 토픽에서 라벨된 Flow를 읽는다.

'모델 테스트용 데이터를 파이프에 추가'의 소비측. 학습 소스(StreamingTrafficSource)와
대칭 구조이며, 읽어들인 (Flow, is_attack)을 학습된 탐지기로 채점해 지표를 낸다.
"""

from __future__ import annotations

from collections.abc import Iterator

from waf.domain.model.flow import Flow
from waf.infrastructure.hmm.benchmark import ScoredMetrics, score_labeled_flows
from waf.infrastructure.hmm.hmm_detector import HmmDetector
from waf.infrastructure.streaming.channel import MessageChannel
from waf.infrastructure.streaming.codecs import decode_labeled_flow

DEFAULT_EVAL_TOPIC = "waf.eval.labeled"


class StreamingLabeledSource:
    def __init__(self, channel: MessageChannel, topic: str = DEFAULT_EVAL_TOPIC) -> None:
        self._channel = channel
        self._topic = topic

    def labeled(self) -> Iterator[tuple[Flow, bool]]:
        """평가 토픽의 메시지를 (Flow, is_attack) 쌍으로 디코딩해 흘려보낸다."""
        for message in self._channel.poll(self._topic):
            yield decode_labeled_flow(message)


def evaluate_stream(
    detector: HmmDetector,
    source: StreamingLabeledSource,
    target_fpr: float = 0.05,
) -> ScoredMetrics:
    """파이프 평가 토픽을 비운 뒤 학습된 탐지기로 채점한다.

    CLI/CSIC 평가와 동일한 score_labeled_flows 핵심을 재사용하므로 결과 의미가 일관된다.
    """
    return score_labeled_flows(detector, list(source.labeled()), target_fpr)
