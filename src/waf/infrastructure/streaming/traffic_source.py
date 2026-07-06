"""StreamingTrafficSource — 파이프 토픽 → 학습/검증 Flow 스트림.

도메인 TrafficSource 포트(normal/validation)를 메시지 채널로 구현한다. CalibrateModel은
이 포트에만 의존하므로 Kafka/인메모리 어느 쪽이든 학습 코드 변경 없이 동작한다.
"""

from __future__ import annotations

from collections.abc import Iterator

from waf.domain.model.flow import Flow
from waf.infrastructure.streaming.channel import MessageChannel
from waf.infrastructure.streaming.codecs import decode_flow

# 학습/검증 도메인을 토픽으로 분리한다(각 도메인 = 각 토픽).
DEFAULT_NORMAL_TOPIC = "waf.train.normal"
DEFAULT_VALIDATION_TOPIC = "waf.train.validation"


class StreamingTrafficSource:
    def __init__(
        self,
        channel: MessageChannel,
        normal_topic: str = DEFAULT_NORMAL_TOPIC,
        validation_topic: str = DEFAULT_VALIDATION_TOPIC,
    ) -> None:
        self._channel = channel
        self._normal_topic = normal_topic
        self._validation_topic = validation_topic

    def normal(self) -> Iterator[Flow]:
        """정상 토픽에 쌓인 메시지를 Flow로 디코딩해 흘려보낸다."""
        for message in self._channel.poll(self._normal_topic):
            yield decode_flow(message)

    def validation(self) -> Iterator[Flow]:
        """검증(임계치 보정) 토픽의 메시지를 Flow로 디코딩해 흘려보낸다."""
        for message in self._channel.poll(self._validation_topic):
            yield decode_flow(message)
