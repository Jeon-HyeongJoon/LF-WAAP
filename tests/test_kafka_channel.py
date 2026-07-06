"""KafkaChannel — 실제 Kafka 어댑터의 인터페이스 계약.

실제 브로커 라운드트립은 docker-compose가 필요하므로(수동 통합 테스트), 여기서는 어댑터가
MessageChannel 표면(publish/poll/stream)을 갖추는지만 확인한다. kafka-python 미설치 시 skip.
"""

import pytest

pytest.importorskip("kafka")

from waf.infrastructure.streaming.kafka_channel import KafkaChannel  # noqa: E402


def test_kafka_channel_exposes_message_channel_surface() -> None:
    for method in ("publish", "poll", "stream"):
        assert callable(getattr(KafkaChannel, method))
