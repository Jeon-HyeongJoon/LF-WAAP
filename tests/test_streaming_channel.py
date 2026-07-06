"""MessageChannel — Kafka/인메모리를 갈아끼우기 위한 메시징 좌석(seam).

상위 계층(학습/평가/추론)은 이 포트에만 의존하므로, 테스트는 브로커 없이 InMemoryChannel
페이크로 빠르고 결정적으로 돌고, 실제 실행만 Kafka 어댑터로 바뀐다.
"""

from waf.infrastructure.streaming.memory import InMemoryChannel


def test_publish_then_poll_returns_messages_in_order() -> None:
    channel = InMemoryChannel()
    channel.publish("waf.train.normal", b"a")
    channel.publish("waf.train.normal", b"b")

    assert channel.poll("waf.train.normal") == [b"a", b"b"]


def test_topics_are_isolated() -> None:
    channel = InMemoryChannel()
    channel.publish("t1", b"a")
    channel.publish("t2", b"b")

    assert channel.poll("t1") == [b"a"]
    assert channel.poll("t2") == [b"b"]


def test_poll_unknown_topic_is_empty() -> None:
    assert InMemoryChannel().poll("nope") == []


def test_stream_yields_messages_in_order() -> None:
    channel = InMemoryChannel()
    channel.publish("waf.inspect.requests", b"req1")
    channel.publish("waf.inspect.requests", b"req2")

    assert list(channel.stream("waf.inspect.requests")) == [b"req1", b"req2"]
