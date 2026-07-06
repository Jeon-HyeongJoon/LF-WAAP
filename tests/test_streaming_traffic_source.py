"""StreamingTrafficSource — 파이프 토픽에서 학습/검증 Flow를 읽는 TrafficSource 어댑터.

기존 CalibrateModel 유스케이스는 TrafficSource 포트에만 의존하므로, 이 어댑터를 끼우면
'토픽에 데이터를 넣으면 학습이 된다'가 학습 코드 수정 없이 성립한다.
"""

from waf.domain.model.flow import Direction, Flow
from waf.infrastructure.streaming.codecs import encode_flow
from waf.infrastructure.streaming.memory import InMemoryChannel
from waf.infrastructure.streaming.traffic_source import StreamingTrafficSource


def test_reads_normal_and_validation_from_their_topics() -> None:
    channel = InMemoryChannel()
    n1 = Flow("http", 80, Direction.INBOUND, b"a")
    n2 = Flow("http", 80, Direction.INBOUND, b"b")
    v1 = Flow("http", 80, Direction.INBOUND, b"c")
    channel.publish("waf.train.normal", encode_flow(n1))
    channel.publish("waf.train.normal", encode_flow(n2))
    channel.publish("waf.train.validation", encode_flow(v1))

    source = StreamingTrafficSource(channel)

    assert list(source.normal()) == [n1, n2]
    assert list(source.validation()) == [v1]
