"""InMemoryChannel — 브로커 없는 MessageChannel 페이크.

토픽별 리스트에 메시지를 쌓아 두는 가장 단순한 구현. 테스트와 로컬 실험에서 Kafka 없이
파이프라인 전체를 돌려 보기 위한 것이며, 동작 계약은 KafkaChannel과 동일하다.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator


class InMemoryChannel:
    def __init__(self) -> None:
        self._topics: dict[str, list[bytes]] = defaultdict(list)

    def publish(self, topic: str, message: bytes) -> None:
        self._topics[topic].append(message)

    def poll(self, topic: str) -> list[bytes]:
        # 발행 순서를 보존한 사본을 돌려준다(호출자가 리스트를 건드려도 안전).
        return list(self._topics.get(topic, []))

    def stream(self, topic: str) -> Iterator[bytes]:
        # 인메모리에선 현재 쌓인 메시지를 순서대로 흘려보낸다(테스트에서 유한 종료).
        # 실제 Kafka 어댑터는 같은 시그니처로 신규 메시지를 계속 소비한다.
        yield from list(self._topics.get(topic, []))
