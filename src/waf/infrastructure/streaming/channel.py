"""MessageChannel — 메시징 포트(좌석).

publish/poll/stream 세 동작만 노출하는 얇은 추상화. Kafka든 인메모리든 이 인터페이스만
만족하면 상위 계층(학습/평가/추론)에 그대로 끼울 수 있다. poll은 '현재 토픽에 쌓인 메시지
전체'를 배치로 읽고(학습·평가용), stream은 메시지를 하나씩 흘려보낸다(실시간 추론용).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class MessageChannel(Protocol):
    def publish(self, topic: str, message: bytes) -> None:
        """토픽에 바이트 메시지 한 건을 발행한다."""
        ...

    def poll(self, topic: str) -> list[bytes]:
        """토픽에 현재 쌓인 메시지를 발행 순서대로 모두 돌려준다(배치 읽기)."""
        ...

    def stream(self, topic: str) -> Iterator[bytes]:
        """토픽 메시지를 발행 순서대로 하나씩 흘려보낸다(실시간 추론용)."""
        ...
