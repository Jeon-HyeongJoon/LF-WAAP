"""KafkaChannel — MessageChannel의 실제 Kafka 구현(kafka-python).

InMemoryChannel과 '동일한 계약(publish/poll/stream)'을 만족하므로, 상위 계층(학습/평가/추론)
코드는 그대로 두고 이 어댑터로만 바꾸면 실제 브로커 위에서 돈다.

설계 메모(POC 수준):
  * publish: 보낼 때마다 flush → 단순/동기적(처리량보다 이해하기 쉬움 우선).
  * poll: 'earliest부터 timeout까지' 읽어 토픽 현재 내용을 배치로 비운다(학습/평가용).
  * stream: 커밋하며 신규 메시지를 계속 소비한다(실시간 추론용).

kafka-python은 선택 의존성이라 import를 메서드 안으로 미룬다(미설치 환경에서도 모듈 자체는
임포트 가능). 설치: pip install -e ".[kafka]".
"""

from __future__ import annotations

from collections.abc import Iterator


class KafkaChannel:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        poll_timeout_ms: int = 1000,
        group_id: str = "waf-inference",
    ) -> None:
        from kafka import KafkaProducer

        self._bootstrap = bootstrap_servers
        self._poll_timeout_ms = poll_timeout_ms
        self._group_id = group_id
        self._producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

    def publish(self, topic: str, message: bytes) -> None:
        self._producer.send(topic, message)
        self._producer.flush()

    def poll(self, topic: str) -> list[bytes]:
        # group_id=None + earliest + consumer_timeout_ms: 토픽을 처음부터 끝까지 한 번 읽고
        # 더 올 메시지가 없으면(타임아웃) 종료 → 학습/평가용 '배치 드레인'.
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self._bootstrap,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            group_id=None,
            consumer_timeout_ms=self._poll_timeout_ms,
        )
        try:
            return [record.value for record in consumer]
        finally:
            consumer.close()

    def stream(self, topic: str) -> Iterator[bytes]:
        # 타임아웃 없이 신규 메시지를 계속 소비(실시간 추론). 오프셋은 그룹에 커밋된다.
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self._bootstrap,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id=self._group_id,
        )
        for record in consumer:
            yield record.value
