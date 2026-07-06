"""실시간 추론 파이프 — 요청 토픽을 소비해 WAF 판정을 판정 토픽으로 발행한다.

WAF 진입점인 BlockDecisionService(룰셋+HMM 조합)를 그대로 호출하므로, 학습한 HMM 모델이
실제 차단 결정에 어떻게 쓰이는지 한눈에 보인다: 요청 in → decide() → verdict out.
"""

from __future__ import annotations

from waf.domain.service.block_decision_service import BlockDecisionService
from waf.infrastructure.streaming.channel import MessageChannel
from waf.infrastructure.streaming.codecs import decode_request, encode_verdict

DEFAULT_REQUESTS_TOPIC = "waf.inspect.requests"
DEFAULT_VERDICTS_TOPIC = "waf.inspect.verdicts"


class StreamingInspector:
    def __init__(
        self,
        channel: MessageChannel,
        decider: BlockDecisionService,
        requests_topic: str = DEFAULT_REQUESTS_TOPIC,
        verdicts_topic: str = DEFAULT_VERDICTS_TOPIC,
    ) -> None:
        self._channel = channel
        self._decider = decider
        self._requests_topic = requests_topic
        self._verdicts_topic = verdicts_topic

    def run(self) -> int:
        """요청 토픽을 흘려보내며 건건이 판정→발행한다. 처리한 요청 수를 돌려준다.

        인메모리 채널에선 현재 쌓인 요청을 모두 처리하고 끝나고, Kafka 채널에선 같은
        코드가 신규 요청을 계속 소비한다(채널의 stream 구현 차이만으로 배치/실시간 전환).
        """
        processed = 0
        for message in self._channel.stream(self._requests_topic):
            request = decode_request(message)
            verdict = self._decider.decide(request)
            self._channel.publish(self._verdicts_topic, encode_verdict(verdict))
            processed += 1
        return processed
