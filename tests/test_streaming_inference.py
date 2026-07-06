"""StreamingInspector — 실시간 추론: 요청 토픽을 소비해 판정 토픽으로 결과를 낸다.

WAF 진입점 BlockDecisionService.decide를 그대로 쓰므로 룰셋·HMM 어느 탐지기든 동일하게
동작한다(여기선 브로커·hmmlearn 없이 룰셋만으로 검증).
"""

from waf.domain.model.http_request import HttpRequest
from waf.domain.service.block_decision_service import BlockDecisionService
from waf.infrastructure.ruleset.ruleset_detector import RuleSetDetector
from waf.infrastructure.streaming.codecs import decode_verdict, encode_request
from waf.infrastructure.streaming.inference import StreamingInspector
from waf.infrastructure.streaming.memory import InMemoryChannel


def test_inspector_consumes_requests_and_emits_verdicts() -> None:
    channel = InMemoryChannel()
    service = BlockDecisionService([RuleSetDetector.default()])
    channel.publish(
        "waf.inspect.requests",
        encode_request(HttpRequest("GET", "/products", "id=42&sort=price")),
    )
    channel.publish(
        "waf.inspect.requests",
        encode_request(HttpRequest("GET", "/items", "id=1 OR 1=1")),
    )

    processed = StreamingInspector(channel, service).run()

    assert processed == 2
    verdicts = [decode_verdict(m) for m in channel.poll("waf.inspect.verdicts")]
    assert [v.blocked for v in verdicts] == [False, True]  # benign, then SQLi
    assert "SQLI" in verdicts[1].reason
