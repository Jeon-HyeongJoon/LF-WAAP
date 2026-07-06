"""codecs — 도메인 객체 ↔ 파이프 바이트(JSON) 변환.

파이프에는 바이트만 흐르므로, Flow/HttpRequest를 토픽에 넣고 꺼낼 직렬화가 필요하다.
payload는 임의 바이트라 base64로 싣는다(JSON 안전).
"""

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.flow import Direction, Flow, PacketMeta
from waf.domain.model.http_request import HttpRequest
from waf.domain.model.verdict import Verdict
from waf.infrastructure.streaming.codecs import (
    decode_flow,
    decode_labeled_flow,
    decode_request,
    decode_verdict,
    encode_flow,
    encode_labeled_flow,
    encode_request,
    encode_verdict,
)


def test_flow_roundtrip_preserves_all_fields() -> None:
    flow = Flow(
        "HTTP", 8080, Direction.INBOUND,
        payload=b"GET /\x00\xff",
        packets=(PacketMeta(Direction.INBOUND, 5, 1.5, "PA"),),
    )

    restored = decode_flow(encode_flow(flow))

    assert restored == flow


def test_encode_returns_bytes() -> None:
    flow = Flow("http", 80, Direction.INBOUND, b"abc")
    assert isinstance(encode_flow(flow), bytes)


def test_labeled_flow_roundtrip_preserves_flow_and_label() -> None:
    flow = Flow("http", 80, Direction.INBOUND, b"' OR 1=1--\xff")

    restored_flow, is_attack = decode_labeled_flow(encode_labeled_flow(flow, is_attack=True))

    assert restored_flow == flow
    assert is_attack is True


def test_labeled_flow_carries_normal_label() -> None:
    flow = Flow("http", 80, Direction.INBOUND, b"GET /home")

    _, is_attack = decode_labeled_flow(encode_labeled_flow(flow, is_attack=False))

    assert is_attack is False


def test_http_request_roundtrip_preserves_fields() -> None:
    request = HttpRequest(
        method="post", path="/login", query="next=/home",
        headers={"Host": "shop.example.com", "Content-Type": "application/json"},
        body="user=admin", client_ip="1.2.3.4",
    )

    restored = decode_request(encode_request(request))

    assert restored == request  # __post_init__ 정규화(대문자 메서드·소문자 헤더)까지 일치


def test_verdict_message_carries_decision_and_reason() -> None:
    verdict = Verdict.block(
        (DetectionSignal("ruleset", blocked=True, reason="SQLi pattern", score=1.0),)
    )

    message = decode_verdict(encode_verdict(verdict))

    assert message.blocked is True
    assert message.decision == "BLOCK"
    assert "SQLi" in message.reason
