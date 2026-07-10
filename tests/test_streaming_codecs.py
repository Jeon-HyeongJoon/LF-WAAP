"""codecs — 도메인 객체 ↔ 파이프 바이트(JSON) 변환.

파이프에는 바이트만 흐르므로, Flow/HttpRequest를 토픽에 넣고 꺼낼 직렬화가 필요하다.
payload는 임의 바이트라 base64로 싣는다(JSON 안전).
"""

import json

import pytest

from waf.domain.model.detection import DetectionSignal, SignalAction
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


def test_labeled_flow_rejects_non_boolean_label() -> None:
    document = json.loads(
        encode_labeled_flow(
            Flow("http", 80, Direction.INBOUND, b"GET /home"),
            is_attack=False,
        )
    )
    document["is_attack"] = "false"

    with pytest.raises(ValueError, match="is_attack"):
        decode_labeled_flow(json.dumps(document).encode("utf-8"))


def test_decode_flow_rejects_non_object_message() -> None:
    with pytest.raises(ValueError, match="flow"):
        decode_flow(json.dumps(["not", "an", "object"]).encode("utf-8"))


def test_decode_labeled_flow_rejects_non_object_message() -> None:
    with pytest.raises(ValueError, match="labeled flow"):
        decode_labeled_flow(json.dumps(["not", "an", "object"]).encode("utf-8"))


def test_http_request_roundtrip_preserves_fields() -> None:
    request = HttpRequest(
        method="post", path="/login", query="next=/home",
        headers={"Host": "shop.example.com", "Content-Type": "application/json"},
        body="user=admin", client_ip="1.2.3.4",
    )

    restored = decode_request(encode_request(request))

    assert restored == request  # __post_init__ 정규화(대문자 메서드·소문자 헤더)까지 일치


@pytest.mark.parametrize("field", ["method", "path"])
def test_decode_request_rejects_null_required_request_fields(field: str) -> None:
    document = {
        "method": "GET",
        "path": "/products",
        "query": "",
        "headers": {},
        "body": "",
        "client_ip": "",
    }
    document[field] = None

    with pytest.raises(ValueError, match=field):
        decode_request(json.dumps(document).encode("utf-8"))


def test_verdict_message_carries_decision_and_reason() -> None:
    verdict = Verdict.block(
        (DetectionSignal("ruleset", blocked=True, reason="SQLi pattern", score=1.0),)
    )

    message = decode_verdict(encode_verdict(verdict))

    assert message.blocked is True
    assert message.decision == "BLOCK"
    assert "SQLi" in message.reason


def test_verdict_message_rejects_non_boolean_blocked_flag() -> None:
    document = json.loads(
        encode_verdict(
            Verdict.block(
                (
                    DetectionSignal(
                        "ruleset",
                        blocked=True,
                        reason="SQLi pattern",
                        score=1.0,
                    ),
                )
            )
        )
    )
    document["blocked"] = "false"

    with pytest.raises(ValueError, match="blocked"):
        decode_verdict(json.dumps(document).encode("utf-8"))


def test_verdict_message_rejects_non_object_message() -> None:
    with pytest.raises(ValueError, match="verdict"):
        decode_verdict(json.dumps(["not", "an", "object"]).encode("utf-8"))


def test_verdict_message_rejects_unknown_decision() -> None:
    document = json.loads(
        encode_verdict(
            Verdict.block(
                (
                    DetectionSignal(
                        "ruleset",
                        blocked=True,
                        reason="SQLi pattern",
                        score=1.0,
                    ),
                )
            )
        )
    )
    document["decision"] = "DROP"

    with pytest.raises(ValueError, match="decision"):
        decode_verdict(json.dumps(document).encode("utf-8"))


def test_verdict_message_rejects_non_list_signals() -> None:
    document = json.loads(
        encode_verdict(
            Verdict.block(
                (
                    DetectionSignal(
                        "ruleset",
                        blocked=True,
                        reason="SQLi pattern",
                        score=1.0,
                    ),
                )
            )
        )
    )
    document["signals"] = {"detector": "ruleset"}

    with pytest.raises(ValueError, match="signals"):
        decode_verdict(json.dumps(document).encode("utf-8"))


def test_verdict_message_rejects_non_object_signal() -> None:
    document = json.loads(
        encode_verdict(
            Verdict.block(
                (
                    DetectionSignal(
                        "ruleset",
                        blocked=True,
                        reason="SQLi pattern",
                        score=1.0,
                    ),
                )
            )
        )
    )
    document["signals"] = ["not-object"]

    with pytest.raises(ValueError, match="signals"):
        decode_verdict(json.dumps(document).encode("utf-8"))


def test_verdict_message_rejects_non_boolean_signal_blocked_flag() -> None:
    document = json.loads(
        encode_verdict(
            Verdict.block(
                (
                    DetectionSignal(
                        "ruleset",
                        blocked=True,
                        reason="SQLi pattern",
                        score=1.0,
                    ),
                )
            )
        )
    )
    document["signals"][0]["blocked"] = "false"

    with pytest.raises(ValueError, match="signals.blocked"):
        decode_verdict(json.dumps(document).encode("utf-8"))


def test_verdict_message_rejects_unknown_signal_action() -> None:
    document = json.loads(
        encode_verdict(
            Verdict.block(
                (
                    DetectionSignal(
                        "ruleset",
                        blocked=True,
                        reason="SQLi pattern",
                        score=1.0,
                    ),
                )
            )
        )
    )
    document["signals"][0]["action"] = "DROP"

    with pytest.raises(ValueError, match="signals.action"):
        decode_verdict(json.dumps(document).encode("utf-8"))


def test_verdict_message_rejects_non_finite_signal_score() -> None:
    document = json.loads(
        encode_verdict(
            Verdict.block(
                (
                    DetectionSignal(
                        "ruleset",
                        blocked=True,
                        reason="SQLi pattern",
                        score=1.0,
                    ),
                )
            )
        )
    )
    document["signals"][0]["score"] = "NaN"

    with pytest.raises(ValueError, match="signals.score"):
        decode_verdict(json.dumps(document).encode("utf-8"))


def test_verdict_message_carries_operational_decision() -> None:
    verdict = Verdict.challenge(
        (
            DetectionSignal(
                "workflow",
                blocked=False,
                reason="transition anomaly",
                score=2.0,
                action=SignalAction.CHALLENGE,
            ),
        )
    )

    message = decode_verdict(encode_verdict(verdict))

    assert message.blocked is False
    assert message.decision == "CHALLENGE"
    assert "transition anomaly" in message.reason


def test_verdict_message_carries_structured_signal_details() -> None:
    verdict = Verdict.alert(
        (
            DetectionSignal(
                "workflow",
                blocked=False,
                reason="transition anomaly",
                score=2.5,
                action=SignalAction.ALERT,
            ),
            DetectionSignal("ruleset", blocked=False, reason="no rule matched", score=0.0),
        )
    )

    message = decode_verdict(encode_verdict(verdict))

    assert len(message.signals) == 2
    assert message.signals[0].detector == "workflow"
    assert message.signals[0].action == "ALERT"
    assert message.signals[0].blocked is False
    assert message.signals[0].score == 2.5
    assert message.signals[0].reason == "transition anomaly"


def test_verdict_message_carries_runtime_identity_metadata() -> None:
    verdict = Verdict.block(
        (DetectionSignal("ruleset", blocked=True, reason="SQLi pattern", score=1.0),)
    )

    message = decode_verdict(
        encode_verdict(
            verdict,
            correlation_id="req-1",
            tenant_id="tenant-a",
            service_id="shop",
            runtime_version="deploy-7",
            config_fingerprint="cfg123",
        )
    )

    assert message.correlation_id == "req-1"
    assert message.tenant_id == "tenant-a"
    assert message.service_id == "shop"
    assert message.runtime_version == "deploy-7"
    assert message.config_fingerprint == "cfg123"
