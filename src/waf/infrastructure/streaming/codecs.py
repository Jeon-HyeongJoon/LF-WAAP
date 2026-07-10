"""codecs — 도메인 객체 ↔ 파이프 바이트(JSON) 직렬화.

파이프(Kafka/인메모리)에는 바이트만 흐른다. 여기서 Flow를 JSON으로 인코딩/디코딩한다.
payload·packet은 임의 바이트/메타라 base64·중첩 객체로 안전하게 싣는다.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.flow import Direction, Flow, PacketMeta
from waf.domain.model.http_request import HttpRequest
from waf.domain.model.verdict import Verdict
from waf.infrastructure.runtime import WafAuditRecord


def _packet_to_dict(packet: PacketMeta) -> dict[str, object]:
    return {
        "direction": packet.direction.value,
        "length": packet.length,
        "inter_arrival_ms": packet.inter_arrival_ms,
        "tcp_flags": packet.tcp_flags,
    }


def _packet_from_dict(raw: dict[str, Any]) -> PacketMeta:
    return PacketMeta(
        direction=Direction(str(raw["direction"])),
        length=int(raw["length"]),
        inter_arrival_ms=float(raw["inter_arrival_ms"]),
        tcp_flags=str(raw["tcp_flags"]),
    )


def _flow_to_dict(flow: Flow) -> dict[str, object]:
    return {
        "protocol": flow.protocol,
        "port": flow.port,
        "direction": flow.direction.value,
        "payload_b64": base64.b64encode(flow.payload).decode("ascii"),
        "packets": [_packet_to_dict(p) for p in flow.packets],
    }


def _flow_from_dict(document: dict[str, Any]) -> Flow:
    return Flow(
        protocol=str(document["protocol"]),
        port=int(document["port"]),
        direction=Direction(str(document["direction"])),
        payload=base64.b64decode(document["payload_b64"]),
        packets=tuple(_packet_from_dict(p) for p in document["packets"]),
    )


def encode_flow(flow: Flow) -> bytes:
    """Flow → JSON 바이트. payload는 base64로 실어 바이너리도 안전하게 전송."""
    return json.dumps(_flow_to_dict(flow)).encode("utf-8")


def decode_flow(message: bytes) -> Flow:
    """JSON 바이트 → Flow (encode_flow의 역변환)."""
    return _flow_from_dict(_json_object_from_message(message, "flow"))


def encode_labeled_flow(flow: Flow, is_attack: bool) -> bytes:
    """평가용 라벨 Flow → JSON 바이트. Flow 문서에 정답 라벨(is_attack)을 함께 싣는다."""
    document = _flow_to_dict(flow)
    document["is_attack"] = is_attack
    return json.dumps(document).encode("utf-8")


def decode_labeled_flow(message: bytes) -> tuple[Flow, bool]:
    """JSON 바이트 → (Flow, is_attack). 모델 테스트(평가) 토픽 소비용."""
    document = _json_object_from_message(message, "labeled flow")
    is_attack = document["is_attack"]
    if not isinstance(is_attack, bool):
        raise ValueError("labeled flow field must be a boolean: is_attack")
    return _flow_from_dict(document), is_attack


def _json_object_from_message(message: bytes, message_type: str) -> dict[str, Any]:
    document = json.loads(message)
    if not isinstance(document, dict):
        raise ValueError(f"{message_type} message must be a JSON object")
    return document


# --- 실시간 추론: HttpRequest(입력) / Verdict(출력) -------------------------


def encode_request(request: HttpRequest) -> bytes:
    """HttpRequest → JSON 바이트. 추론 요청 토픽(waf.inspect.requests)에 싣는다."""
    document = {
        "method": request.method,
        "path": request.path,
        "query": request.query,
        "headers": dict(request.headers),
        "body": request.body,
        "client_ip": request.client_ip,
    }
    return json.dumps(document).encode("utf-8")


def decode_request(message: bytes) -> HttpRequest:
    """JSON 바이트 → HttpRequest (encode_request의 역변환)."""
    document = json.loads(message)
    if not isinstance(document, dict):
        raise ValueError("request message must be a JSON object")
    return HttpRequest(
        method=_required_request_text(document, "method"),
        path=_required_request_text(document, "path"),
        query=_optional_request_text(document, "query"),
        headers=_request_headers(document.get("headers", {})),
        body=_optional_request_text(document, "body"),
        client_ip=_optional_request_text(document, "client_ip"),
    )


def _required_request_text(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str):
        raise ValueError(f"request field must be a string: {field}")
    if not value.strip():
        raise ValueError(f"request field is required: {field}")
    return value


def _optional_request_text(document: dict[str, Any], field: str) -> str:
    value = document.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"request field must be a string: {field}")
    return value


def _request_headers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("request field must be an object: headers")
    headers: dict[str, str] = {}
    for name, header_value in value.items():
        if not isinstance(name, str) or not isinstance(header_value, str):
            raise ValueError("request headers must contain string keys and values")
        headers[name] = header_value
    return headers


@dataclass(frozen=True, slots=True)
class SignalMessage:
    detector: str
    action: str
    blocked: bool
    reason: str
    score: float


@dataclass(frozen=True, slots=True)
class VerdictMessage:
    """판정 결과의 파이프용 요약. 출력 토픽 소비측이 쓰기 쉽게 평탄화."""

    blocked: bool
    decision: str
    reason: str
    signals: tuple[SignalMessage, ...]
    correlation_id: str = ""
    tenant_id: str = ""
    service_id: str = ""
    runtime_version: str = ""
    config_fingerprint: str = ""


def _signal_to_dict(signal: DetectionSignal) -> dict[str, object]:
    return {
        "detector": signal.detector,
        "action": signal.action.value,
        "blocked": signal.blocked,
        "reason": signal.reason,
        "score": signal.score,
    }


def _signal_from_dict(document: dict[str, Any]) -> SignalMessage:
    return SignalMessage(
        detector=str(document["detector"]),
        action=str(document["action"]),
        blocked=_required_bool(document["blocked"], "signals.blocked"),
        reason=str(document["reason"]),
        score=float(document["score"]),
    )


def encode_verdict(
    verdict: Verdict,
    *,
    correlation_id: str = "",
    tenant_id: str = "",
    service_id: str = "",
    runtime_version: str = "",
    config_fingerprint: str = "",
) -> bytes:
    """Verdict → JSON 바이트. 판정 결과 토픽(waf.inspect.verdicts)에 싣는다."""
    document = {
        "correlation_id": correlation_id,
        "tenant_id": tenant_id,
        "service_id": service_id,
        "runtime_version": runtime_version,
        "config_fingerprint": config_fingerprint,
        "blocked": verdict.is_blocked,
        "decision": verdict.decision.value,
        "reason": verdict.reason,
        "signals": [_signal_to_dict(signal) for signal in verdict.signals],
    }
    return json.dumps(document).encode("utf-8")


def decode_verdict(message: bytes) -> VerdictMessage:
    """JSON 바이트 → VerdictMessage (판정 요약)."""
    document = json.loads(message)
    return VerdictMessage(
        blocked=_required_bool(document["blocked"], "blocked"),
        decision=str(document["decision"]),
        reason=str(document["reason"]),
        signals=tuple(_signal_from_dict(signal) for signal in document.get("signals", ())),
        correlation_id=str(document.get("correlation_id", "")),
        tenant_id=str(document.get("tenant_id", "")),
        service_id=str(document.get("service_id", "")),
        runtime_version=str(document.get("runtime_version", "")),
        config_fingerprint=str(document.get("config_fingerprint", "")),
    )


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"message field must be a boolean: {field}")
    return value


def encode_audit_record(audit: WafAuditRecord) -> bytes:
    """WafAuditRecord → JSON 바이트. 감사 토픽(waf.inspect.audit)에 싣는다."""
    return json.dumps(audit.to_dict()).encode("utf-8")
