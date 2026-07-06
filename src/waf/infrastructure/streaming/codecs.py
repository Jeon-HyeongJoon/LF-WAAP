"""codecs — 도메인 객체 ↔ 파이프 바이트(JSON) 직렬화.

파이프(Kafka/인메모리)에는 바이트만 흐른다. 여기서 Flow를 JSON으로 인코딩/디코딩한다.
payload·packet은 임의 바이트/메타라 base64·중첩 객체로 안전하게 싣는다.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from waf.domain.model.flow import Direction, Flow, PacketMeta
from waf.domain.model.http_request import HttpRequest
from waf.domain.model.verdict import Verdict


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
    return _flow_from_dict(json.loads(message))


def encode_labeled_flow(flow: Flow, is_attack: bool) -> bytes:
    """평가용 라벨 Flow → JSON 바이트. Flow 문서에 정답 라벨(is_attack)을 함께 싣는다."""
    document = _flow_to_dict(flow)
    document["is_attack"] = is_attack
    return json.dumps(document).encode("utf-8")


def decode_labeled_flow(message: bytes) -> tuple[Flow, bool]:
    """JSON 바이트 → (Flow, is_attack). 모델 테스트(평가) 토픽 소비용."""
    document = json.loads(message)
    return _flow_from_dict(document), bool(document["is_attack"])


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
    return HttpRequest(
        method=str(document["method"]),
        path=str(document["path"]),
        query=str(document["query"]),
        headers=dict(document["headers"]),
        body=str(document["body"]),
        client_ip=str(document["client_ip"]),
    )


@dataclass(frozen=True, slots=True)
class VerdictMessage:
    """판정 결과의 파이프용 요약(차단 여부 + 사유). 출력 토픽 소비측이 쓰기 쉽게 평탄화."""

    blocked: bool
    decision: str  # "ALLOW" | "BLOCK"
    reason: str


def encode_verdict(verdict: Verdict) -> bytes:
    """Verdict → JSON 바이트. 판정 결과 토픽(waf.inspect.verdicts)에 싣는다."""
    document = {
        "blocked": verdict.is_blocked,
        "decision": verdict.decision.value,
        "reason": verdict.reason,
    }
    return json.dumps(document).encode("utf-8")


def decode_verdict(message: bytes) -> VerdictMessage:
    """JSON 바이트 → VerdictMessage (판정 요약)."""
    document = json.loads(message)
    return VerdictMessage(
        blocked=bool(document["blocked"]),
        decision=str(document["decision"]),
        reason=str(document["reason"]),
    )
