"""실시간 추론 파이프 — 요청 토픽을 소비해 WAF 판정을 판정 토픽으로 발행한다.

WAF 진입점인 BlockDecisionService(룰셋+HMM 조합)를 그대로 호출하므로, 학습한 HMM 모델이
실제 차단 결정에 어떻게 쓰이는지 한눈에 보인다: 요청 in → decide() → verdict out.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from collections.abc import Iterable
from uuid import uuid4

from waf.domain.model.http_request import HttpRequest
from waf.domain.service.block_decision_service import BlockDecisionService
from waf.infrastructure.runtime import OperationalWaf, RUNTIME_DOCUMENT_SCHEMA_VERSION
from waf.infrastructure.streaming.channel import MessageChannel
from waf.infrastructure.streaming.codecs import decode_request, encode_audit_record, encode_verdict

DEFAULT_REQUESTS_TOPIC = "waf.inspect.requests"
DEFAULT_VERDICTS_TOPIC = "waf.inspect.verdicts"
DEFAULT_AUDIT_TOPIC = "waf.inspect.audit"
DEFAULT_ERRORS_TOPIC = "waf.inspect.errors"


@dataclass(frozen=True, slots=True)
class StreamingRunReport:
    consumed: int = 0
    inspected: int = 0
    failed: int = 0
    verdicts_published: int = 0
    audit_records_published: int = 0
    errors_published: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "consumed": self.consumed,
            "inspected": self.inspected,
            "failed": self.failed,
            "verdicts_published": self.verdicts_published,
            "audit_records_published": self.audit_records_published,
            "errors_published": self.errors_published,
        }


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


class StreamingRuntimeInspector:
    def __init__(
        self,
        channel: MessageChannel,
        waf: OperationalWaf,
        requests_topic: str = DEFAULT_REQUESTS_TOPIC,
        verdicts_topic: str = DEFAULT_VERDICTS_TOPIC,
        audit_topic: str = DEFAULT_AUDIT_TOPIC,
        errors_topic: str = DEFAULT_ERRORS_TOPIC,
    ) -> None:
        self._channel = channel
        self._waf = waf
        self._requests_topic = requests_topic
        self._verdicts_topic = verdicts_topic
        self._audit_topic = audit_topic
        self._errors_topic = errors_topic
        status = waf.status()
        self._tenant_id = status.tenant_id
        self._service_id = status.service_id
        self._runtime_version = status.runtime_version
        self._config_fingerprint = status.config_fingerprint

    def run(self, *, max_messages: int | None = None) -> int:
        return self.run_report(max_messages=max_messages).consumed

    def run_report(self, *, max_messages: int | None = None) -> StreamingRunReport:
        if max_messages is not None:
            if isinstance(max_messages, bool) or not isinstance(max_messages, int):
                raise ValueError("max_messages must be a non-negative integer")
            if max_messages < 0:
                raise ValueError("max_messages must be a non-negative integer")
        consumed = 0
        inspected = 0
        failed = 0
        verdicts_published = 0
        audit_records_published = 0
        errors_published = 0
        for message in _limit_messages(
            self._channel.stream(self._requests_topic), max_messages=max_messages
        ):
            consumed += 1
            correlation_id = _new_correlation_id()
            try:
                request = decode_request(message)
                correlation_id = _correlation_id(request) or correlation_id
                outcome = self._waf.inspect_with_audit(request)
                audit = outcome.audit.with_correlation_id(correlation_id)
                inspected += 1
                self._channel.publish(
                    self._verdicts_topic,
                    encode_verdict(
                        outcome.verdict,
                        correlation_id=correlation_id,
                        tenant_id=self._tenant_id,
                        service_id=self._service_id,
                        runtime_version=self._runtime_version,
                        config_fingerprint=self._config_fingerprint,
                    ),
                )
                verdicts_published += 1
                self._channel.publish(self._audit_topic, encode_audit_record(audit))
                audit_records_published += 1
            except Exception as exc:
                self._channel.publish(
                    self._errors_topic,
                    _encode_error(
                        exc,
                        stage="decode_or_inspect",
                        message_type="request",
                        correlation_id=correlation_id,
                        tenant_id=self._tenant_id,
                        service_id=self._service_id,
                        runtime_version=self._runtime_version,
                        config_fingerprint=self._config_fingerprint,
                        message=message,
                    ),
                )
                failed += 1
                errors_published += 1
        return StreamingRunReport(
            consumed=consumed,
            inspected=inspected,
            failed=failed,
            verdicts_published=verdicts_published,
            audit_records_published=audit_records_published,
            errors_published=errors_published,
        )


def _limit_messages(messages: Iterable[bytes], *, max_messages: int | None) -> Iterable[bytes]:
    if max_messages is None:
        return messages
    return islice(messages, max_messages)


def _encode_error(
    exc: Exception,
    *,
    stage: str,
    message_type: str,
    correlation_id: str,
    tenant_id: str,
    service_id: str,
    runtime_version: str,
    config_fingerprint: str,
    message: bytes,
) -> bytes:
    document = {
        "schema_version": RUNTIME_DOCUMENT_SCHEMA_VERSION,
        "correlation_id": correlation_id,
        "tenant_id": tenant_id,
        "service_id": service_id,
        "runtime_version": runtime_version,
        "config_fingerprint": config_fingerprint,
        "failed_at": datetime.now(UTC).isoformat(),
        "message_fingerprint": _message_fingerprint(message),
        "message_size_bytes": len(message),
        "stage": stage,
        "message_type": message_type,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    return json.dumps(document).encode("utf-8")


def _correlation_id(request: HttpRequest) -> str:
    return str(request.headers.get("x-request-id", "")).strip()


def _new_correlation_id() -> str:
    return uuid4().hex


def _message_fingerprint(message: bytes) -> str:
    return hashlib.sha256(message).hexdigest()[:16]
