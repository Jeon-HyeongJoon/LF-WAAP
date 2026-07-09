import json
from datetime import UTC, datetime

from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.runtime import OperationalWaf, WafRuntimeConfig
from waf.infrastructure.streaming.codecs import decode_verdict, encode_request
from waf.infrastructure.streaming.inference import StreamingRuntimeInspector
from waf.infrastructure.streaming.memory import InMemoryChannel


def test_runtime_inspector_emits_verdicts_and_audit_records() -> None:
    channel = InMemoryChannel()
    waf = OperationalWaf.from_config(WafRuntimeConfig(tenant_id="t1", service_id="shop"))
    channel.publish(
        "waf.inspect.requests",
        encode_request(
            HttpRequest(
                "GET",
                "/search",
                query="q=' OR 1=1--",
                headers={"X-Request-Id": "req-123"},
                client_ip="203.0.113.10",
            )
        ),
    )

    processed = StreamingRuntimeInspector(channel, waf).run()

    assert processed == 1
    verdict = decode_verdict(channel.poll("waf.inspect.verdicts")[0])
    assert verdict.decision == "BLOCK"
    assert verdict.correlation_id == "req-123"
    assert verdict.tenant_id == "t1"
    assert verdict.service_id == "shop"
    assert verdict.runtime_version == "local"
    assert verdict.config_fingerprint == waf.status().config_fingerprint
    audits = channel.poll("waf.inspect.audit")
    assert len(audits) == 1
    audit = json.loads(audits[0])
    assert audit["correlation_id"] == "req-123"
    assert audit["tenant_id"] == "t1"
    assert audit["service_id"] == "shop"
    assert audit["client_ip"] == "203.0.113.10"
    assert audit["inspected_at"]
    assert audit["inspection_duration_ms"] >= 0.0
    assert audit["decision"] == "BLOCK"
    assert audit["event"]["route_template"] == "/search"
    assert audit["signals"][0]["detector"] == "ruleset"


def test_runtime_inspector_publishes_errors_and_continues_after_bad_message() -> None:
    channel = InMemoryChannel()
    waf = OperationalWaf.from_config(WafRuntimeConfig(tenant_id="t1", service_id="shop"))
    channel.publish("waf.inspect.requests", b"{not-json")
    channel.publish(
        "waf.inspect.requests",
        encode_request(HttpRequest("GET", "/products", query="id=42")),
    )

    consumed = StreamingRuntimeInspector(channel, waf).run()

    assert consumed == 2
    assert len(channel.poll("waf.inspect.errors")) == 1
    error = json.loads(channel.poll("waf.inspect.errors")[0])
    assert error["stage"] == "decode_or_inspect"
    assert error["message_type"] == "request"
    assert error["correlation_id"]
    assert error["tenant_id"] == "t1"
    assert error["service_id"] == "shop"
    assert error["runtime_version"] == "local"
    assert error["config_fingerprint"]
    assert len(channel.poll("waf.inspect.verdicts")) == 1
    assert len(channel.poll("waf.inspect.audit")) == 1


def test_runtime_inspector_error_records_have_schema_and_utc_timestamp() -> None:
    channel = InMemoryChannel()
    waf = OperationalWaf.from_config(WafRuntimeConfig(tenant_id="t1", service_id="shop"))
    channel.publish("waf.inspect.requests", b"{not-json")

    before = datetime.now(UTC)
    StreamingRuntimeInspector(channel, waf).run()
    after = datetime.now(UTC)

    error = json.loads(channel.poll("waf.inspect.errors")[0])
    failed_at = datetime.fromisoformat(str(error["failed_at"]))
    assert error["schema_version"] == 1
    assert failed_at.tzinfo is UTC
    assert before <= failed_at <= after


def test_runtime_inspector_error_records_fingerprint_messages_without_raw_payload() -> None:
    channel = InMemoryChannel()
    waf = OperationalWaf.from_config(WafRuntimeConfig(tenant_id="t1", service_id="shop"))
    bad_message = b"{not-json"
    channel.publish("waf.inspect.requests", bad_message)
    channel.publish("waf.inspect.requests", bad_message)

    StreamingRuntimeInspector(channel, waf).run()

    errors = [json.loads(message) for message in channel.poll("waf.inspect.errors")]
    assert len(errors) == 2
    assert errors[0]["message_fingerprint"]
    assert errors[0]["message_fingerprint"] == errors[1]["message_fingerprint"]
    assert errors[0]["message_size_bytes"] == len(bad_message)
    assert "{not-json" not in json.dumps(errors)


def test_runtime_inspector_reports_operational_counts() -> None:
    channel = InMemoryChannel()
    waf = OperationalWaf.from_config(WafRuntimeConfig(tenant_id="t1", service_id="shop"))
    channel.publish("waf.inspect.requests", b"{not-json")
    channel.publish(
        "waf.inspect.requests",
        encode_request(HttpRequest("GET", "/products", query="id=42")),
    )

    report = StreamingRuntimeInspector(channel, waf).run_report()

    assert report.consumed == 2
    assert report.inspected == 1
    assert report.failed == 1
    assert report.verdicts_published == 1
    assert report.audit_records_published == 1
    assert report.errors_published == 1


def test_runtime_inspector_can_limit_messages_per_run() -> None:
    channel = InMemoryChannel()
    waf = OperationalWaf.from_config(WafRuntimeConfig(tenant_id="t1", service_id="shop"))
    channel.publish(
        "waf.inspect.requests",
        encode_request(HttpRequest("GET", "/products", query="id=42")),
    )
    channel.publish(
        "waf.inspect.requests",
        encode_request(HttpRequest("GET", "/search", query="q=test")),
    )

    report = StreamingRuntimeInspector(channel, waf).run_report(max_messages=1)

    assert report.consumed == 1
    assert report.inspected == 1
    assert len(channel.poll("waf.inspect.verdicts")) == 1
    assert len(channel.poll("waf.inspect.audit")) == 1
