"""CI smoke for the CSIC training/evaluation MVP path.

The real CSIC CSV files are intentionally ignored because they are large. This
script creates a deterministic CSIC-shaped fixture, then exercises the same
learning/evaluation code used for the full dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learning_data"))

from calibrate_cli import evaluate_model, train_model  # noqa: E402
from csic_to_packets import iter_balanced_packets  # noqa: E402
from csic_traffic_source import record_to_request  # noqa: E402
from waf.domain.model.http_request import HttpRequest  # noqa: E402
from waf.infrastructure.runtime import OperationalWaf, WafRuntimeConfig  # noqa: E402
from waf.infrastructure.streaming.codecs import decode_verdict, encode_verdict  # noqa: E402
from waf.infrastructure.streaming.codecs import encode_request  # noqa: E402
from waf.infrastructure.streaming.inference import StreamingRuntimeInspector  # noqa: E402
from waf.infrastructure.streaming.memory import InMemoryChannel  # noqa: E402
from waf.infrastructure.behavior import CanonicalEventMapper  # noqa: E402
from waf.infrastructure.behavior.workflow_detector import (  # noqa: E402
    EnforcementMode,
    WorkflowBehaviorDetector,
)

HEADER = "classification,Method,host,content-type,lenght,content,URL"
HOST = "localhost:8080"
RULESET_PATH = ROOT / "src" / "waf" / "infrastructure" / "ruleset" / "rules.yaml"


def _row(label: int, method: str, path: str, body: str) -> str:
    return (
        f"{label},{method},{HOST},application/x-www-form-urlencoded,"
        f"{len(body.encode('latin-1'))},{body},http://{HOST}{path} HTTP/1.1"
    )


def _normal_add_cart(i: int) -> str:
    body = f"id={i}&nombre=Vino{i}&precio=100&cantidad=1&B1=Add"
    return _row(0, "POST", "/tienda1/publico/anadir.jsp", body)


def _normal_pay(i: int) -> str:
    body = f"modo=tarjeta&id={i}&precio=100&B1=Pagar"
    return _row(0, "POST", "/tienda1/publico/pagar.jsp", body)


def _attack_sqli(i: int) -> str:
    body = f"id={i}&nombre=Vino{i}&precio=100&cantidad=1&B1=%27OR%271%27%3D%271"
    return _row(1, "POST", "/tienda1/publico/anadir.jsp", body)


def _attack_lfi(i: int) -> str:
    body = f"modo=tarjeta&id=..%2F..%2Fetc%2Fpasswd&precio={i}&B1=Pagar"
    return _row(1, "POST", "/tienda1/publico/pagar.jsp", body)


def _request(method: str, path: str, body: str, session_id: str) -> HttpRequest:
    return HttpRequest(
        method,
        path,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Session-Id": session_id,
        },
        body=body,
    )


def write_fixture(output_dir: Path) -> tuple[Path, Path]:
    fixture_dir = output_dir / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    normal_csv = fixture_dir / "csic_mvp_normal.csv"
    eval_csv = fixture_dir / "csic_mvp_eval.csv"

    normal_rows = [
        row
        for i in range(36)
        for row in (
            _normal_add_cart(i),
            _normal_pay(i),
        )
    ]
    eval_rows = [
        *(_normal_add_cart(100 + i) for i in range(12)),
        *(_normal_pay(200 + i) for i in range(12)),
        *(_attack_sqli(300 + i) for i in range(12)),
        *(_attack_lfi(400 + i) for i in range(12)),
    ]

    normal_csv.write_text(HEADER + "\n" + "\n".join(normal_rows) + "\n", encoding="latin-1")
    eval_csv.write_text(HEADER + "\n" + "\n".join(eval_rows) + "\n", encoding="latin-1")
    return normal_csv, eval_csv


def discover_actions(eval_csv: Path) -> dict[str, object]:
    mapper = CanonicalEventMapper(tenant_id="ci", service_id="csic-mvp")
    action_ids: set[str] = set()
    route_templates: set[str] = set()
    events = 0
    for record in iter_balanced_packets(eval_csv):
        event = mapper.map(record_to_request(record))
        events += 1
        action_ids.add(event.action_id)
        route_templates.add(f"{event.method} {event.route_template}")
    return {
        "events": events,
        "action_count": len(action_ids),
        "route_templates": sorted(route_templates),
        "action_ids": sorted(action_ids),
    }


def verify_workflow_detection() -> dict[str, object]:
    mapper = CanonicalEventMapper(tenant_id="ci", service_id="csic-mvp-workflow")
    detector = WorkflowBehaviorDetector(mapper=mapper, mode=EnforcementMode.BLOCK)
    normal_sequences = [
        [
            _request("POST", "/tienda1/publico/anadir.jsp", f"id={i}&cantidad=1&B1=Add", f"train-{i}"),
            _request("POST", "/tienda1/publico/pagar.jsp", f"id={i}&precio=100&B1=Pagar", f"train-{i}"),
        ]
        for i in range(12)
    ]
    validation_sequences = [
        [
            _request("POST", "/tienda1/publico/anadir.jsp", f"id={i}&cantidad=1&B1=Add", f"val-{i}"),
            _request("POST", "/tienda1/publico/pagar.jsp", f"id={i}&precio=100&B1=Pagar", f"val-{i}"),
        ]
        for i in range(12, 16)
    ]
    detector.train_request_sequences(normal_sequences, validation_sequences)

    normal_first = detector.inspect(
        _request("POST", "/tienda1/publico/anadir.jsp", "id=99&cantidad=1&B1=Add", "ok")
    )
    normal_next = detector.inspect(
        _request("POST", "/tienda1/publico/pagar.jsp", "id=99&precio=100&B1=Pagar", "ok")
    )
    anomaly = detector.inspect(
        _request("POST", "/tienda1/publico/pagar.jsp", "id=100&precio=100&B1=Pagar", "jump")
    )

    return {
        "normal_first_blocked": normal_first.blocked,
        "normal_next_blocked": normal_next.blocked,
        "jump_blocked": anomaly.blocked,
        "jump_reason": anomaly.reason,
        "jump_score": anomaly.score,
    }


def verify_workflow_snapshot_boot(output_dir: Path) -> dict[str, object]:
    service_id = "csic-mvp-workflow"
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="ci", service_id=service_id),
        mode=EnforcementMode.BLOCK,
    )
    normal_sequences = [
        [
            _request("POST", "/tienda1/publico/anadir.jsp", f"id={i}&cantidad=1&B1=Add", f"train-{i}"),
            _request("POST", "/tienda1/publico/pagar.jsp", f"id={i}&precio=100&B1=Pagar", f"train-{i}"),
        ]
        for i in range(12)
    ]
    validation_sequences = [
        [
            _request("POST", "/tienda1/publico/anadir.jsp", f"id={i}&cantidad=1&B1=Add", f"val-{i}"),
            _request("POST", "/tienda1/publico/pagar.jsp", f"id={i}&precio=100&B1=Pagar", f"val-{i}"),
        ]
        for i in range(12, 16)
    ]
    trainer.train_request_sequences(normal_sequences, validation_sequences)
    artifact_path = output_dir / "workflow_model_snapshot.json"
    trainer.export_model_snapshot().save(artifact_path)

    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="ci",
            service_id=service_id,
            runtime_version="ci-smoke",
            workflow_mode=EnforcementMode.BLOCK,
            workflow_model_path=artifact_path,
        ),
    )
    status = waf.status().to_dict()
    waf.inspect(
        _request("POST", "/tienda1/publico/anadir.jsp", "id=99&cantidad=1&B1=Add", "boot")
    )
    outcome = waf.inspect_with_audit(
        _request("POST", "/tienda1/publico/pagar.jsp", "id=100&precio=100&B1=Pagar", "jump")
    )
    verdict = outcome.verdict
    audit = outcome.audit.to_dict()
    workflow_status = status["detectors"]["workflow"]
    assert isinstance(workflow_status, dict)
    metadata = workflow_status["metadata"]
    assert isinstance(metadata, dict)
    audit_fingerprints = audit["detector_fingerprints"]
    assert isinstance(audit_fingerprints, dict)
    return {
        "artifact_path": str(artifact_path),
        "ready": workflow_status["ready"],
        "model_fingerprint": metadata["model_fingerprint"],
        "audit_model_fingerprint": audit_fingerprints["workflow"],
        "jump_blocked": verdict.is_blocked,
    }


def verify_runtime_status() -> dict[str, object]:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig.from_env(
            {
                "LF_WAAP_TENANT_ID": "ci",
                "LF_WAAP_SERVICE_ID": "csic-mvp",
                "LF_WAAP_RUNTIME_VERSION": "ci-smoke",
                "LF_WAAP_RULESET_PATH": str(RULESET_PATH),
            }
        )
    )
    status = waf.status()
    return status.to_dict()


def verify_structured_verdict_message() -> dict[str, object]:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="ci",
            service_id="csic-mvp",
            runtime_version="ci-smoke",
            ruleset_path=RULESET_PATH,
        )
    )
    status = waf.status().to_dict()
    request = HttpRequest(
        "GET",
        "/search",
        query="q=' OR 1=1--",
        headers={"Host": HOST, "X-Request-Id": "ci-direct-req-1"},
        client_ip="203.0.113.20",
    )
    outcome = waf.inspect_with_audit(request)
    verdict = outcome.verdict
    audit = outcome.audit.to_dict()
    event = audit["event"]
    assert isinstance(event, dict)
    signals = audit["signals"]
    assert isinstance(signals, list)
    detectors = status["detectors"]
    assert isinstance(detectors, dict)
    ruleset_status = detectors["ruleset"]
    assert isinstance(ruleset_status, dict)
    ruleset_metadata = ruleset_status["metadata"]
    assert isinstance(ruleset_metadata, dict)
    detector_fingerprints = audit["detector_fingerprints"]
    assert isinstance(detector_fingerprints, dict)
    message = decode_verdict(encode_verdict(verdict))
    metrics = waf.metrics().to_dict()
    decisions = metrics["decisions"]
    signals_by_detector = metrics["signals_by_detector"]
    signal_actions_by_detector = metrics["signal_actions_by_detector"]
    requests_by_route_template = metrics["requests_by_route_template"]
    decisions_by_route_template = metrics["decisions_by_route_template"]
    requests_by_business_action = metrics["requests_by_business_action"]
    business_action_catalog = metrics["business_action_catalog"]
    assert isinstance(decisions, dict)
    assert isinstance(signals_by_detector, dict)
    assert isinstance(signal_actions_by_detector, dict)
    assert isinstance(requests_by_route_template, dict)
    assert isinstance(decisions_by_route_template, dict)
    assert isinstance(requests_by_business_action, dict)
    assert isinstance(business_action_catalog, dict)
    ruleset_signal_actions = signal_actions_by_detector["ruleset"]
    workflow_signal_actions = signal_actions_by_detector["workflow"]
    search_decisions = decisions_by_route_template["GET /search"]
    search_action_id = next(iter(requests_by_business_action))
    search_action = business_action_catalog[search_action_id]
    assert isinstance(ruleset_signal_actions, dict)
    assert isinstance(workflow_signal_actions, dict)
    assert isinstance(search_decisions, dict)
    assert isinstance(search_action, dict)
    reset_snapshot = waf.reset_metrics().to_dict()
    reset_metrics = waf.metrics().to_dict()
    return {
        "decision": message.decision,
        "blocked": message.blocked,
        "signal_count": len(message.signals),
        "first_signal": {
            "detector": message.signals[0].detector,
            "action": message.signals[0].action,
            "blocked": message.signals[0].blocked,
        },
        "audit_event_route": event["route_template"],
        "audit_schema_version": audit["schema_version"],
        "audit_tenant_id": audit["tenant_id"],
        "audit_service_id": audit["service_id"],
        "audit_correlation_id": audit["correlation_id"],
        "audit_client_ip": audit["client_ip"],
        "audit_signal_count": len(signals),
        "audit_has_inspected_at": bool(audit["inspected_at"]),
        "audit_duration_ms": float(audit["inspection_duration_ms"]),
        "audit_runtime_version": audit["runtime_version"],
        "audit_config_fingerprint": audit["config_fingerprint"],
        "metrics_schema_version": metrics["schema_version"],
        "metrics_started_at": metrics["started_at"],
        "metrics_window_started_at": metrics["metrics_window_started_at"],
        "metrics_uptime_seconds": metrics["uptime_seconds"],
        "metrics_inspected_total": metrics["inspected_total"],
        "metrics_blocked_total": decisions["BLOCK"],
        "metrics_allowed_total": decisions["ALLOW"],
        "metrics_duration_total_ms": metrics["inspection_duration_total_ms"],
        "metrics_duration_avg_ms": metrics["inspection_duration_avg_ms"],
        "metrics_duration_max_ms": metrics["inspection_duration_max_ms"],
        "metrics_signals_total": metrics["signals_total"],
        "metrics_ruleset_signals": signals_by_detector["ruleset"],
        "metrics_workflow_signals": signals_by_detector["workflow"],
        "metrics_ruleset_block_signals": ruleset_signal_actions["BLOCK"],
        "metrics_workflow_allow_signals": workflow_signal_actions["ALLOW"],
        "metrics_search_route_count": requests_by_route_template["GET /search"],
        "metrics_search_route_blocked": search_decisions["BLOCK"],
        "metrics_business_action_count": len(requests_by_business_action),
        "metrics_business_action_catalog_count": len(business_action_catalog),
        "metrics_business_action_route": search_action["route_template"],
        "metrics_business_action_method": search_action["method"],
        "metrics_business_action_schema": search_action["param_schema_id"],
        "metrics_business_action_auth_state": search_action["auth_state"],
        "reset_snapshot_inspected_total": reset_snapshot["inspected_total"],
        "reset_snapshot_window_started_at": reset_snapshot["metrics_window_started_at"],
        "reset_metrics_window_started_at": reset_metrics["metrics_window_started_at"],
        "reset_metrics_inspected_total": reset_metrics["inspected_total"],
        "reset_metrics_signal_total": reset_metrics["signals_total"],
        "reset_metrics_route_count": len(reset_metrics["requests_by_route_template"]),
        "reset_metrics_business_action_catalog_count": len(
            reset_metrics["business_action_catalog"]
        ),
        "ruleset_fingerprint_matched": (
            detector_fingerprints["ruleset"] == ruleset_metadata["model_fingerprint"]
        ),
    }


def verify_ruleset_only_audit_event() -> dict[str, object]:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="ci",
            service_id="csic-mvp",
            runtime_version="ci-smoke",
            ruleset_path=RULESET_PATH,
            enable_workflow=False,
        )
    )
    outcome = waf.inspect_with_audit(
        HttpRequest("GET", "/products/42", query="id=42", headers={"Host": HOST})
    )
    audit = outcome.audit.to_dict()
    event = audit["event"]
    assert isinstance(event, dict)
    return {
        "audit_correlation_id": audit["correlation_id"],
        "event_route": event["route_template"],
        "event_method": event["method"],
        "event_action_id": event["action_id"],
        "event_tenant_id": event["tenant_id"],
        "event_service_id": event["service_id"],
    }


def verify_streaming_audit_topic() -> dict[str, object]:
    channel = InMemoryChannel()
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="ci",
            service_id="csic-mvp",
            runtime_version="ci-smoke",
            ruleset_path=RULESET_PATH,
        )
    )
    channel.publish("waf.inspect.requests", b"{not-json")
    channel.publish(
        "waf.inspect.requests",
        encode_request(
            HttpRequest(
                "GET",
                "/search",
                query="q=' OR 1=1--",
                headers={"Host": HOST, "X-Request-Id": "ci-req-1"},
                client_ip="203.0.113.30",
            )
        ),
    )
    report = StreamingRuntimeInspector(channel, waf).run_report()
    verdict = decode_verdict(channel.poll("waf.inspect.verdicts")[0])
    audit_document = json.loads(channel.poll("waf.inspect.audit")[0])
    error_document = json.loads(channel.poll("waf.inspect.errors")[0])
    return {
        "processed": report.consumed,
        "inspected": report.inspected,
        "failed": report.failed,
        "verdicts_published": report.verdicts_published,
        "audit_records_published": report.audit_records_published,
        "errors_published": report.errors_published,
        "verdict_count": len(channel.poll("waf.inspect.verdicts")),
        "audit_count": len(channel.poll("waf.inspect.audit")),
        "error_count": len(channel.poll("waf.inspect.errors")),
        "verdict_correlation_id": verdict.correlation_id,
        "verdict_tenant_id": verdict.tenant_id,
        "verdict_service_id": verdict.service_id,
        "verdict_runtime_version": verdict.runtime_version,
        "verdict_has_config_fingerprint": bool(verdict.config_fingerprint),
        "audit_tenant_id": audit_document["tenant_id"],
        "audit_service_id": audit_document["service_id"],
        "audit_schema_version": audit_document["schema_version"],
        "audit_correlation_id": audit_document["correlation_id"],
        "audit_client_ip": audit_document["client_ip"],
        "audit_has_inspected_at": bool(audit_document["inspected_at"]),
        "audit_duration_ms": float(audit_document["inspection_duration_ms"]),
        "error_has_correlation_id": bool(error_document["correlation_id"]),
        "error_schema_version": error_document["schema_version"],
        "error_has_failed_at": bool(error_document["failed_at"]),
        "error_has_message_fingerprint": bool(error_document["message_fingerprint"]),
        "error_message_size_bytes": error_document["message_size_bytes"],
        "error_tenant_id": error_document["tenant_id"],
        "error_service_id": error_document["service_id"],
        "error_runtime_version": error_document["runtime_version"],
        "error_has_config_fingerprint": bool(error_document["config_fingerprint"]),
    }


def run(output_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_csv, eval_csv = write_fixture(output_dir)
    model_path = output_dir / "csic_mvp_hmm.pkl"

    train = train_model(
        normal_csv,
        model_path,
        validation_ratio=0.2,
        n_states=3,
        n_iter=15,
        target_fpr=0.1,
    )
    outcome = evaluate_model(eval_csv, model_path, target_fpr=0.1)
    actions = discover_actions(eval_csv)
    workflow = verify_workflow_detection()
    workflow_snapshot_boot = verify_workflow_snapshot_boot(output_dir)
    runtime_status = verify_runtime_status()
    verdict_message = verify_structured_verdict_message()
    ruleset_only_audit = verify_ruleset_only_audit_event()
    streaming_audit = verify_streaming_audit_topic()

    if not model_path.exists():
        raise RuntimeError("CSIC MVP model was not written")
    if train.normal_count <= 0 or train.validation_count <= 0:
        raise RuntimeError("CSIC MVP training did not consume normal and validation data")
    if outcome.report is None or outcome.classification is None:
        raise RuntimeError("CSIC MVP evaluation did not produce metrics for both classes")
    if outcome.covered <= 0 or outcome.normal_stats.count <= 0 or outcome.attack_stats.count <= 0:
        raise RuntimeError("CSIC MVP evaluation did not cover normal and attack flows")
    if int(actions["action_count"]) < 2:
        raise RuntimeError("CSIC MVP action discovery did not identify multiple actions")
    if workflow["normal_first_blocked"] or workflow["normal_next_blocked"]:
        raise RuntimeError("CSIC MVP workflow detector blocked a normal sequence")
    if not workflow["jump_blocked"]:
        raise RuntimeError("CSIC MVP workflow detector did not block a jump transition")
    if (
        not workflow_snapshot_boot["ready"]
        or not workflow_snapshot_boot["model_fingerprint"]
        or workflow_snapshot_boot["audit_model_fingerprint"]
        != workflow_snapshot_boot["model_fingerprint"]
        or not workflow_snapshot_boot["jump_blocked"]
    ):
        raise RuntimeError("CSIC MVP workflow snapshot boot did not preserve detection")
    detectors = runtime_status["detectors"]
    assert isinstance(detectors, dict)
    ruleset_status = detectors["ruleset"]
    workflow_status = detectors["workflow"]
    assert isinstance(ruleset_status, dict)
    assert isinstance(workflow_status, dict)
    if not ruleset_status["ready"] or workflow_status["ready"]:
        raise RuntimeError("CSIC MVP runtime status did not expose expected detector readiness")
    if (
        runtime_status["ready"]
        or not runtime_status["serving"]
        or runtime_status["serving_detectors"] != ["ruleset"]
        or runtime_status["degraded_detectors"] != ["workflow"]
    ):
        raise RuntimeError("CSIC MVP runtime status did not expose degraded serving state")
    if runtime_status["runtime_version"] != "ci-smoke" or not runtime_status["config_fingerprint"]:
        raise RuntimeError("CSIC MVP runtime status did not expose runtime version metadata")
    if runtime_status["schema_version"] != 1:
        raise RuntimeError("CSIC MVP runtime status did not expose schema version")
    if not runtime_status["started_at"] or runtime_status["uptime_seconds"] < 0.0:
        raise RuntimeError("CSIC MVP runtime status did not expose lifecycle metadata")
    if runtime_status["tenant_id"] != "ci" or runtime_status["service_id"] != "csic-mvp":
        raise RuntimeError("CSIC MVP runtime status did not expose tenant/service identity")
    if (
        verdict_message["decision"] != "BLOCK"
        or int(verdict_message["signal_count"]) < 1
        or verdict_message["audit_tenant_id"] != "ci"
        or verdict_message["audit_service_id"] != "csic-mvp"
        or verdict_message["audit_schema_version"] != 1
        or verdict_message["audit_correlation_id"] != "ci-direct-req-1"
        or verdict_message["audit_client_ip"] != "203.0.113.20"
        or int(verdict_message["audit_signal_count"]) < 1
        or not verdict_message["audit_has_inspected_at"]
        or verdict_message["audit_duration_ms"] < 0.0
        or verdict_message["audit_runtime_version"] != "ci-smoke"
        or not verdict_message["audit_config_fingerprint"]
        or verdict_message["metrics_schema_version"] != 1
        or not verdict_message["metrics_started_at"]
        or not verdict_message["metrics_window_started_at"]
        or verdict_message["metrics_uptime_seconds"] < 0.0
        or verdict_message["metrics_inspected_total"] != 1
        or verdict_message["metrics_blocked_total"] != 1
        or verdict_message["metrics_allowed_total"] != 0
        or verdict_message["metrics_duration_total_ms"] < 0.0
        or verdict_message["metrics_duration_avg_ms"] < 0.0
        or verdict_message["metrics_duration_max_ms"] < verdict_message["metrics_duration_avg_ms"]
        or verdict_message["metrics_signals_total"] != 2
        or verdict_message["metrics_ruleset_signals"] != 1
        or verdict_message["metrics_workflow_signals"] != 1
        or verdict_message["metrics_ruleset_block_signals"] != 1
        or verdict_message["metrics_workflow_allow_signals"] != 1
        or verdict_message["metrics_search_route_count"] != 1
        or verdict_message["metrics_search_route_blocked"] != 1
        or verdict_message["metrics_business_action_count"] != 1
        or verdict_message["metrics_business_action_catalog_count"] != 1
        or verdict_message["metrics_business_action_route"] != "/search"
        or verdict_message["metrics_business_action_method"] != "GET"
        or verdict_message["metrics_business_action_schema"] != "query:q"
        or verdict_message["metrics_business_action_auth_state"] != "anonymous"
        or verdict_message["reset_snapshot_inspected_total"] != 1
        or not verdict_message["reset_snapshot_window_started_at"]
        or not verdict_message["reset_metrics_window_started_at"]
        or verdict_message["reset_metrics_window_started_at"]
        < verdict_message["reset_snapshot_window_started_at"]
        or verdict_message["reset_metrics_inspected_total"] != 0
        or verdict_message["reset_metrics_signal_total"] != 0
        or verdict_message["reset_metrics_route_count"] != 0
        or verdict_message["reset_metrics_business_action_catalog_count"] != 0
        or not verdict_message["ruleset_fingerprint_matched"]
    ):
        raise RuntimeError("CSIC MVP verdict message did not preserve structured signals")
    if (
        not ruleset_only_audit["audit_correlation_id"]
        or ruleset_only_audit["event_route"] != "/products/{num}"
        or ruleset_only_audit["event_method"] != "GET"
        or not str(ruleset_only_audit["event_action_id"]).startswith("auto:csic-mvp:act_")
        or ruleset_only_audit["event_tenant_id"] != "ci"
        or ruleset_only_audit["event_service_id"] != "csic-mvp"
    ):
        raise RuntimeError("CSIC MVP ruleset-only audit did not preserve canonical event")
    if (
        streaming_audit["processed"] != 2
        or streaming_audit["inspected"] != 1
        or streaming_audit["failed"] != 1
        or streaming_audit["verdicts_published"] != 1
        or streaming_audit["audit_records_published"] != 1
        or streaming_audit["errors_published"] != 1
        or streaming_audit["audit_count"] != 1
        or streaming_audit["error_count"] != 1
        or streaming_audit["verdict_correlation_id"] != "ci-req-1"
        or streaming_audit["verdict_tenant_id"] != "ci"
        or streaming_audit["verdict_service_id"] != "csic-mvp"
        or streaming_audit["verdict_runtime_version"] != "ci-smoke"
        or not streaming_audit["verdict_has_config_fingerprint"]
        or streaming_audit["audit_tenant_id"] != "ci"
        or streaming_audit["audit_service_id"] != "csic-mvp"
        or streaming_audit["audit_schema_version"] != 1
        or streaming_audit["audit_correlation_id"] != "ci-req-1"
        or streaming_audit["audit_client_ip"] != "203.0.113.30"
        or not streaming_audit["audit_has_inspected_at"]
        or streaming_audit["audit_duration_ms"] < 0.0
        or not streaming_audit["error_has_correlation_id"]
        or streaming_audit["error_schema_version"] != 1
        or not streaming_audit["error_has_failed_at"]
        or not streaming_audit["error_has_message_fingerprint"]
        or streaming_audit["error_message_size_bytes"] != len(b"{not-json")
        or streaming_audit["error_tenant_id"] != "ci"
        or streaming_audit["error_service_id"] != "csic-mvp"
        or streaming_audit["error_runtime_version"] != "ci-smoke"
        or not streaming_audit["error_has_config_fingerprint"]
    ):
        raise RuntimeError("CSIC MVP streaming runtime inspector did not emit audit records")

    metrics = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "fixture": {
            "normal_csv": str(normal_csv),
            "eval_csv": str(eval_csv),
        },
        "model_path": str(model_path),
        "training": asdict(train),
        "evaluation": {
            "covered": outcome.covered,
            "uncovered": outcome.uncovered,
            "normal": asdict(outcome.normal_stats),
            "attack": asdict(outcome.attack_stats),
            "report": asdict(outcome.report),
            "classification": asdict(outcome.classification),
        },
        "actions": actions,
        "workflow": workflow,
        "workflow_snapshot_boot": workflow_snapshot_boot,
        "runtime_status": runtime_status,
        "verdict_message": verdict_message,
        "ruleset_only_audit": ruleset_only_audit,
        "streaming_audit": streaming_audit,
    }
    (output_dir / "csic_mvp_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "csic_mvp_summary.md").write_text(render_summary(metrics), encoding="utf-8")
    return metrics


def render_summary(metrics: dict[str, object]) -> str:
    training = metrics["training"]
    evaluation = metrics["evaluation"]
    actions = metrics["actions"]
    workflow = metrics["workflow"]
    workflow_snapshot_boot = metrics["workflow_snapshot_boot"]
    runtime_status = metrics["runtime_status"]
    verdict_message = metrics["verdict_message"]
    ruleset_only_audit = metrics["ruleset_only_audit"]
    streaming_audit = metrics["streaming_audit"]
    assert isinstance(training, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(actions, dict)
    assert isinstance(workflow, dict)
    assert isinstance(workflow_snapshot_boot, dict)
    assert isinstance(runtime_status, dict)
    assert isinstance(verdict_message, dict)
    assert isinstance(ruleset_only_audit, dict)
    assert isinstance(streaming_audit, dict)
    runtime_detectors = runtime_status["detectors"]
    assert isinstance(runtime_detectors, dict)
    runtime_ruleset = runtime_detectors["ruleset"]
    runtime_workflow = runtime_detectors["workflow"]
    assert isinstance(runtime_ruleset, dict)
    assert isinstance(runtime_workflow, dict)
    report = evaluation["report"]
    classification = evaluation["classification"]
    assert isinstance(report, dict)
    assert isinstance(classification, dict)
    return "\n".join(
        [
            "# CSIC MVP Smoke",
            "",
            f"- elapsed_seconds: {metrics['elapsed_seconds']}",
            f"- train_normal: {training['normal_count']}",
            f"- train_validation: {training['validation_count']}",
            f"- eval_covered: {evaluation['covered']}",
            f"- eval_uncovered: {evaluation['uncovered']}",
            f"- action_count: {actions['action_count']}",
            f"- workflow_jump_blocked: {workflow['jump_blocked']}",
            f"- workflow_snapshot_ready: {workflow_snapshot_boot['ready']}",
            f"- workflow_snapshot_jump_blocked: {workflow_snapshot_boot['jump_blocked']}",
            f"- workflow_audit_fingerprint_matched: "
            f"{workflow_snapshot_boot['audit_model_fingerprint'] == workflow_snapshot_boot['model_fingerprint']}",
            f"- runtime_ruleset_ready: {runtime_ruleset['ready']}",
            f"- runtime_workflow_ready: {runtime_workflow['ready']}",
            f"- runtime_ready: {runtime_status['ready']}",
            f"- runtime_serving: {runtime_status['serving']}",
            f"- runtime_identity: {runtime_status['tenant_id']}/{runtime_status['service_id']}",
            f"- runtime_version: {runtime_status['runtime_version']}",
            f"- runtime_uptime_seconds: {float(runtime_status['uptime_seconds']):.3f}",
            f"- verdict_signal_count: {verdict_message['signal_count']}",
            f"- runtime_metrics_inspected: {verdict_message['metrics_inspected_total']}",
            f"- runtime_metrics_blocked: {verdict_message['metrics_blocked_total']}",
            f"- runtime_metrics_duration_avg_ms: "
            f"{float(verdict_message['metrics_duration_avg_ms']):.3f}",
            f"- runtime_metrics_signals: {verdict_message['metrics_signals_total']}",
            f"- runtime_metrics_search_route_count: "
            f"{verdict_message['metrics_search_route_count']}",
            f"- runtime_metrics_business_action_catalog: "
            f"{verdict_message['metrics_business_action_catalog_count']} "
            f"({verdict_message['metrics_business_action_method']} "
            f"{verdict_message['metrics_business_action_route']})",
            f"- audit_signal_count: {verdict_message['audit_signal_count']}",
            f"- ruleset_only_audit_route: {ruleset_only_audit['event_route']}",
            f"- audit_has_inspected_at: {verdict_message['audit_has_inspected_at']}",
            f"- audit_duration_ms: {float(verdict_message['audit_duration_ms']):.3f}",
            f"- ruleset_audit_fingerprint_matched: {verdict_message['ruleset_fingerprint_matched']}",
            f"- streaming_consumed: {streaming_audit['processed']}",
            f"- streaming_inspected: {streaming_audit['inspected']}",
            f"- streaming_failed: {streaming_audit['failed']}",
            f"- streaming_audit_count: {streaming_audit['audit_count']}",
            f"- streaming_error_count: {streaming_audit['error_count']}",
            f"- streaming_correlation_id: {streaming_audit['audit_correlation_id']}",
            f"- roc_auc: {float(report['roc_auc']):.3f}",
            f"- f1: {float(classification['f1']):.3f}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/ci/csic-mvp")
    args = parser.parse_args()
    summary = render_summary(run(Path(args.output_dir)))
    print(summary)


if __name__ == "__main__":
    main()
