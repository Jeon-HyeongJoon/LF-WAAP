import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.http_request import HttpRequest
from waf.domain.model.verdict import Decision
from waf.domain.service.block_decision_service import BlockPolicy
from waf.infrastructure.runtime import OperationalWaf, WafRuntimeConfig
from waf.infrastructure.behavior.workflow_detector import (
    EnforcementMode,
    WorkflowBehaviorDetector,
    WorkflowModelSnapshot,
)
from waf.infrastructure.behavior import CanonicalEventMapper


def _request(method: str, path: str, body: str = "", session: str = "") -> HttpRequest:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if session:
        headers["X-Session-Id"] = session
    return HttpRequest(method, path, headers=headers, body=body)


def _checkout(session: str) -> list[HttpRequest]:
    return [
        _request("POST", "/cart/items", "sku=1&quantity=1", session),
        _request("POST", "/checkout/payment", "orderId=1&paymentToken=tok", session),
        _request("POST", "/orders/1/confirm", "orderId=1", session),
    ]


class _NamedDetector:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        return DetectionSignal(self._name, blocked=False, reason="stub", score=0.0)


def test_operational_waf_reports_detector_statuses() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", runtime_version="deploy-1")
    )

    status = waf.status()

    assert status.runtime_version == "deploy-1"
    assert status.config_fingerprint
    assert status.started_at
    assert status.uptime_seconds >= 0.0
    assert status.detectors["ruleset"].ready is True
    assert status.detectors["workflow"].ready is False
    assert status.ready is False
    assert status.serving is True
    document = status.to_dict()
    assert document["ready"] is False
    assert document["schema_version"] == 1
    assert document["serving"] is True
    assert document["serving_detectors"] == ["ruleset"]
    assert document["degraded_detectors"] == ["workflow"]
    assert document["tenant_id"] == "t1"
    assert document["service_id"] == "shop"
    assert document["runtime_version"] == "deploy-1"
    assert document["config_fingerprint"] == status.config_fingerprint
    assert datetime.fromisoformat(str(document["started_at"])).tzinfo is UTC
    assert document["uptime_seconds"] >= 0.0
    assert document["detectors"]["workflow"]["reason"] == "not trained"
    json.dumps(document)


def test_operational_waf_applies_configured_workflow_mode_after_training() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            workflow_mode=EnforcementMode.CHALLENGE,
            workflow_max_sessions=10,
        )
    )
    assert waf.workflow_detector is not None
    waf.workflow_detector.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )

    waf.inspect(_request("POST", "/cart/items", "sku=1&quantity=1", "u1"))
    verdict = waf.inspect(_request("POST", "/orders/999/confirm", "orderId=999", "u1"))

    assert verdict.decision is Decision.CHALLENGE
    assert not verdict.is_blocked
    assert "transition anomaly" in verdict.reason

    status = waf.status()
    workflow = status.detectors["workflow"]
    assert status.ready is True
    assert workflow.ready is True
    assert workflow.metadata["mode"] == "CHALLENGE"
    assert workflow.metadata["action_count"] >= 3
    assert workflow.metadata["transition_count"] >= 3
    assert workflow.metadata["session_ttl_seconds"] == 1800.0
    assert workflow.metadata["max_sessions"] == 10
    assert workflow.metadata["session_count"] >= 1
    assert workflow.metadata["threshold"] is not None


def test_operational_waf_blocks_ruleset_hits_without_workflow_training() -> None:
    waf = OperationalWaf.from_config(WafRuntimeConfig(tenant_id="t1", service_id="shop"))

    verdict = waf.inspect(HttpRequest("GET", "/search", query="q=' OR 1=1--"))

    assert verdict.decision is Decision.BLOCK
    assert verdict.is_blocked
    assert "ruleset" in verdict.reason


def test_operational_waf_reports_runtime_decision_metrics() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            runtime_version="deploy-1",
            enable_workflow=False,
        )
    )

    assert waf.metrics().to_dict()["inspected_total"] == 0

    waf.inspect(HttpRequest("GET", "/products/42", query="id=42"))
    waf.inspect(HttpRequest("GET", "/search", query="q=' OR 1=1--"))

    document = waf.metrics().to_dict()
    assert document["schema_version"] == 1
    assert document["tenant_id"] == "t1"
    assert document["service_id"] == "shop"
    assert document["runtime_version"] == "deploy-1"
    assert document["config_fingerprint"] == waf.status().config_fingerprint
    assert datetime.fromisoformat(str(document["started_at"])).tzinfo is UTC
    assert document["uptime_seconds"] >= 0.0
    assert document["inspected_total"] == 2
    assert document["decisions"] == {
        "ALLOW": 1,
        "ALERT": 0,
        "CHALLENGE": 0,
        "BLOCK": 1,
    }
    assert document["inspection_duration_total_ms"] >= 0.0
    assert document["inspection_duration_avg_ms"] >= 0.0
    assert document["inspection_duration_max_ms"] >= document["inspection_duration_avg_ms"]
    assert document["signals_total"] == 2
    assert document["signals_by_detector"] == {"ruleset": 2}
    assert document["signal_actions_by_detector"] == {
        "ruleset": {
            "ALLOW": 1,
            "ALERT": 0,
            "CHALLENGE": 0,
            "BLOCK": 1,
        }
    }
    assert document["requests_by_route_template"] == {
        "GET /products/{num}": 1,
        "GET /search": 1,
    }
    assert document["decisions_by_route_template"] == {
        "GET /products/{num}": {
            "ALLOW": 1,
            "ALERT": 0,
            "CHALLENGE": 0,
            "BLOCK": 0,
        },
        "GET /search": {
            "ALLOW": 0,
            "ALERT": 0,
            "CHALLENGE": 0,
            "BLOCK": 1,
        },
    }
    action_counts = document["requests_by_business_action"]
    assert sum(action_counts.values()) == 2
    assert all(action_id.startswith("auto:shop:act_") for action_id in action_counts)


def test_runtime_metrics_explain_auto_business_actions_without_raw_values() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            enable_workflow=False,
        )
    )

    waf.inspect(
        HttpRequest(
            "POST",
            "/orders/42/confirm",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body="orderId=42&paymentToken=tok-secret",
        )
    )

    document = waf.metrics().to_dict()
    action_counts = document["requests_by_business_action"]
    assert isinstance(action_counts, dict)
    action_id = next(iter(action_counts))
    catalog = document["business_action_catalog"]

    assert catalog == {
        action_id: {
            "action_id": action_id,
            "method": "POST",
            "route_template": "/orders/{num}/confirm",
            "param_schema_id": "body:orderId,paymentToken",
            "auth_state": "anonymous",
            "action_confidence": 1.0,
        }
    }
    assert "tok-secret" not in json.dumps(catalog)


def test_audited_inspection_updates_runtime_decision_metrics() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )

    outcome = waf.inspect_with_audit(HttpRequest("GET", "/search", query="q=' OR 1=1--"))

    assert outcome.verdict.decision is Decision.BLOCK
    document = waf.metrics().to_dict()
    assert document["inspected_total"] == 1
    assert document["decisions"]["BLOCK"] == 1
    assert document["inspection_duration_total_ms"] >= 0.0
    assert document["inspection_duration_avg_ms"] >= 0.0
    assert document["inspection_duration_max_ms"] >= document["inspection_duration_avg_ms"]
    assert document["signals_by_detector"] == {"ruleset": 1}


def test_operational_waf_can_reset_runtime_metrics_without_restarting() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )
    started_at = waf.metrics().to_dict()["started_at"]
    initial_window_started_at = waf.metrics().to_dict()["metrics_window_started_at"]
    assert initial_window_started_at == started_at
    waf.inspect(HttpRequest("GET", "/search", query="q=' OR 1=1--"))

    before_reset = waf.reset_metrics().to_dict()
    after_reset = waf.metrics().to_dict()

    assert before_reset["inspected_total"] == 1
    assert before_reset["metrics_window_started_at"] == initial_window_started_at
    assert after_reset["started_at"] == started_at
    reset_window = datetime.fromisoformat(str(after_reset["metrics_window_started_at"]))
    runtime_started = datetime.fromisoformat(str(started_at))
    assert reset_window >= runtime_started
    assert after_reset["inspected_total"] == 0
    assert after_reset["decisions"] == {
        "ALLOW": 0,
        "ALERT": 0,
        "CHALLENGE": 0,
        "BLOCK": 0,
    }
    assert after_reset["inspection_duration_total_ms"] == 0.0
    assert after_reset["inspection_duration_avg_ms"] == 0.0
    assert after_reset["inspection_duration_max_ms"] == 0.0
    assert after_reset["signals_total"] == 0
    assert after_reset["signals_by_detector"] == {}
    assert after_reset["signal_actions_by_detector"] == {}
    assert after_reset["requests_by_route_template"] == {}
    assert after_reset["decisions_by_route_template"] == {}
    assert after_reset["requests_by_business_action"] == {}
    assert after_reset["business_action_catalog"] == {}


def test_runtime_metrics_caps_route_and_business_action_cardinality() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            enable_workflow=False,
            metrics_max_cardinality=1,
        )
    )

    waf.inspect(HttpRequest("GET", "/products/42", query="id=42"))
    waf.inspect(HttpRequest("GET", "/search", query="q=' OR 1=1--"))

    document = waf.metrics().to_dict()
    assert document["requests_by_route_template"] == {
        "GET /products/{num}": 1,
        "__other__": 1,
    }
    assert document["decisions_by_route_template"] == {
        "GET /products/{num}": {
            "ALLOW": 1,
            "ALERT": 0,
            "CHALLENGE": 0,
            "BLOCK": 0,
        },
        "__other__": {
            "ALLOW": 0,
            "ALERT": 0,
            "CHALLENGE": 0,
            "BLOCK": 1,
        },
    }
    action_counts = document["requests_by_business_action"]
    assert action_counts["__other__"] == 1
    assert sum(action_counts.values()) == 2
    assert len(action_counts) == 2
    assert "__other__" not in document["business_action_catalog"]
    assert len(document["business_action_catalog"]) == 1


def test_audit_record_identifies_ruleset_fingerprint() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )
    expected_fingerprint = waf.status().detectors["ruleset"].metadata["model_fingerprint"]

    outcome = waf.inspect_with_audit(HttpRequest("GET", "/search", query="q=' OR 1=1--"))

    document = outcome.audit.to_dict()
    assert document["schema_version"] == 1
    assert document["tenant_id"] == "t1"
    assert document["service_id"] == "shop"
    assert document["detector_fingerprints"]["ruleset"] == expected_fingerprint


def test_audit_record_uses_request_id_header_as_correlation_id() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )

    outcome = waf.inspect_with_audit(
        HttpRequest(
            "GET",
            "/search",
            query="q=test",
            headers={"X-Request-Id": "req-direct-1"},
        )
    )

    assert outcome.audit.to_dict()["correlation_id"] == "req-direct-1"


def test_audit_record_includes_client_ip_for_investigation() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )

    outcome = waf.inspect_with_audit(
        HttpRequest("GET", "/search", query="q=test", client_ip="203.0.113.9")
    )

    assert outcome.audit.to_dict()["client_ip"] == "203.0.113.9"


def test_audit_record_generates_correlation_id_when_request_id_is_absent() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )

    first = waf.inspect_with_audit(HttpRequest("GET", "/products", query="id=1"))
    second = waf.inspect_with_audit(HttpRequest("GET", "/products", query="id=2"))

    first_id = first.audit.to_dict()["correlation_id"]
    second_id = second.audit.to_dict()["correlation_id"]
    assert first_id
    assert second_id
    assert first_id != second_id


def test_ruleset_only_audit_record_includes_canonical_event() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )

    outcome = waf.inspect_with_audit(HttpRequest("GET", "/products/42", query="id=42"))

    event = outcome.audit.to_dict()["event"]
    assert event["tenant_id"] == "t1"
    assert event["service_id"] == "shop"
    assert event["method"] == "GET"
    assert event["route_template"] == "/products/{num}"
    assert event["action_id"].startswith("auto:shop:act_")


def test_audit_record_includes_utc_inspection_timestamp() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )

    before = datetime.now(UTC)
    outcome = waf.inspect_with_audit(HttpRequest("GET", "/search", query="q=' OR 1=1--"))
    after = datetime.now(UTC)

    inspected_at = datetime.fromisoformat(str(outcome.audit.to_dict()["inspected_at"]))
    assert inspected_at.tzinfo is UTC
    assert before <= inspected_at <= after


def test_audit_record_includes_inspection_duration_ms() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(tenant_id="t1", service_id="shop", enable_workflow=False)
    )

    outcome = waf.inspect_with_audit(HttpRequest("GET", "/search", query="q=' OR 1=1--"))

    duration_ms = outcome.audit.to_dict()["inspection_duration_ms"]
    assert isinstance(duration_ms, float)
    assert duration_ms >= 0.0


def test_operational_waf_can_boot_from_custom_ruleset_artifact(tmp_path) -> None:
    ruleset_path = tmp_path / "rules.yaml"
    ruleset_path.write_text(
        """
rules:
  - id: CUSTOM-ADMIN
    description: custom admin path block
    pattern: "/internal-admin"
""".lstrip(),
        encoding="utf-8",
    )
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            ruleset_path=ruleset_path,
            enable_workflow=False,
        )
    )

    status = waf.status().detectors["ruleset"]
    verdict = waf.inspect(HttpRequest("GET", "/internal-admin"))
    audit = waf.inspect_with_audit(HttpRequest("GET", "/internal-admin")).audit.to_dict()

    assert verdict.decision is Decision.BLOCK
    assert "CUSTOM-ADMIN" in verdict.reason
    assert status.metadata["rule_count"] == 1
    assert audit["detector_fingerprints"]["ruleset"] == status.metadata["model_fingerprint"]


def test_operational_waf_rejects_invalid_ruleset_artifact(tmp_path) -> None:
    ruleset_path = tmp_path / "rules.yaml"
    ruleset_path.write_text(
        """
rules:
  - id: BAD-REGEX
    description: invalid regex
    pattern: "["
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="BAD-REGEX"):
        OperationalWaf.from_config(
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                ruleset_path=ruleset_path,
                enable_workflow=False,
            )
        )


def test_operational_waf_rejects_ruleset_fingerprint_mismatch(tmp_path) -> None:
    ruleset_path = tmp_path / "rules.yaml"
    ruleset_path.write_text(
        """
fingerprint: deadbeef
rules:
  - id: CUSTOM-ADMIN
    description: custom admin path block
    pattern: "/internal-admin"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        OperationalWaf.from_config(
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                ruleset_path=ruleset_path,
                enable_workflow=False,
            )
        )


def test_operational_waf_can_boot_with_serialized_workflow_model_snapshot() -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    snapshot_document = trainer.export_model_snapshot().to_dict()
    json.dumps(snapshot_document)

    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            workflow_mode=EnforcementMode.BLOCK,
        ),
        workflow_model_snapshot=WorkflowModelSnapshot.from_dict(snapshot_document),
    )

    status = waf.status()
    workflow = status.detectors["workflow"]
    assert status.ready is True
    assert status.serving is True
    assert status.serving_detectors == ("ruleset", "workflow")
    assert status.degraded_detectors == ()
    assert workflow.ready is True
    assert workflow.metadata["model_fingerprint"]

    waf.inspect(_request("POST", "/cart/items", "sku=1&quantity=1", "u1"))
    verdict = waf.inspect(_request("POST", "/orders/999/confirm", "orderId=999", "u1"))

    assert verdict.decision is Decision.BLOCK
    assert "transition anomaly" in verdict.reason


def test_operational_waf_can_serve_with_workflow_only_snapshot() -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )

    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            workflow_mode=EnforcementMode.BLOCK,
            enable_ruleset=False,
        ),
        workflow_model_snapshot=trainer.export_model_snapshot(),
    )

    status = waf.status()
    assert status.ready is True
    assert status.serving is True
    assert status.serving_detectors == ("workflow",)


def test_operational_waf_can_boot_from_workflow_model_artifact_file(tmp_path) -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    artifact_path = tmp_path / "models" / "workflow-model.json"
    trainer.export_model_snapshot().save(artifact_path)

    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            workflow_mode=EnforcementMode.BLOCK,
            workflow_model_path=artifact_path,
        ),
    )

    assert artifact_path.exists()
    workflow_status = waf.status().detectors["workflow"]
    assert workflow_status.ready is True
    assert workflow_status.metadata["model_fingerprint"]
    waf.inspect(_request("POST", "/cart/items", "sku=1&quantity=1", "u1"))
    verdict = waf.inspect(_request("POST", "/orders/999/confirm", "orderId=999", "u1"))

    assert verdict.decision is Decision.BLOCK


def test_workflow_model_artifact_records_tenant_and_service(tmp_path) -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    artifact_path = tmp_path / "workflow-model.json"
    trainer.export_model_snapshot().save(artifact_path)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    loaded = WorkflowModelSnapshot.load(artifact_path)

    assert artifact["tenant_id"] == "t1"
    assert artifact["service_id"] == "shop"
    assert loaded.tenant_id == "t1"
    assert loaded.service_id == "shop"


@pytest.mark.parametrize("context_field", ["tenant_id", "service_id"])
def test_workflow_model_artifact_load_rejects_missing_tenant_or_service(
    tmp_path, context_field: str
) -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    artifact_path = tmp_path / "workflow-model.json"
    trainer.export_model_snapshot().save(artifact_path)
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    document.pop(context_field)
    artifact_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=context_field):
        WorkflowModelSnapshot.load(artifact_path)


@pytest.mark.parametrize("context_field", ["tenant_id", "service_id"])
def test_workflow_model_artifact_load_rejects_context_with_surrounding_whitespace(
    tmp_path, context_field: str
) -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    artifact_path = tmp_path / "workflow-model.json"
    trainer.export_model_snapshot().save(artifact_path)
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    document[context_field] = f" {document[context_field]} "
    artifact_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=context_field):
        WorkflowModelSnapshot.load(artifact_path)


def test_operational_waf_rejects_workflow_model_for_different_service(tmp_path) -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="other-shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    artifact_path = tmp_path / "workflow-model.json"
    trainer.export_model_snapshot().save(artifact_path)

    with pytest.raises(ValueError, match="service_id"):
        OperationalWaf.from_config(
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_mode=EnforcementMode.BLOCK,
                workflow_model_path=artifact_path,
            )
        )


def test_operational_waf_rejects_workflow_model_for_different_tenant(tmp_path) -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="tenant-other", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    artifact_path = tmp_path / "workflow-model.json"
    trainer.export_model_snapshot().save(artifact_path)

    with pytest.raises(ValueError, match="tenant_id"):
        OperationalWaf.from_config(
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_mode=EnforcementMode.BLOCK,
                workflow_model_path=artifact_path,
            )
        )


def test_audit_record_identifies_workflow_model_fingerprint(tmp_path) -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    artifact_path = tmp_path / "workflow-model.json"
    trainer.export_model_snapshot().save(artifact_path)
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            workflow_mode=EnforcementMode.BLOCK,
            workflow_model_path=artifact_path,
        )
    )
    expected_fingerprint = waf.status().detectors["workflow"].metadata["model_fingerprint"]
    waf.inspect(_request("POST", "/cart/items", "sku=1&quantity=1", "u1"))

    outcome = waf.inspect_with_audit(
        _request("POST", "/orders/999/confirm", "orderId=999", "u1")
    )

    document = outcome.audit.to_dict()
    assert document["detector_fingerprints"]["workflow"] == expected_fingerprint


def test_workflow_model_artifact_load_rejects_fingerprint_mismatch(tmp_path) -> None:
    trainer = WorkflowBehaviorDetector(
        mapper=CanonicalEventMapper(tenant_id="t1", service_id="shop"),
        mode=EnforcementMode.BLOCK,
    )
    trainer.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    artifact_path = tmp_path / "workflow-model.json"
    trainer.export_model_snapshot().save(artifact_path)
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    document["model"]["threshold"] = -999.0
    artifact_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        WorkflowModelSnapshot.load(artifact_path)


def test_workflow_model_snapshot_rejects_negative_transition_count() -> None:
    with pytest.raises(ValueError, match="transition count"):
        WorkflowModelSnapshot.from_dict(
            {
                "target_fpr": 0.01,
                "alpha": 0.1,
                "transitions": {"<START>": {"auto:shop:act_bad": -1}},
                "actions": ["auto:shop:act_bad"],
                "threshold": -1.0,
            }
        )


def test_workflow_model_snapshot_rejects_boolean_transition_count() -> None:
    with pytest.raises(ValueError, match="transition count"):
        WorkflowModelSnapshot.from_dict(
            {
                "target_fpr": 0.01,
                "alpha": 0.1,
                "transitions": {"<START>": {"auto:shop:act_cart": True}},
                "actions": ["auto:shop:act_cart"],
                "threshold": -1.0,
            }
        )


def test_workflow_model_snapshot_rejects_null_action_id() -> None:
    with pytest.raises(ValueError, match="action"):
        WorkflowModelSnapshot.from_dict(
            {
                "target_fpr": 0.01,
                "alpha": 0.1,
                "transitions": {"<START>": {"auto:shop:act_cart": 3}},
                "actions": [None],
                "threshold": -1.0,
            }
        )


def test_workflow_model_snapshot_rejects_duplicate_action_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        WorkflowModelSnapshot.from_dict(
            {
                "target_fpr": 0.01,
                "alpha": 0.1,
                "transitions": {"<START>": {"auto:shop:act_cart": 3}},
                "actions": ["auto:shop:act_cart", "auto:shop:act_cart"],
                "threshold": -1.0,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_fpr", True),
        ("alpha", "Infinity"),
        ("alpha", False),
        ("threshold", "NaN"),
    ],
)
def test_workflow_model_snapshot_rejects_non_finite_numeric_values(
    field: str, value: str
) -> None:
    document = {
        "target_fpr": 0.01,
        "alpha": 0.1,
        "transitions": {"<START>": {"auto:shop:act_cart": 3}},
        "actions": ["auto:shop:act_cart"],
        "threshold": -1.0,
    }
    document[field] = value

    with pytest.raises(ValueError, match=field):
        WorkflowModelSnapshot.from_dict(document)


def test_workflow_model_artifact_load_rejects_untrained_model(tmp_path) -> None:
    artifact_path = tmp_path / "workflow-model.json"
    WorkflowModelSnapshot(
        target_fpr=0.01,
        alpha=0.1,
        transitions={},
        actions=(),
        threshold=None,
        tenant_id="t1",
        service_id="shop",
    ).save(artifact_path)

    with pytest.raises(ValueError, match="trained"):
        WorkflowModelSnapshot.load(artifact_path)


def test_operational_waf_returns_json_safe_audit_record() -> None:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="t1",
            service_id="shop",
            workflow_mode=EnforcementMode.ALERT,
            runtime_version="deploy-audit",
        )
    )
    assert waf.workflow_detector is not None
    waf.workflow_detector.train_request_sequences(
        [_checkout(f"train-{i}") for i in range(8)],
        [_checkout(f"val-{i}") for i in range(2)],
    )
    waf.inspect(_request("POST", "/cart/items", "sku=1&quantity=1", "u1"))

    outcome = waf.inspect_with_audit(
        _request("POST", "/orders/999/confirm", "orderId=999", "u1")
    )

    assert outcome.verdict.decision is Decision.ALERT
    document = outcome.audit.to_dict()
    assert document["runtime_version"] == "deploy-audit"
    assert document["config_fingerprint"] == waf.status().config_fingerprint
    assert document["decision"] == "ALERT"
    assert document["event"]["tenant_id"] == "t1"
    assert document["event"]["service_id"] == "shop"
    assert document["event"]["route_template"] == "/orders/{num}/confirm"
    assert document["event"]["session_id"].startswith("header:x-session-id:")
    assert document["signals"][0]["detector"] == "ruleset"
    assert document["signals"][1]["detector"] == "workflow"
    assert document["signals"][1]["action"] == "ALERT"
    json.dumps(document)


def test_runtime_config_can_be_loaded_from_environment_mapping() -> None:
    config = WafRuntimeConfig.from_env(
        {
            "LF_WAAP_TENANT_ID": "tenant-a",
            "LF_WAAP_SERVICE_ID": "shop",
            "LF_WAAP_RUNTIME_VERSION": "deploy-42",
            "LF_WAAP_WORKFLOW_MODE": "block",
            "LF_WAAP_WORKFLOW_MODEL_PATH": "models/workflow.json",
            "LF_WAAP_WORKFLOW_SESSION_TTL_SECONDS": "60.5",
            "LF_WAAP_WORKFLOW_MAX_SESSIONS": "2500",
            "LF_WAAP_BLOCK_POLICY": "all",
            "LF_WAAP_RULESET_PATH": "rules/custom.yaml",
            "LF_WAAP_ENABLE_RULESET": "true",
            "LF_WAAP_ENABLE_WORKFLOW": "false",
            "LF_WAAP_METRICS_MAX_CARDINALITY": "500",
        }
    )

    assert config.tenant_id == "tenant-a"
    assert config.service_id == "shop"
    assert config.runtime_version == "deploy-42"
    assert config.workflow_mode is EnforcementMode.BLOCK
    assert config.workflow_model_path == Path("models/workflow.json")
    assert config.workflow_session_ttl_seconds == 60.5
    assert config.workflow_max_sessions == 2500
    assert config.block_policy is BlockPolicy.ALL
    assert config.ruleset_path == Path("rules/custom.yaml")
    assert config.enable_ruleset is True
    assert config.enable_workflow is False
    assert config.metrics_max_cardinality == 500


@pytest.mark.parametrize(
    ("env", "message"),
    [
        (
            {"LF_WAAP_SERVICE_ID": "shop"},
            "LF_WAAP_TENANT_ID",
        ),
        (
            {
                "LF_WAAP_TENANT_ID": "tenant-a",
                "LF_WAAP_SERVICE_ID": "shop",
                "LF_WAAP_WORKFLOW_MODE": "panic",
            },
            "LF_WAAP_WORKFLOW_MODE",
        ),
        (
            {
                "LF_WAAP_TENANT_ID": "tenant-a",
                "LF_WAAP_SERVICE_ID": "shop",
                "LF_WAAP_ENABLE_WORKFLOW": "maybe",
            },
            "LF_WAAP_ENABLE_WORKFLOW",
        ),
        (
            {
                "LF_WAAP_TENANT_ID": "tenant-a",
                "LF_WAAP_SERVICE_ID": "shop",
                "LF_WAAP_WORKFLOW_MAX_SESSIONS": "many",
            },
            "LF_WAAP_WORKFLOW_MAX_SESSIONS",
        ),
        (
            {
                "LF_WAAP_TENANT_ID": "tenant-a",
                "LF_WAAP_SERVICE_ID": "shop",
                "LF_WAAP_METRICS_MAX_CARDINALITY": "many",
            },
            "LF_WAAP_METRICS_MAX_CARDINALITY",
        ),
    ],
)
def test_runtime_config_from_env_rejects_invalid_operational_values(
    env: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        WafRuntimeConfig.from_env(env)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                ruleset_path="missing-rules.yaml",
                enable_workflow=False,
            ),
            "ruleset_path",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_model_path="missing-workflow.json",
            ),
            "workflow_model_path",
        ),
    ],
)
def test_runtime_config_rejects_missing_operational_artifacts(
    config: WafRuntimeConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        OperationalWaf.from_config(config)


def test_runtime_config_rejects_directory_operational_artifact_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="ruleset_path"):
        OperationalWaf.from_config(
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                ruleset_path=tmp_path,
                enable_workflow=False,
            )
        )


def test_operational_waf_rejects_duplicate_detector_names() -> None:
    with pytest.raises(ValueError, match="duplicate detector name"):
        OperationalWaf.from_config(
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                enable_workflow=False,
            ),
            extra_detectors=(_NamedDetector("ruleset"),),
        )


def test_operational_waf_rejects_blank_detector_names() -> None:
    with pytest.raises(ValueError, match="detector name"):
        OperationalWaf.from_config(
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                enable_workflow=False,
            ),
            extra_detectors=(_NamedDetector(" "),),
        )


def test_operational_waf_rejects_detector_names_with_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="detector name"):
        OperationalWaf.from_config(
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                enable_workflow=False,
            ),
            extra_detectors=(_NamedDetector(" custom "),),
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (WafRuntimeConfig(tenant_id=1, service_id="shop"), "tenant_id"),
        (WafRuntimeConfig(tenant_id="", service_id="shop"), "tenant_id"),
        (WafRuntimeConfig(tenant_id="   ", service_id="shop"), "tenant_id"),
        (WafRuntimeConfig(tenant_id=" t1", service_id="shop"), "tenant_id"),
        (WafRuntimeConfig(tenant_id="t1 ", service_id="shop"), "tenant_id"),
        (WafRuntimeConfig(tenant_id="t1", service_id=None), "service_id"),
        (WafRuntimeConfig(tenant_id="t1", service_id=""), "service_id"),
        (WafRuntimeConfig(tenant_id="t1", service_id="\t"), "service_id"),
        (WafRuntimeConfig(tenant_id="t1", service_id=" shop"), "service_id"),
        (WafRuntimeConfig(tenant_id="t1", service_id="shop "), "service_id"),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                runtime_version=3,
            ),
            "runtime_version",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                runtime_version=" ",
            ),
            "runtime_version",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                runtime_version=" deploy-1",
            ),
            "runtime_version",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                runtime_version="deploy-1 ",
            ),
            "runtime_version",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_session_ttl_seconds=0.0,
            ),
            "workflow_session_ttl_seconds",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_session_ttl_seconds=True,
            ),
            "workflow_session_ttl_seconds",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_max_sessions=0,
            ),
            "workflow_max_sessions",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_max_sessions=True,
            ),
            "workflow_max_sessions",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                enable_ruleset="false",
            ),
            "enable_ruleset",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                enable_workflow=1,
            ),
            "enable_workflow",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_mode="BLOCK",
            ),
            "workflow_mode",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                block_policy="ANY",
            ),
            "block_policy",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                enable_ruleset=False,
                enable_workflow=False,
            ),
            "at least one detector",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                enable_ruleset=False,
                enable_workflow=True,
            ),
            "workflow_model",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                workflow_model_path="workflow-model.json",
                enable_workflow=False,
            ),
            "workflow_model_path",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                ruleset_path="rules.yaml",
                enable_ruleset=False,
            ),
            "ruleset_path",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                metrics_max_cardinality=0,
            ),
            "metrics_max_cardinality",
        ),
        (
            WafRuntimeConfig(
                tenant_id="t1",
                service_id="shop",
                metrics_max_cardinality=True,
            ),
            "metrics_max_cardinality",
        ),
    ],
)
def test_runtime_config_validation_rejects_invalid_operational_config(
    config: WafRuntimeConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        OperationalWaf.from_config(config)
