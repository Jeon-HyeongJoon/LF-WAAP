from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior import CanonicalEventMapper
from waf.infrastructure.behavior.session_state import ManualClock
from waf.infrastructure.behavior.workflow_detector import (
    EnforcementMode,
    WorkflowBehaviorDetector,
)
from waf.domain.model.detection import SignalAction


def _request(method: str, path: str, body: str = "", session: str = "") -> HttpRequest:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if session:
        headers["X-Session-Id"] = session
    return HttpRequest(method, path, headers=headers, body=body)


def _normal_checkout(session: str) -> list[HttpRequest]:
    return [
        _request("POST", "/cart/items", "sku=1&quantity=1", session),
        _request("POST", "/checkout/payment", "orderId=1&paymentToken=tok", session),
        _request("POST", "/orders/1/confirm", "orderId=1", session),
    ]


def test_workflow_detector_blocks_unusual_action_transition_in_block_mode() -> None:
    mapper = CanonicalEventMapper(tenant_id="tenant-a", service_id="shop")
    detector = WorkflowBehaviorDetector(mapper=mapper, mode=EnforcementMode.BLOCK)
    detector.train_request_sequences(
        [_normal_checkout(f"train-{i}") for i in range(8)],
        [_normal_checkout(f"val-{i}") for i in range(2)],
    )

    assert not detector.inspect(_request("POST", "/cart/items", "sku=1&quantity=1", "u1")).blocked
    signal = detector.inspect(_request("POST", "/orders/999/confirm", "orderId=999", "u1"))

    assert signal.blocked
    assert signal.detector == "workflow"
    assert "transition anomaly" in signal.reason


def test_workflow_detector_alerts_without_blocking_in_alert_mode() -> None:
    mapper = CanonicalEventMapper(tenant_id="tenant-a", service_id="shop")
    detector = WorkflowBehaviorDetector(mapper=mapper, mode=EnforcementMode.ALERT)
    detector.train_request_sequences(
        [_normal_checkout(f"train-{i}") for i in range(8)],
        [_normal_checkout(f"val-{i}") for i in range(2)],
    )

    detector.inspect(_request("POST", "/cart/items", "sku=1&quantity=1", "u1"))
    signal = detector.inspect(_request("POST", "/orders/999/confirm", "orderId=999", "u1"))

    assert not signal.blocked
    assert signal.action is SignalAction.ALERT
    assert "transition anomaly" in signal.reason


def test_workflow_detector_expires_session_state_after_ttl() -> None:
    clock = ManualClock()
    mapper = CanonicalEventMapper(tenant_id="tenant-a", service_id="shop")
    detector = WorkflowBehaviorDetector(
        mapper=mapper,
        mode=EnforcementMode.BLOCK,
        session_ttl_seconds=10.0,
        clock=clock,
    )
    detector.train_request_sequences(
        [_normal_checkout(f"train-{i}") for i in range(8)],
        [_normal_checkout(f"val-{i}") for i in range(2)],
    )

    first = detector.inspect(_request("POST", "/cart/items", "sku=1&quantity=1", "u1"))
    second = detector.inspect(
        _request("POST", "/checkout/payment", "orderId=1&paymentToken=tok", "u1")
    )
    clock.advance(11.0)
    expired = detector.inspect(
        _request("POST", "/checkout/payment", "orderId=1&paymentToken=tok", "u1")
    )

    assert not first.blocked
    assert not second.blocked
    assert expired.blocked
    assert "transition anomaly" in expired.reason


def test_workflow_detector_evicts_oldest_session_when_capacity_is_exceeded() -> None:
    mapper = CanonicalEventMapper(tenant_id="tenant-a", service_id="shop")
    detector = WorkflowBehaviorDetector(
        mapper=mapper,
        mode=EnforcementMode.BLOCK,
        max_sessions=1,
    )
    detector.train_request_sequences(
        [_normal_checkout(f"train-{i}") for i in range(8)],
        [_normal_checkout(f"val-{i}") for i in range(2)],
    )

    first_session_start = detector.inspect(
        _request("POST", "/cart/items", "sku=1&quantity=1", "u1")
    )
    second_session_start = detector.inspect(
        _request("POST", "/cart/items", "sku=2&quantity=1", "u2")
    )
    evicted_session_next = detector.inspect(
        _request("POST", "/checkout/payment", "orderId=1&paymentToken=tok", "u1")
    )

    assert not first_session_start.blocked
    assert not second_session_start.blocked
    assert evicted_session_next.blocked
    assert "transition anomaly" in evicted_session_next.reason
