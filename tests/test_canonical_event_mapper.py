from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior import CanonicalEventMapper
from waf.infrastructure.behavior.session_identity import SessionIdentityResolver


def test_canonical_event_mapper_assigns_stable_internal_action_id() -> None:
    mapper = CanonicalEventMapper(tenant_id="tenant-a", service_id="shop")
    request = HttpRequest(
        "POST",
        "/orders/12345/confirm",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body="orderId=12345&paymentToken=tok",
        client_ip="10.0.0.7",
    )

    first = mapper.map(request)
    second = mapper.map(request)

    assert first.tenant_id == "tenant-a"
    assert first.service_id == "shop"
    assert first.session_id == "ip:10.0.0.7"
    assert first.route_template == "/orders/{num}/confirm"
    assert first.param_schema_id == "body:orderId,paymentToken"
    assert first.action_id.startswith("auto:shop:act_")
    assert second.action_id == first.action_id


def test_canonical_event_mapper_separates_same_route_by_schema() -> None:
    mapper = CanonicalEventMapper(tenant_id="tenant-a", service_id="shop")

    payment = mapper.map(
        HttpRequest(
            "POST",
            "/submit",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body="orderId=12345&paymentToken=tok",
        )
    )
    profile = mapper.map(
        HttpRequest(
            "POST",
            "/submit",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body="displayName=hope",
        )
    )

    assert payment.action_id != profile.action_id


def test_canonical_event_mapper_uses_hashed_cookie_session_before_client_ip() -> None:
    mapper = CanonicalEventMapper(
        tenant_id="tenant-a",
        service_id="shop",
        session_resolver=SessionIdentityResolver(cookie_names=("JSESSIONID",)),
    )
    first = mapper.map(
        HttpRequest(
            "GET",
            "/account",
            headers={"Cookie": "JSESSIONID=secret-session-a"},
            client_ip="203.0.113.10",
        )
    )
    second = mapper.map(
        HttpRequest(
            "GET",
            "/account",
            headers={"Cookie": "JSESSIONID=secret-session-b"},
            client_ip="203.0.113.10",
        )
    )

    assert first.session_id != second.session_id
    assert first.session_id.startswith("cookie:JSESSIONID:")
    assert "secret-session-a" not in first.session_id


def test_session_identity_can_include_user_agent_in_ip_fallback() -> None:
    resolver = SessionIdentityResolver(include_user_agent_in_ip_fallback=True)
    first = resolver.resolve(
        HttpRequest(
            "GET",
            "/",
            headers={"User-Agent": "browser-a"},
            client_ip="203.0.113.10",
        )
    )
    second = resolver.resolve(
        HttpRequest(
            "GET",
            "/",
            headers={"User-Agent": "browser-b"},
            client_ip="203.0.113.10",
        )
    )

    assert first.value != second.value
    assert first.value.startswith("ip_ua:")
    assert first.source == "client_ip_user_agent"
    assert "203.0.113.10" not in first.value
    assert "browser-a" not in first.value
