from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.session_identity import SessionIdentityResolver


def test_session_identity_resolver_hashes_explicit_session_header() -> None:
    resolver = SessionIdentityResolver()

    identity = resolver.resolve(
        HttpRequest("GET", "/", headers={"X-Session-Id": "raw-session-token"})
    )

    assert identity.source == "header:x-session-id"
    assert identity.value.startswith("header:x-session-id:")
    assert "raw-session-token" not in identity.value


def test_session_identity_resolver_falls_back_to_client_ip() -> None:
    resolver = SessionIdentityResolver()

    identity = resolver.resolve(HttpRequest("GET", "/", client_ip="198.51.100.8"))

    assert identity.source == "client_ip"
    assert identity.value == "ip:198.51.100.8"
