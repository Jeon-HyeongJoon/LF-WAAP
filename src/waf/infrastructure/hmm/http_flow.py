"""Adapter: HttpRequest (WAF L7 unit) -> Flow (HMM observation unit).

Serializes the request to bytes the way it would appear on the wire so the
byte-level model sees a realistic payload, and derives the partition metadata
(http protocol, destination port, inbound direction).
"""

from __future__ import annotations

from waf.domain.model.flow import Direction, Flow
from waf.domain.model.http_request import HttpRequest

_DEFAULT_HTTP_PORT = 80


def _port_of(request: HttpRequest) -> int:
    host = request.headers.get("host", "")
    if ":" in host:
        try:
            return int(host.rsplit(":", 1)[1])
        except ValueError:
            return _DEFAULT_HTTP_PORT
    return _DEFAULT_HTTP_PORT


def http_request_to_flow(request: HttpRequest) -> Flow:
    target = f"{request.path}?{request.query}" if request.query else request.path
    head = f"{request.method} {target} HTTP/1.1\r\n"
    headers = "".join(f"{k}: {v}\r\n" for k, v in request.headers.items())
    raw = f"{head}{headers}\r\n{request.body}"
    return Flow(
        protocol="http",
        port=_port_of(request),
        direction=Direction.INBOUND,
        payload=raw.encode("utf-8", errors="surrogateescape"),
    )
