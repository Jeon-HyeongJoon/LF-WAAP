"""Endpoint signatures used to recognize inferred business actions."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl

from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.route_template import RouteTemplate


@dataclass(frozen=True, slots=True)
class EndpointSignature:
    method: str
    route_template: str
    query_keys: tuple[str, ...] = ()
    body_keys: tuple[str, ...] = ()
    content_type: str = ""
    auth_state: str = "anonymous"

    @property
    def param_schema_id(self) -> str:
        parts: list[str] = []
        if self.body_keys:
            parts.append(f"body:{','.join(self.body_keys)}")
        if self.query_keys:
            parts.append(f"query:{','.join(self.query_keys)}")
        return "|".join(parts) if parts else "none"

    @property
    def key(self) -> str:
        return "|".join(
            (
                self.method,
                self.route_template,
                self.param_schema_id,
                self.content_type,
                self.auth_state,
            )
        )


class EndpointSignatureBuilder:
    """Build a stable action-recognition signature without raw parameter values."""

    def build(self, request: HttpRequest, route: RouteTemplate) -> EndpointSignature:
        content_type = _base_content_type(request.headers.get("content-type", ""))
        return EndpointSignature(
            method=route.method,
            route_template=route.template,
            query_keys=_keys_from_urlencoded(request.query),
            body_keys=_body_keys(request.body, content_type),
            content_type=content_type,
            auth_state=_auth_state(request),
        )


def _base_content_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _keys_from_urlencoded(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(sorted({key for key, _ in parse_qsl(value, keep_blank_values=True)}))


def _body_keys(body: str, content_type: str) -> tuple[str, ...]:
    if content_type != "application/x-www-form-urlencoded":
        return ()
    return _keys_from_urlencoded(body)


def _auth_state(request: HttpRequest) -> str:
    if request.headers.get("authorization") or "jsessionid=" in request.headers.get("cookie", "").lower():
        return "authenticated"
    return "anonymous"
