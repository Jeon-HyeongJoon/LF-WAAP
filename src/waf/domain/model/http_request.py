"""HttpRequest — value object representing an inspected HTTP request.

This is the WAF's unit of inspection. It is intentionally transport-agnostic:
an mitmproxy addon, a FastAPI proxy, or a test fixture can all construct one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    path: str
    query: str = ""
    headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    body: str = ""
    client_ip: str = ""

    def __post_init__(self) -> None:
        # Normalize headers into an immutable, lowercase-keyed mapping.
        method = _required_text(self.method, "method").upper()
        path = _required_text(self.path, "path")
        query = _optional_text(self.query, "query")
        body = _optional_text(self.body, "body")
        client_ip = _optional_text(self.client_ip, "client_ip")
        normalized = _normalized_headers(self.headers)
        object.__setattr__(self, "headers", MappingProxyType(normalized))
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "client_ip", client_ip)

    @property
    def inspectable_payload(self) -> str:
        """Concatenated surface that detectors scan (path + query + body)."""
        return f"{self.path}?{self.query}\n{self.body}" if self.query else f"{self.path}\n{self.body}"


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"request {name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"request {name} is required")
    if value != normalized:
        raise ValueError(f"request {name} must not contain surrounding whitespace")
    return value


def _optional_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"request {name} must be a string")
    return value


def _normalized_headers(headers: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(headers, Mapping):
        raise ValueError("request headers must be a mapping")
    normalized: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("request headers must contain string keys and values")
        header_name = name.strip().lower()
        if not header_name:
            raise ValueError("request headers must not contain blank names")
        if header_name != name.lower():
            raise ValueError("request headers must not contain surrounding whitespace")
        normalized[header_name] = value
    return normalized
