"""HttpRequest — value object representing an inspected HTTP request.

This is the WAF's unit of inspection. It is intentionally transport-agnostic:
an mitmproxy addon, a FastAPI proxy, or a test fixture can all construct one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


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
        normalized = {k.lower(): v for k, v in dict(self.headers).items()}
        object.__setattr__(self, "headers", MappingProxyType(normalized))
        object.__setattr__(self, "method", self.method.upper())

    @property
    def inspectable_payload(self) -> str:
        """Concatenated surface that detectors scan (path + query + body)."""
        return f"{self.path}?{self.query}\n{self.body}" if self.query else f"{self.path}\n{self.body}"
