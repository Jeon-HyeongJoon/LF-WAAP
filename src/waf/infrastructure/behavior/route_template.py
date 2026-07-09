"""Route template discovery for workflow-oriented behavior models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote

from waf.domain.model.http_request import HttpRequest

_NUMERIC_SEGMENT = re.compile(r"^\d+$")
_UUID_SEGMENT = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_HEX_TOKEN_SEGMENT = re.compile(r"^[0-9a-fA-F]{8,}$")


@dataclass(frozen=True, slots=True)
class RouteTemplate:
    method: str
    template: str

    @property
    def key(self) -> str:
        return f"{self.method} {self.template}"


class RouteTemplateResolver:
    """Infer a low-cardinality route template from an HTTP request path."""

    def resolve(self, request: HttpRequest) -> RouteTemplate:
        return RouteTemplate(request.method, self._normalize_path(request.path))

    def _normalize_path(self, path: str) -> str:
        parts: list[str] = []
        for raw_segment in path.split("/"):
            if not raw_segment:
                continue
            segment = unquote(raw_segment)
            parts.append(self._normalize_segment(segment))
        return "/" + "/".join(parts) if parts else "/"

    @staticmethod
    def _normalize_segment(segment: str) -> str:
        if _NUMERIC_SEGMENT.match(segment):
            return "{num}"
        if _UUID_SEGMENT.match(segment) or _HEX_TOKEN_SEGMENT.match(segment):
            return "{id}"
        return segment.lower()
