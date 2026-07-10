"""Privacy-preserving session identity resolution for workflow state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from http.cookies import SimpleCookie

from waf.domain.model.http_request import HttpRequest


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    value: str
    source: str


class SessionIdentityResolver:
    """Resolve a stable session key without storing raw cookies or bearer tokens."""

    def __init__(
        self,
        *,
        header_names: tuple[str, ...] = ("x-session-id",),
        cookie_names: tuple[str, ...] = ("JSESSIONID", "SESSIONID", "sessionid", "sid"),
        include_user_agent_in_ip_fallback: bool = False,
    ) -> None:
        self._header_names = tuple(name.lower() for name in header_names)
        self._cookie_names = cookie_names
        self._include_user_agent_in_ip_fallback = include_user_agent_in_ip_fallback

    def resolve(self, request: HttpRequest) -> SessionIdentity:
        for header_name in self._header_names:
            value = request.headers.get(header_name, "").strip()
            if value:
                return SessionIdentity(
                    f"header:{header_name}:{_fingerprint(value)}",
                    f"header:{header_name}",
                )

        cookie = SimpleCookie()
        cookie.load(request.headers.get("cookie", ""))
        for cookie_name in self._cookie_names:
            morsel = cookie.get(cookie_name)
            if morsel is not None and morsel.value:
                return SessionIdentity(
                    f"cookie:{cookie_name}:{_fingerprint(morsel.value)}",
                    f"cookie:{cookie_name}",
                )

        if request.client_ip:
            if self._include_user_agent_in_ip_fallback:
                user_agent = request.headers.get("user-agent", "").strip()
                if user_agent:
                    return SessionIdentity(
                        f"ip_ua:{_fingerprint(request.client_ip + chr(10) + user_agent)}",
                        "client_ip_user_agent",
                    )
            return SessionIdentity(f"ip:{request.client_ip}", "client_ip")
        return SessionIdentity("__default__", "default")


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
