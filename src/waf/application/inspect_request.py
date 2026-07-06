"""InspectRequest — application use case.

The thin entry point a transport adapter (mitmproxy addon, FastAPI proxy, CLI)
calls. It owns no business rules; it delegates to the domain service and returns
a Verdict. Keeping this separate means the proxy layer can be added later without
touching the domain.
"""

from __future__ import annotations

from waf.domain.model.http_request import HttpRequest
from waf.domain.model.verdict import Verdict
from waf.domain.service.block_decision_service import BlockDecisionService


class InspectRequest:
    def __init__(self, decision_service: BlockDecisionService) -> None:
        self._decision_service = decision_service

    def __call__(self, request: HttpRequest) -> Verdict:
        return self._decision_service.decide(request)
