"""Map HTTP requests into canonical events with self-discovered action IDs."""

from __future__ import annotations

from waf.domain.model.canonical_event import CanonicalWebEvent
from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.action_catalog import ActionCatalog
from waf.infrastructure.behavior.endpoint_signature import EndpointSignatureBuilder
from waf.infrastructure.behavior.route_template import RouteTemplateResolver
from waf.infrastructure.behavior.session_identity import SessionIdentityResolver


class CanonicalEventMapper:
    """Pipeline: request -> route template -> endpoint signature -> action id."""

    def __init__(
        self,
        *,
        tenant_id: str,
        service_id: str,
        route_resolver: RouteTemplateResolver | None = None,
        signature_builder: EndpointSignatureBuilder | None = None,
        action_catalog: ActionCatalog | None = None,
        session_resolver: SessionIdentityResolver | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._service_id = service_id
        self._route_resolver = route_resolver or RouteTemplateResolver()
        self._signature_builder = signature_builder or EndpointSignatureBuilder()
        self._action_catalog = action_catalog or ActionCatalog(service_id=service_id)
        self._session_resolver = session_resolver or SessionIdentityResolver()

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def service_id(self) -> str:
        return self._service_id

    def map(self, request: HttpRequest) -> CanonicalWebEvent:
        route = self._route_resolver.resolve(request)
        signature = self._signature_builder.build(request, route)
        action = self._action_catalog.recognize(signature)
        return CanonicalWebEvent(
            tenant_id=self._tenant_id,
            service_id=self._service_id,
            session_id=self._session_resolver.resolve(request).value,
            method=signature.method,
            route_template=signature.route_template,
            action_id=action.action_id,
            action_confidence=action.confidence,
            param_schema_id=signature.param_schema_id,
            auth_state=signature.auth_state,
        )
