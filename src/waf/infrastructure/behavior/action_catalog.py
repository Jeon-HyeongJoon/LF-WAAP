"""Self-discovered action catalog for services without explicit action IDs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from waf.infrastructure.behavior.endpoint_signature import EndpointSignature


@dataclass(frozen=True, slots=True)
class RecognizedAction:
    action_id: str
    signature: EndpointSignature
    confidence: float
    reason: str


class ActionCatalog:
    """Assign stable internal action IDs to endpoint signatures."""

    def __init__(self, service_id: str) -> None:
        if not service_id:
            raise ValueError("service_id is required")
        self._service_id = service_id
        self._by_signature: dict[str, RecognizedAction] = {}

    def recognize(self, signature: EndpointSignature) -> RecognizedAction:
        existing = self._by_signature.get(signature.key)
        if existing is not None:
            return existing

        action = RecognizedAction(
            action_id=self._action_id(signature),
            signature=signature,
            confidence=1.0,
            reason="exact endpoint signature match",
        )
        self._by_signature[signature.key] = action
        return action

    def _action_id(self, signature: EndpointSignature) -> str:
        digest = hashlib.sha256(signature.key.encode("utf-8")).hexdigest()[:12]
        return f"auto:{self._service_id}:act_{digest}"
