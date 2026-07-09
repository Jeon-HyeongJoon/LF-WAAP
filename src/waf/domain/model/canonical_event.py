"""Canonical web event used by workflow-oriented behavior models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CanonicalWebEvent:
    tenant_id: str
    service_id: str
    session_id: str
    method: str
    route_template: str
    action_id: str
    action_confidence: float
    param_schema_id: str
    auth_state: str

    @property
    def transition_state(self) -> str:
        return self.action_id
