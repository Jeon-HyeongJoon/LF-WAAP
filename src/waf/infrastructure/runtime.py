"""Operational WAF runtime assembly and health surface."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TypeVar
from uuid import uuid4

from waf.domain.detector.detector import Detector
from waf.domain.model.canonical_event import CanonicalWebEvent
from waf.domain.model.detection import DetectionSignal
from waf.domain.model.http_request import HttpRequest
from waf.domain.model.verdict import Decision, Verdict
from waf.domain.service.block_decision_service import BlockDecisionService, BlockPolicy
from waf.infrastructure.behavior.canonical_event_mapper import CanonicalEventMapper
from waf.infrastructure.behavior.workflow_detector import (
    EnforcementMode,
    WorkflowBehaviorDetector,
    WorkflowModelSnapshot,
    WorkflowTransitionModel,
)
from waf.infrastructure.ruleset.ruleset_detector import RuleSetDetector

E = TypeVar("E", bound=Enum)
RUNTIME_DOCUMENT_SCHEMA_VERSION = 1
METRIC_OVERFLOW_KEY = "__other__"


@dataclass(frozen=True, slots=True)
class WafRuntimeConfig:
    tenant_id: str
    service_id: str
    runtime_version: str = "local"
    workflow_mode: EnforcementMode = EnforcementMode.SHADOW
    workflow_model_path: str | Path | None = None
    workflow_session_ttl_seconds: float = 1800.0
    workflow_max_sessions: int = 100_000
    block_policy: BlockPolicy = BlockPolicy.ANY
    ruleset_path: str | Path | None = None
    enable_ruleset: bool = True
    enable_workflow: bool = True
    metrics_max_cardinality: int = 10_000

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "WafRuntimeConfig":
        return cls(
            tenant_id=_env_required(env, "LF_WAAP_TENANT_ID"),
            service_id=_env_required(env, "LF_WAAP_SERVICE_ID"),
            runtime_version=_env_get(env, "LF_WAAP_RUNTIME_VERSION", "local"),
            workflow_mode=_env_enum(
                env,
                "LF_WAAP_WORKFLOW_MODE",
                EnforcementMode,
                EnforcementMode.SHADOW,
            ),
            workflow_model_path=_env_path(env, "LF_WAAP_WORKFLOW_MODEL_PATH"),
            workflow_session_ttl_seconds=_env_float(
                env, "LF_WAAP_WORKFLOW_SESSION_TTL_SECONDS", 1800.0
            ),
            workflow_max_sessions=_env_int(
                env, "LF_WAAP_WORKFLOW_MAX_SESSIONS", 100_000
            ),
            block_policy=_env_enum(
                env,
                "LF_WAAP_BLOCK_POLICY",
                BlockPolicy,
                BlockPolicy.ANY,
            ),
            ruleset_path=_env_path(env, "LF_WAAP_RULESET_PATH"),
            enable_ruleset=_env_bool(env, "LF_WAAP_ENABLE_RULESET", True),
            enable_workflow=_env_bool(env, "LF_WAAP_ENABLE_WORKFLOW", True),
            metrics_max_cardinality=_env_int(
                env, "LF_WAAP_METRICS_MAX_CARDINALITY", 10_000
            ),
        )


@dataclass(frozen=True, slots=True)
class DetectorRuntimeStatus:
    ready: bool
    reason: str
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class WafRuntimeStatus:
    ready: bool
    serving: bool
    detectors: dict[str, DetectorRuntimeStatus]
    serving_detectors: tuple[str, ...]
    degraded_detectors: tuple[str, ...]
    tenant_id: str
    service_id: str
    runtime_version: str
    config_fingerprint: str
    started_at: str
    uptime_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUNTIME_DOCUMENT_SCHEMA_VERSION,
            "ready": self.ready,
            "serving": self.serving,
            "serving_detectors": list(self.serving_detectors),
            "degraded_detectors": list(self.degraded_detectors),
            "tenant_id": self.tenant_id,
            "service_id": self.service_id,
            "runtime_version": self.runtime_version,
            "config_fingerprint": self.config_fingerprint,
            "started_at": self.started_at,
            "uptime_seconds": self.uptime_seconds,
            "detectors": {
                name: status.to_dict() for name, status in self.detectors.items()
            },
        }


@dataclass(frozen=True, slots=True)
class WafRuntimeMetrics:
    tenant_id: str
    service_id: str
    runtime_version: str
    config_fingerprint: str
    started_at: str
    uptime_seconds: float
    metrics_window_started_at: str
    inspected_total: int
    decisions: dict[str, int]
    inspection_duration_total_ms: float
    inspection_duration_avg_ms: float
    inspection_duration_max_ms: float
    signals_total: int
    signals_by_detector: dict[str, int]
    signal_actions_by_detector: dict[str, dict[str, int]]
    requests_by_route_template: dict[str, int]
    decisions_by_route_template: dict[str, dict[str, int]]
    requests_by_business_action: dict[str, int]
    business_action_catalog: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUNTIME_DOCUMENT_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "service_id": self.service_id,
            "runtime_version": self.runtime_version,
            "config_fingerprint": self.config_fingerprint,
            "started_at": self.started_at,
            "uptime_seconds": self.uptime_seconds,
            "metrics_window_started_at": self.metrics_window_started_at,
            "inspected_total": self.inspected_total,
            "decisions": self.decisions,
            "inspection_duration_total_ms": self.inspection_duration_total_ms,
            "inspection_duration_avg_ms": self.inspection_duration_avg_ms,
            "inspection_duration_max_ms": self.inspection_duration_max_ms,
            "signals_total": self.signals_total,
            "signals_by_detector": self.signals_by_detector,
            "signal_actions_by_detector": self.signal_actions_by_detector,
            "requests_by_route_template": self.requests_by_route_template,
            "decisions_by_route_template": self.decisions_by_route_template,
            "requests_by_business_action": self.requests_by_business_action,
            "business_action_catalog": self.business_action_catalog,
        }


@dataclass(frozen=True, slots=True)
class WafAuditRecord:
    verdict: Verdict
    event: CanonicalWebEvent | None
    tenant_id: str = ""
    service_id: str = ""
    correlation_id: str = ""
    client_ip: str = ""
    runtime_version: str = ""
    config_fingerprint: str = ""
    detector_fingerprints: dict[str, str] | None = None
    inspected_at: str = ""
    inspection_duration_ms: float = 0.0

    def with_correlation_id(self, correlation_id: str) -> "WafAuditRecord":
        return WafAuditRecord(
            verdict=self.verdict,
            event=self.event,
            tenant_id=self.tenant_id,
            service_id=self.service_id,
            correlation_id=correlation_id,
            client_ip=self.client_ip,
            runtime_version=self.runtime_version,
            config_fingerprint=self.config_fingerprint,
            detector_fingerprints=self.detector_fingerprints,
            inspected_at=self.inspected_at,
            inspection_duration_ms=self.inspection_duration_ms,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUNTIME_DOCUMENT_SCHEMA_VERSION,
            "correlation_id": self.correlation_id,
            "tenant_id": self.tenant_id,
            "service_id": self.service_id,
            "client_ip": self.client_ip,
            "inspected_at": self.inspected_at,
            "inspection_duration_ms": self.inspection_duration_ms,
            "runtime_version": self.runtime_version,
            "config_fingerprint": self.config_fingerprint,
            "detector_fingerprints": self.detector_fingerprints or {},
            "decision": self.verdict.decision.value,
            "blocked": self.verdict.is_blocked,
            "reason": self.verdict.reason,
            "event": _event_to_dict(self.event),
            "signals": [_signal_to_dict(signal) for signal in self.verdict.signals],
        }


@dataclass(frozen=True, slots=True)
class WafInspectionOutcome:
    verdict: Verdict
    audit: WafAuditRecord


class OperationalWaf:
    """Assembled WAF runtime used by adapters and smoke tests."""

    def __init__(
        self,
        *,
        decider: BlockDecisionService,
        workflow_detector: WorkflowBehaviorDetector | None,
        audit_event_mapper: CanonicalEventMapper,
        detectors: Sequence[Detector],
        tenant_id: str,
        service_id: str,
        runtime_version: str,
        config_fingerprint: str,
        metrics_max_cardinality: int,
    ) -> None:
        self._decider = decider
        self._workflow_detector = workflow_detector
        self._audit_event_mapper = audit_event_mapper
        self._detectors = tuple(detectors)
        self._tenant_id = tenant_id
        self._service_id = service_id
        self._runtime_version = runtime_version
        self._config_fingerprint = config_fingerprint
        self._metrics_max_cardinality = metrics_max_cardinality
        self._started_at = _utc_now_iso()
        self._metrics_window_started_at = self._started_at
        self._started_monotonic = time.monotonic()
        self._metrics_lock = Lock()
        self._inspected_total = 0
        self._decision_counts: Counter[str] = Counter()
        self._inspection_duration_total_ms = 0.0
        self._inspection_duration_max_ms = 0.0
        self._signals_total = 0
        self._signal_counts_by_detector: Counter[str] = Counter()
        self._signal_action_counts_by_detector: dict[str, Counter[str]] = {}
        self._request_counts_by_route_template: Counter[str] = Counter()
        self._decision_counts_by_route_template: dict[str, Counter[str]] = {}
        self._request_counts_by_business_action: Counter[str] = Counter()
        self._business_action_catalog: dict[str, dict[str, object]] = {}

    @classmethod
    def from_config(
        cls,
        config: WafRuntimeConfig,
        *,
        extra_detectors: Sequence[Detector] = (),
        workflow_model_snapshot: WorkflowModelSnapshot | None = None,
    ) -> "OperationalWaf":
        _validate_config(config)
        if config.workflow_model_path is not None and workflow_model_snapshot is not None:
            raise ValueError(
                "workflow_model_path and workflow_model_snapshot cannot both be provided"
            )
        if (
            not config.enable_ruleset
            and config.enable_workflow
            and config.workflow_model_path is None
            and workflow_model_snapshot is None
        ):
            raise ValueError(
                "workflow_model_path or workflow_model_snapshot is required "
                "when ruleset is disabled"
            )
        effective_workflow_snapshot = workflow_model_snapshot
        if config.workflow_model_path is not None:
            effective_workflow_snapshot = WorkflowModelSnapshot.load(
                config.workflow_model_path
            )
        if effective_workflow_snapshot is not None:
            _validate_workflow_snapshot_context(
                effective_workflow_snapshot,
                tenant_id=config.tenant_id,
                service_id=config.service_id,
            )
        detectors: list[Detector] = []
        workflow_detector: WorkflowBehaviorDetector | None = None
        audit_event_mapper = CanonicalEventMapper(
            tenant_id=config.tenant_id,
            service_id=config.service_id,
        )
        if config.enable_ruleset:
            detectors.append(
                RuleSetDetector.from_yaml(config.ruleset_path)
                if config.ruleset_path is not None
                else RuleSetDetector.default()
            )
        if config.enable_workflow:
            workflow_detector = WorkflowBehaviorDetector(
                mapper=audit_event_mapper,
                mode=config.workflow_mode,
                model=(
                    WorkflowTransitionModel.from_snapshot(effective_workflow_snapshot)
                    if effective_workflow_snapshot is not None
                    else None
                ),
                session_ttl_seconds=config.workflow_session_ttl_seconds,
                max_sessions=config.workflow_max_sessions,
            )
            detectors.append(workflow_detector)
        detectors.extend(extra_detectors)
        _validate_detector_names(detectors)
        if not detectors:
            raise ValueError("at least one detector must be enabled")
        return cls(
            decider=BlockDecisionService(detectors, policy=config.block_policy),
            workflow_detector=workflow_detector,
            audit_event_mapper=audit_event_mapper,
            detectors=detectors,
            tenant_id=config.tenant_id,
            service_id=config.service_id,
            runtime_version=config.runtime_version,
            config_fingerprint=_config_fingerprint(config),
            metrics_max_cardinality=config.metrics_max_cardinality,
        )

    @property
    def workflow_detector(self) -> WorkflowBehaviorDetector | None:
        return self._workflow_detector

    def inspect(self, request: HttpRequest) -> Verdict:
        started = time.perf_counter()
        verdict = self._decider.decide(request)
        event = self._audit_event_mapper.map(request)
        duration_ms = (time.perf_counter() - started) * 1000.0
        self._record_verdict(
            verdict, inspection_duration_ms=duration_ms, event=event
        )
        return verdict

    def inspect_with_audit(self, request: HttpRequest) -> WafInspectionOutcome:
        started = time.perf_counter()
        verdict = self._decider.decide(request)
        event = self._audit_event_mapper.map(request)
        duration_ms = (time.perf_counter() - started) * 1000.0
        self._record_verdict(
            verdict, inspection_duration_ms=duration_ms, event=event
        )
        return WafInspectionOutcome(
            verdict=verdict,
            audit=WafAuditRecord(
                verdict=verdict,
                event=event,
                tenant_id=self._tenant_id,
                service_id=self._service_id,
                correlation_id=_request_correlation_id(request),
                client_ip=request.client_ip,
                runtime_version=self._runtime_version,
                config_fingerprint=self._config_fingerprint,
                detector_fingerprints=_detector_fingerprints(self._detectors),
                inspected_at=_utc_now_iso(),
                inspection_duration_ms=duration_ms,
            ),
        )

    def metrics(self) -> WafRuntimeMetrics:
        with self._metrics_lock:
            return self._metrics_snapshot_locked()

    def reset_metrics(self) -> WafRuntimeMetrics:
        with self._metrics_lock:
            snapshot = self._metrics_snapshot_locked()
            self._reset_metrics_locked()
            return snapshot

    def _metrics_snapshot_locked(self) -> WafRuntimeMetrics:
        inspected_total = self._inspected_total
        decisions = {
            decision.value: self._decision_counts[decision.value]
            for decision in Decision
        }
        duration_total_ms = self._inspection_duration_total_ms
        duration_avg_ms = duration_total_ms / inspected_total if inspected_total else 0.0
        duration_max_ms = self._inspection_duration_max_ms
        signals_total = self._signals_total
        signals_by_detector = dict(self._signal_counts_by_detector)
        signal_actions_by_detector = {
            detector: {
                decision.value: action_counts[decision.value]
                for decision in Decision
            }
            for detector, action_counts in self._signal_action_counts_by_detector.items()
        }
        requests_by_route_template = dict(self._request_counts_by_route_template)
        decisions_by_route_template = {
            route_template: {
                decision.value: decision_counts[decision.value]
                for decision in Decision
            }
            for route_template, decision_counts in self._decision_counts_by_route_template.items()
        }
        requests_by_business_action = dict(self._request_counts_by_business_action)
        business_action_catalog = {
            action_id: dict(metadata)
            for action_id, metadata in self._business_action_catalog.items()
        }
        return WafRuntimeMetrics(
            tenant_id=self._tenant_id,
            service_id=self._service_id,
            runtime_version=self._runtime_version,
            config_fingerprint=self._config_fingerprint,
            started_at=self._started_at,
            uptime_seconds=self._uptime_seconds(),
            metrics_window_started_at=self._metrics_window_started_at,
            inspected_total=inspected_total,
            decisions=decisions,
            inspection_duration_total_ms=duration_total_ms,
            inspection_duration_avg_ms=duration_avg_ms,
            inspection_duration_max_ms=duration_max_ms,
            signals_total=signals_total,
            signals_by_detector=signals_by_detector,
            signal_actions_by_detector=signal_actions_by_detector,
            requests_by_route_template=requests_by_route_template,
            decisions_by_route_template=decisions_by_route_template,
            requests_by_business_action=requests_by_business_action,
            business_action_catalog=business_action_catalog,
        )

    def _reset_metrics_locked(self) -> None:
        self._inspected_total = 0
        self._decision_counts.clear()
        self._inspection_duration_total_ms = 0.0
        self._inspection_duration_max_ms = 0.0
        self._signals_total = 0
        self._signal_counts_by_detector.clear()
        self._signal_action_counts_by_detector.clear()
        self._request_counts_by_route_template.clear()
        self._decision_counts_by_route_template.clear()
        self._request_counts_by_business_action.clear()
        self._business_action_catalog.clear()
        self._metrics_window_started_at = _utc_now_iso()

    def status(self) -> WafRuntimeStatus:
        detectors = {detector.name: _detector_status(detector) for detector in self._detectors}
        serving_detectors = tuple(
            name for name, status in detectors.items() if status.ready
        )
        degraded_detectors = tuple(
            name for name, status in detectors.items() if not status.ready
        )
        return WafRuntimeStatus(
            ready=all(status.ready for status in detectors.values()),
            serving=bool(serving_detectors),
            detectors=detectors,
            serving_detectors=serving_detectors,
            degraded_detectors=degraded_detectors,
            tenant_id=self._tenant_id,
            service_id=self._service_id,
            runtime_version=self._runtime_version,
            config_fingerprint=self._config_fingerprint,
            started_at=self._started_at,
            uptime_seconds=self._uptime_seconds(),
        )

    def _record_verdict(
        self,
        verdict: Verdict,
        *,
        inspection_duration_ms: float,
        event: CanonicalWebEvent,
    ) -> None:
        with self._metrics_lock:
            self._inspected_total += 1
            self._decision_counts[verdict.decision.value] += 1
            self._inspection_duration_total_ms += inspection_duration_ms
            self._inspection_duration_max_ms = max(
                self._inspection_duration_max_ms, inspection_duration_ms
            )
            self._signals_total += len(verdict.signals)
            for signal in verdict.signals:
                self._signal_counts_by_detector[signal.detector] += 1
                action_counts = self._signal_action_counts_by_detector.setdefault(
                    signal.detector, Counter()
                )
                action_counts[signal.action.value] += 1
            route_key = _bounded_metric_key(
                self._request_counts_by_route_template,
                _route_metric_key(event),
                self._metrics_max_cardinality,
            )
            self._request_counts_by_route_template[route_key] += 1
            decision_counts = self._decision_counts_by_route_template.setdefault(
                route_key, Counter()
            )
            decision_counts[verdict.decision.value] += 1
            action_key = _bounded_metric_key(
                self._request_counts_by_business_action,
                event.action_id,
                self._metrics_max_cardinality,
            )
            self._request_counts_by_business_action[action_key] += 1
            if action_key == event.action_id:
                self._business_action_catalog.setdefault(
                    action_key, _business_action_metadata(event)
                )

    def _uptime_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._started_monotonic)


def _detector_status(detector: Detector) -> DetectorRuntimeStatus:
    metadata = _detector_metadata(detector)
    is_trained = getattr(detector, "is_trained", None)
    if isinstance(is_trained, bool):
        if is_trained:
            return DetectorRuntimeStatus(True, "ready", metadata)
        return DetectorRuntimeStatus(False, "not trained", metadata)
    return DetectorRuntimeStatus(True, "ready", metadata)


def _validate_detector_names(detectors: Sequence[Detector]) -> None:
    seen: set[str] = set()
    for detector in detectors:
        raw_name = detector.name
        name = raw_name.strip()
        if not name:
            raise ValueError("detector name is required")
        if raw_name != name:
            raise ValueError(f"detector name must not contain surrounding whitespace: {raw_name!r}")
        if name in seen:
            raise ValueError(f"duplicate detector name: {name}")
        seen.add(name)


def _request_correlation_id(request: HttpRequest) -> str:
    return str(request.headers.get("x-request-id", "")).strip() or uuid4().hex


def _route_metric_key(event: CanonicalWebEvent) -> str:
    return f"{event.method} {event.route_template}"


def _bounded_metric_key(counter: Counter[str], key: str, max_cardinality: int) -> str:
    if key in counter or len(counter) < max_cardinality:
        return key
    return METRIC_OVERFLOW_KEY


def _env_required(env: Mapping[str, str], name: str) -> str:
    value = _env_get(env, name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _env_get(env: Mapping[str, str], name: str, default: str) -> str:
    return env.get(name, default).strip()


def _env_path(env: Mapping[str, str], name: str) -> Path | None:
    value = _env_get(env, name, "")
    if not value:
        return None
    return Path(value)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = _env_get(env, name, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = _env_get(env, name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = _env_get(env, name, "")
    if not value:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_enum(
    env: Mapping[str, str],
    name: str,
    enum_type: type[E],
    default: E,
) -> E:
    value = _env_get(env, name, "")
    if not value:
        return default
    normalized = value.upper()
    try:
        return enum_type[normalized]
    except KeyError as exc:
        raise ValueError(f"{name} is invalid: {value!r}") from exc


def _validate_config(config: WafRuntimeConfig) -> None:
    _validate_config_identity("runtime_version", config.runtime_version)
    _validate_config_identity("tenant_id", config.tenant_id)
    _validate_config_identity("service_id", config.service_id)
    _validate_positive_number(
        "workflow_session_ttl_seconds", config.workflow_session_ttl_seconds
    )
    _validate_positive_integer("workflow_max_sessions", config.workflow_max_sessions)
    _validate_positive_integer("metrics_max_cardinality", config.metrics_max_cardinality)
    _validate_config_bool("enable_ruleset", config.enable_ruleset)
    _validate_config_bool("enable_workflow", config.enable_workflow)
    _validate_config_enum("workflow_mode", config.workflow_mode, EnforcementMode)
    _validate_config_enum("block_policy", config.block_policy, BlockPolicy)
    if config.workflow_model_path is not None and not config.enable_workflow:
        raise ValueError("workflow_model_path requires enable_workflow")
    if config.ruleset_path is not None and not config.enable_ruleset:
        raise ValueError("ruleset_path requires enable_ruleset")
    if config.workflow_model_path is not None:
        _validate_artifact_file("workflow_model_path", config.workflow_model_path)
    if config.ruleset_path is not None:
        _validate_artifact_file("ruleset_path", config.ruleset_path)
    if not config.enable_ruleset and not config.enable_workflow:
        raise ValueError("at least one detector must be enabled")


def _validate_config_identity(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} is required")
    if value != normalized:
        raise ValueError(f"{name} must not contain surrounding whitespace")


def _validate_positive_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_config_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _validate_config_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{name} must be a {enum_type.__name__}")


def _validate_artifact_file(name: str, path: object) -> None:
    if not isinstance(path, (str, Path)):
        raise ValueError(f"{name} must be a filesystem path")
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ValueError(f"{name} must point to a readable file: {artifact_path}")


def _config_fingerprint(config: WafRuntimeConfig) -> str:
    document = {
        "tenant_id": config.tenant_id,
        "service_id": config.service_id,
        "runtime_version": config.runtime_version,
        "workflow_mode": config.workflow_mode.value,
        "workflow_model_path": (
            str(config.workflow_model_path) if config.workflow_model_path is not None else None
        ),
        "workflow_session_ttl_seconds": config.workflow_session_ttl_seconds,
        "workflow_max_sessions": config.workflow_max_sessions,
        "block_policy": config.block_policy.name,
        "ruleset_path": str(config.ruleset_path) if config.ruleset_path is not None else None,
        "enable_ruleset": config.enable_ruleset,
        "enable_workflow": config.enable_workflow,
        "metrics_max_cardinality": config.metrics_max_cardinality,
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _validate_workflow_snapshot_context(
    snapshot: WorkflowModelSnapshot, *, tenant_id: str, service_id: str
) -> None:
    if snapshot.tenant_id and snapshot.tenant_id != tenant_id:
        raise ValueError(
            "workflow_model_snapshot tenant_id mismatch: "
            f"runtime tenant_id={tenant_id!r}, artifact tenant_id={snapshot.tenant_id!r}"
        )
    if snapshot.service_id and snapshot.service_id != service_id:
        raise ValueError(
            "workflow_model_snapshot service_id mismatch: "
            f"runtime service_id={service_id!r}, artifact service_id={snapshot.service_id!r}"
        )
    service_ids = _auto_action_service_ids(snapshot.actions)
    if service_ids and service_id not in service_ids:
        expected = ", ".join(sorted(service_ids))
        raise ValueError(
            "workflow_model_snapshot service_id mismatch: "
            f"runtime service_id={service_id!r}, artifact service_id={expected!r}"
        )


def _auto_action_service_ids(action_ids: Sequence[str]) -> set[str]:
    service_ids: set[str] = set()
    for action_id in action_ids:
        parts = action_id.split(":")
        if len(parts) >= 3 and parts[0] == "auto" and parts[2].startswith("act_"):
            service_ids.add(parts[1])
    return service_ids


def _detector_metadata(detector: Detector) -> dict[str, object]:
    metadata: dict[str, object] = {}
    mode = getattr(detector, "mode", None)
    if mode is not None:
        metadata["mode"] = getattr(mode, "value", str(mode))
    for name in (
        "threshold",
        "action_count",
        "transition_count",
        "train_count",
        "validation_count",
        "num_partitions",
        "extractor_name",
        "session_ttl_seconds",
        "max_sessions",
        "session_count",
        "model_fingerprint",
        "rule_count",
    ):
        value = getattr(detector, name, None)
        if value is not None and not callable(value):
            metadata[name] = value
    return metadata


def _detector_fingerprints(detectors: Sequence[Detector]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for detector in detectors:
        value = getattr(detector, "model_fingerprint", None)
        if isinstance(value, str) and value:
            fingerprints[detector.name] = value
    return fingerprints


def _event_to_dict(event: CanonicalWebEvent | None) -> dict[str, object] | None:
    if event is None:
        return None
    return {
        "tenant_id": event.tenant_id,
        "service_id": event.service_id,
        "session_id": event.session_id,
        "method": event.method,
        "route_template": event.route_template,
        "action_id": event.action_id,
        "action_confidence": event.action_confidence,
        "param_schema_id": event.param_schema_id,
        "auth_state": event.auth_state,
    }


def _business_action_metadata(event: CanonicalWebEvent) -> dict[str, object]:
    return {
        "action_id": event.action_id,
        "method": event.method,
        "route_template": event.route_template,
        "param_schema_id": event.param_schema_id,
        "auth_state": event.auth_state,
        "action_confidence": event.action_confidence,
    }


def _signal_to_dict(signal: DetectionSignal) -> dict[str, object]:
    return {
        "detector": signal.detector,
        "action": signal.action.value,
        "blocked": signal.blocked,
        "reason": signal.reason,
        "score": signal.score,
    }


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
