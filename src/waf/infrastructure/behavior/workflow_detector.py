"""Workflow transition detector over self-discovered business actions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from waf.domain.model.canonical_event import CanonicalWebEvent
from waf.domain.model.detection import DetectionSignal, SignalAction
from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.canonical_event_mapper import CanonicalEventMapper
from waf.infrastructure.behavior.session_state import Clock, WorkflowSessionStore

START_ACTION = "<START>"


class EnforcementMode(str, Enum):
    SHADOW = "SHADOW"
    ALERT = "ALERT"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class WorkflowAssessment:
    blocked: bool
    score: float
    previous_action_id: str
    current_action_id: str
    reason: str
    covered: bool = True


@dataclass(frozen=True, slots=True)
class WorkflowModelSnapshot:
    ARTIFACT_TYPE: ClassVar[str] = "lf-waap.workflow-transition-model"
    SCHEMA_VERSION: ClassVar[int] = 1

    target_fpr: float
    alpha: float
    transitions: dict[str, dict[str, int]]
    actions: tuple[str, ...]
    threshold: float | None
    tenant_id: str = ""
    service_id: str = ""

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "target_fpr": self.target_fpr,
            "alpha": self.alpha,
            "transitions": {
                previous: {
                    current: count
                    for current, count in sorted(next_actions.items())
                }
                for previous, next_actions in sorted(self.transitions.items())
            },
            "actions": list(self.actions),
            "threshold": self.threshold,
        }

    def to_artifact(self) -> dict[str, object]:
        return {
            "artifact_type": self.ARTIFACT_TYPE,
            "schema_version": self.SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "tenant_id": self.tenant_id,
            "service_id": self.service_id,
            "model": self.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_artifact(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowModelSnapshot":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError("workflow model artifact must be a JSON object")
        return cls.from_artifact(document)

    @classmethod
    def from_dict(cls, document: Mapping[str, object]) -> "WorkflowModelSnapshot":
        transitions = _snapshot_transitions(document.get("transitions"))
        actions = _snapshot_actions(document.get("actions"))
        threshold = document.get("threshold")
        snapshot = cls(
            target_fpr=_snapshot_float(document.get("target_fpr"), "target_fpr"),
            alpha=_snapshot_float(document.get("alpha"), "alpha"),
            transitions=transitions,
            actions=actions,
            threshold=None if threshold is None else _snapshot_float(threshold, "threshold"),
        )
        _validate_snapshot_values(snapshot)
        return snapshot

    @classmethod
    def from_artifact(cls, document: Mapping[str, object]) -> "WorkflowModelSnapshot":
        if document.get("artifact_type") != cls.ARTIFACT_TYPE:
            raise ValueError("workflow model artifact_type is invalid")
        if document.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("workflow model schema_version is unsupported")
        model_document = document.get("model")
        if not isinstance(model_document, Mapping):
            raise ValueError("workflow model artifact model must be an object")
        tenant_id = _artifact_context_value(document, "tenant_id")
        service_id = _artifact_context_value(document, "service_id")
        snapshot = cls.from_dict(model_document)
        snapshot = snapshot.with_context(
            tenant_id=tenant_id,
            service_id=service_id,
        )
        if document.get("fingerprint") != snapshot.fingerprint:
            raise ValueError("workflow model artifact fingerprint mismatch")
        _validate_trained_snapshot(snapshot)
        return snapshot

    def with_context(self, *, tenant_id: str, service_id: str) -> "WorkflowModelSnapshot":
        return WorkflowModelSnapshot(
            target_fpr=self.target_fpr,
            alpha=self.alpha,
            transitions=self.transitions,
            actions=self.actions,
            threshold=self.threshold,
            tenant_id=tenant_id,
            service_id=service_id,
        )


class WorkflowTransitionModel:
    """First-order Markov transition model over canonical business actions."""

    def __init__(self, *, target_fpr: float = 0.01, alpha: float = 0.1) -> None:
        if not 0.0 < target_fpr <= 1.0:
            raise ValueError("target_fpr must be in (0, 1]")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self._target_fpr = target_fpr
        self._alpha = alpha
        self._transitions: dict[str, Counter[str]] = {}
        self._actions: set[str] = set()
        self._threshold: float | None = None

    @property
    def is_trained(self) -> bool:
        return self._threshold is not None

    @property
    def threshold(self) -> float | None:
        return self._threshold

    @property
    def action_count(self) -> int:
        return len(self._actions)

    @property
    def transition_count(self) -> int:
        return sum(len(next_actions) for next_actions in self._transitions.values())

    @property
    def model_fingerprint(self) -> str | None:
        if not self.is_trained:
            return None
        return self.to_snapshot().fingerprint

    def fit(
        self,
        normal_sequences: Sequence[Sequence[str]],
        validation_sequences: Sequence[Sequence[str]],
    ) -> None:
        transitions: defaultdict[str, Counter[str]] = defaultdict(Counter)
        actions: set[str] = set()
        for sequence in normal_sequences:
            previous = START_ACTION
            for current in sequence:
                transitions[previous][current] += 1
                actions.add(current)
                previous = current

        self._transitions = dict(transitions)
        self._actions = actions
        if not actions:
            self._threshold = None
            return

        calibration_scores = list(self._scores(validation_sequences))
        if not calibration_scores:
            calibration_scores = list(self._scores(normal_sequences))
        self._threshold = _quantile(calibration_scores, self._target_fpr)

    def assess_transition(self, previous_action: str, current_action: str) -> WorkflowAssessment:
        if self._threshold is None:
            return WorkflowAssessment(
                False,
                0.0,
                previous_action,
                current_action,
                "workflow model not trained",
                covered=False,
            )
        score = self.score_transition(previous_action, current_action)
        blocked = score < self._threshold
        if blocked:
            reason = (
                "transition anomaly: "
                f"{previous_action!r} -> {current_action!r} "
                f"score {score:.3f} < threshold {self._threshold:.3f}"
            )
        else:
            reason = (
                "transition normal: "
                f"{previous_action!r} -> {current_action!r} "
                f"score {score:.3f} >= threshold {self._threshold:.3f}"
            )
        return WorkflowAssessment(blocked, score, previous_action, current_action, reason)

    def score_transition(self, previous_action: str, current_action: str) -> float:
        vocabulary = max(len(self._actions), 1)
        counts = self._transitions.get(previous_action, Counter())
        total = sum(counts.values())
        probability = (counts[current_action] + self._alpha) / (
            total + self._alpha * vocabulary
        )
        return math.log(probability)

    def to_snapshot(self) -> WorkflowModelSnapshot:
        return WorkflowModelSnapshot(
            target_fpr=self._target_fpr,
            alpha=self._alpha,
            transitions={
                previous: dict(next_actions)
                for previous, next_actions in self._transitions.items()
            },
            actions=tuple(sorted(self._actions)),
            threshold=self._threshold,
        )

    @classmethod
    def from_snapshot(cls, snapshot: WorkflowModelSnapshot) -> "WorkflowTransitionModel":
        model = cls(target_fpr=snapshot.target_fpr, alpha=snapshot.alpha)
        model._transitions = {
            previous: Counter(next_actions)
            for previous, next_actions in snapshot.transitions.items()
        }
        model._actions = set(snapshot.actions)
        model._threshold = snapshot.threshold
        return model

    def _scores(self, sequences: Iterable[Sequence[str]]) -> Iterable[float]:
        for sequence in sequences:
            previous = START_ACTION
            for current in sequence:
                yield self.score_transition(previous, current)
                previous = current


class WorkflowBehaviorDetector:
    """Detector adapter for workflow transition anomalies."""

    name = "workflow"

    def __init__(
        self,
        *,
        mapper: CanonicalEventMapper,
        mode: EnforcementMode = EnforcementMode.SHADOW,
        model: WorkflowTransitionModel | None = None,
        session_ttl_seconds: float = 1800.0,
        max_sessions: int = 100_000,
        clock: Clock | None = None,
    ) -> None:
        self._mapper = mapper
        self._mode = mode
        self._model = model or WorkflowTransitionModel()
        self._sessions = WorkflowSessionStore(
            ttl_seconds=session_ttl_seconds,
            max_sessions=max_sessions,
            clock=clock,
        )

    @property
    def is_trained(self) -> bool:
        return self._model.is_trained

    @property
    def mode(self) -> EnforcementMode:
        return self._mode

    @property
    def threshold(self) -> float | None:
        return self._model.threshold

    @property
    def action_count(self) -> int:
        return self._model.action_count

    @property
    def transition_count(self) -> int:
        return self._model.transition_count

    @property
    def model_fingerprint(self) -> str | None:
        return self._model.model_fingerprint

    @property
    def session_ttl_seconds(self) -> float:
        return self._sessions.ttl_seconds

    @property
    def session_count(self) -> int:
        return self._sessions.size

    @property
    def max_sessions(self) -> int:
        return self._sessions.max_sessions

    def train_request_sequences(
        self,
        normal_sequences: Sequence[Sequence[HttpRequest]],
        validation_sequences: Sequence[Sequence[HttpRequest]],
    ) -> None:
        self.train_event_sequences(
            [_map_actions(self._mapper, sequence) for sequence in normal_sequences],
            [_map_actions(self._mapper, sequence) for sequence in validation_sequences],
        )

    def train_event_sequences(
        self,
        normal_sequences: Sequence[Sequence[CanonicalWebEvent]],
        validation_sequences: Sequence[Sequence[CanonicalWebEvent]],
    ) -> None:
        self._model.fit(
            [[event.action_id for event in sequence] for sequence in normal_sequences],
            [[event.action_id for event in sequence] for sequence in validation_sequences],
        )
        self._sessions.clear()

    def export_model_snapshot(self) -> WorkflowModelSnapshot:
        return self._model.to_snapshot().with_context(
            tenant_id=self._mapper.tenant_id,
            service_id=self._mapper.service_id,
        )

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id)

    def map_request(self, request: HttpRequest) -> CanonicalWebEvent:
        return self._mapper.map(request)

    def assess_event(self, event: CanonicalWebEvent) -> WorkflowAssessment:
        previous = self._sessions.get(event.session_id) or START_ACTION
        assessment = self._model.assess_transition(previous, event.action_id)
        self._sessions.set(event.session_id, event.action_id)
        return assessment

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        event = self._mapper.map(request)
        assessment = self.assess_event(event)
        action = _signal_action(self._mode, assessment.blocked)
        blocks = action is SignalAction.BLOCK
        return DetectionSignal(
            self.name,
            blocked=blocks,
            reason=f"{self._mode.value.lower()}: {assessment.reason}",
            score=max(0.0, -(assessment.score)),
            action=action,
        )


def _map_actions(
    mapper: CanonicalEventMapper, sequence: Sequence[HttpRequest]
) -> list[CanonicalWebEvent]:
    return [mapper.map(request) for request in sequence]


def _signal_action(mode: EnforcementMode, anomalous: bool) -> SignalAction:
    if not anomalous:
        return SignalAction.ALLOW
    if mode is EnforcementMode.BLOCK:
        return SignalAction.BLOCK
    if mode is EnforcementMode.CHALLENGE:
        return SignalAction.CHALLENGE
    if mode is EnforcementMode.ALERT:
        return SignalAction.ALERT
    return SignalAction.ALLOW


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _snapshot_transitions(raw: object) -> dict[str, dict[str, int]]:
    if not isinstance(raw, Mapping):
        raise ValueError("snapshot transitions must be a mapping")
    transitions: dict[str, dict[str, int]] = {}
    for previous, next_actions in raw.items():
        if not isinstance(next_actions, Mapping):
            raise ValueError("snapshot transition rows must be mappings")
        transitions[str(previous)] = {}
        for current, count in next_actions.items():
            transition_count = _snapshot_int(count, "transition count")
            transitions[str(previous)][str(current)] = transition_count
    return transitions


def _snapshot_actions(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple):
        raise ValueError("snapshot actions must be a list")
    return tuple(str(action) for action in raw)


def _snapshot_float(raw: object, name: str) -> float:
    if not isinstance(raw, int | float | str):
        raise ValueError(f"snapshot {name} must be numeric")
    return float(raw)


def _snapshot_int(raw: object, name: str) -> int:
    if not isinstance(raw, int | str):
        raise ValueError(f"snapshot {name} must be an integer")
    return int(raw)


def _validate_snapshot_values(snapshot: WorkflowModelSnapshot) -> None:
    if not 0.0 < snapshot.target_fpr <= 1.0:
        raise ValueError("snapshot target_fpr must be in (0, 1]")
    if snapshot.alpha <= 0.0:
        raise ValueError("snapshot alpha must be positive")
    for previous, next_actions in snapshot.transitions.items():
        if not previous:
            raise ValueError("snapshot transition source must be non-empty")
        for current, count in next_actions.items():
            if not current:
                raise ValueError("snapshot transition target must be non-empty")
            if count < 0:
                raise ValueError("snapshot transition count must be non-negative")


def _validate_trained_snapshot(snapshot: WorkflowModelSnapshot) -> None:
    if snapshot.threshold is None or not snapshot.actions:
        raise ValueError("workflow model artifact must contain a trained model")


def _artifact_context_value(document: Mapping[str, object], name: str) -> str:
    raw_value = str(document.get(name, ""))
    value = raw_value.strip()
    if not value:
        raise ValueError(f"workflow model artifact {name} is required")
    if raw_value != value:
        raise ValueError(
            f"workflow model artifact {name} must not contain surrounding whitespace"
        )
    return value
