"""Hybrid detector combining web-flow Markov transitions with byte-level HMM."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.flow import Flow
from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.markov_flow import (
    START_STATE,
    MarkovAssessment,
    MarkovFlowModel,
    WebFlowStateExtractor,
)
from waf.infrastructure.hmm.hmm_detector import HmmDetector
from waf.infrastructure.hmm.partitioned_hmm import Assessment


@dataclass(frozen=True, slots=True)
class HybridFlowAssessment:
    blocked: bool
    score: float
    markov: MarkovAssessment
    hmm: Assessment
    reason: str


class HybridFlowDetector:
    """Detect anomalous web usage by combining sequence and content models.

    ``train_sequences`` is the primary API for user-flow data: pass normal and
    validation traffic grouped by user/session. ``train`` is kept for the
    existing CalibratableDetector port and treats each input list as one sequence.
    """

    name = "hybrid_markov_hmm"

    def __init__(
        self,
        *,
        markov_model: MarkovFlowModel | None = None,
        hmm_detector: HmmDetector | None = None,
        state_extractor: WebFlowStateExtractor | None = None,
    ) -> None:
        self._markov = markov_model or MarkovFlowModel()
        self._hmm = hmm_detector or HmmDetector()
        self._state_extractor = state_extractor or WebFlowStateExtractor()
        self._session_states: dict[str, str] = {}

    @property
    def is_trained(self) -> bool:
        return self._markov.is_trained and self._hmm.is_trained

    @property
    def markov_model(self) -> MarkovFlowModel:
        return self._markov

    @property
    def hmm_detector(self) -> HmmDetector:
        return self._hmm

    def train(self, normal_flows: Sequence[Flow], validation_flows: Sequence[Flow]) -> None:
        self.train_sequences([normal_flows], [validation_flows])

    def train_sequences(
        self,
        normal_sequences: Sequence[Sequence[Flow]],
        validation_sequences: Sequence[Sequence[Flow]],
    ) -> None:
        self._markov.fit_from_flows(normal_sequences, validation_sequences, self._state_extractor)
        self._hmm.train(_flatten(normal_sequences), _flatten(validation_sequences))
        self._session_states.clear()

    def reset_session(self, session_id: str) -> None:
        self._session_states.pop(session_id, None)

    def assess_flow(self, flow: Flow, *, session_id: str = "__default__") -> HybridFlowAssessment:
        current_state = self._state_extractor.state_from_flow(flow)
        previous_state = self._session_states.get(session_id, START_STATE)

        markov = self._markov.assess_transition(previous_state, current_state)
        hmm = self._hmm.assess_flow(flow)
        self._session_states[session_id] = current_state

        markov_suspicion = _markov_suspicion(markov, self._markov.threshold)
        hmm_suspicion = max(0.0, -hmm.score) if hmm.blocked else 0.0
        blocked = (markov.covered and markov.blocked) or (hmm.covered and hmm.blocked)
        reason = _combine_reason(markov, hmm)
        return HybridFlowAssessment(
            blocked=blocked,
            score=max(markov_suspicion, hmm_suspicion),
            markov=markov,
            hmm=hmm,
            reason=reason,
        )

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        if not self.is_trained:
            return DetectionSignal(self.name, blocked=False, reason="model not trained", score=0.0)

        session_id = _session_id(request)
        flow = self._state_extractor.flow_from_request(request)
        assessment = self.assess_flow(flow, session_id=session_id)
        return DetectionSignal(
            self.name,
            blocked=assessment.blocked,
            reason=assessment.reason,
            score=assessment.score,
        )

    def save(self, path: str | Path) -> None:
        if not self.is_trained:
            raise RuntimeError("cannot save an untrained model")
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "HybridFlowDetector":
        return cast(HybridFlowDetector, pickle.loads(Path(path).read_bytes()))


def _flatten(sequences: Sequence[Sequence[Flow]]) -> list[Flow]:
    return [flow for sequence in sequences for flow in sequence]


def _markov_suspicion(markov: MarkovAssessment, threshold: float | None) -> float:
    if threshold is None or not markov.covered:
        return 0.0
    return max(0.0, threshold - markov.score)


def _combine_reason(markov: MarkovAssessment, hmm: Assessment) -> str:
    parts: list[str] = []
    if markov.covered:
        parts.append(markov.reason)
    if hmm.covered:
        parts.append(hmm.reason)
    if not parts:
        return "no calibrated model covered this flow"
    return " | ".join(parts)


def _session_id(request: HttpRequest) -> str:
    return request.headers.get("x-session-id") or request.client_ip or "__default__"
