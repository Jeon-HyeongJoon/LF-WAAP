"""Markov model for user web-service flow transitions.

The HMM detector models request/packet contents. This module models a different
signal: which web-service state a user visits next. It accepts raw ``Flow``
objects so packet datasets can be grouped into per-user/session sequences before
training.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

from waf.domain.model.flow import Flow
from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.hmm.http_flow import http_request_to_flow
from waf.infrastructure.hmm.partition import partition_key

START_STATE = "<START>"
_HTTP_LINE = re.compile(rb"^([A-Z]+)\s+(\S+)\s+HTTP/\d(?:\.\d)?")
_NUMERIC_SEGMENT = re.compile(r"^\d+$")
_TOKEN_SEGMENT = re.compile(r"^[0-9a-fA-F]{8,}$")


@dataclass(frozen=True, slots=True)
class MarkovAssessment:
    blocked: bool
    score: float
    previous_state: str
    current_state: str
    reason: str
    covered: bool = True


class WebFlowStateExtractor:
    """Convert an HTTP request or network flow into a stable web-flow state."""

    def __init__(self, *, include_query_keys: bool = True) -> None:
        self._include_query_keys = include_query_keys

    def state_from_request(self, request: HttpRequest) -> str:
        target = f"{request.path}?{request.query}" if request.query else request.path
        return self._http_state(request.method, target)

    def state_from_flow(self, flow: Flow) -> str:
        parsed = self._parse_http_line(flow.payload)
        if parsed is not None:
            method, target = parsed
            return self._http_state(method, target)
        return f"flow {partition_key(flow)}"

    def flow_from_request(self, request: HttpRequest) -> Flow:
        return http_request_to_flow(request)

    def _http_state(self, method: str, target: str) -> str:
        split = urlsplit(target)
        path = self._normalize_path(split.path or "/")
        query_keys = self._query_keys(split.query)
        if query_keys:
            return f"{method.upper()} {path}?{query_keys}"
        return f"{method.upper()} {path}"

    def _query_keys(self, query: str) -> str:
        if not self._include_query_keys or not query:
            return ""
        keys = sorted({key for key, _ in parse_qsl(query, keep_blank_values=True)})
        return ",".join(keys)

    @staticmethod
    def _normalize_path(path: str) -> str:
        parts: list[str] = []
        for segment in path.split("/"):
            if not segment:
                continue
            if _NUMERIC_SEGMENT.match(segment):
                parts.append("{num}")
            elif _TOKEN_SEGMENT.match(segment):
                parts.append("{id}")
            else:
                parts.append(segment.lower())
        return "/" + "/".join(parts) if parts else "/"

    @staticmethod
    def _parse_http_line(payload: bytes) -> tuple[str, str] | None:
        first_line = payload.splitlines()[0] if payload else b""
        match = _HTTP_LINE.match(first_line)
        if match is None:
            return None
        method = match.group(1).decode("ascii", errors="ignore")
        target = match.group(2).decode("utf-8", errors="surrogateescape")
        return method, target


class MarkovFlowModel:
    """Context-aware Markov chain calibrated on held-out normal flow sequences."""

    def __init__(
        self,
        *,
        target_fpr: float = 0.01,
        alpha: float = 0.1,
        transition_weight: float = 0.8,
    ) -> None:
        if not 0.0 < target_fpr <= 1.0:
            raise ValueError("target_fpr must be in (0, 1]")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= transition_weight <= 1.0:
            raise ValueError("transition_weight must be in [0, 1]")
        self._target_fpr = target_fpr
        self._alpha = alpha
        self._transition_weight = transition_weight
        self._transitions: dict[str, Counter[str]] = {}
        self._target_counts: Counter[str] = Counter()
        self._states: set[str] = set()
        self._threshold: float | None = None
        self._train_count = 0
        self._validation_count = 0

    @property
    def is_trained(self) -> bool:
        return self._threshold is not None

    @property
    def threshold(self) -> float | None:
        return self._threshold

    @property
    def train_count(self) -> int:
        return self._train_count

    @property
    def validation_count(self) -> int:
        return self._validation_count

    @property
    def states(self) -> tuple[str, ...]:
        return tuple(sorted(self._states))

    def fit(
        self,
        normal_sequences: Sequence[Sequence[str]],
        validation_sequences: Sequence[Sequence[str]],
    ) -> None:
        self._transitions = defaultdict(Counter)
        self._target_counts = Counter()
        self._states = set()
        self._train_count = sum(len(seq) for seq in normal_sequences)
        self._validation_count = sum(len(seq) for seq in validation_sequences)

        for sequence in normal_sequences:
            previous = START_STATE
            for current in sequence:
                self._transitions[previous][current] += 1
                self._target_counts[current] += 1
                self._states.add(current)
                previous = current

        if not self._states:
            self._threshold = None
            return

        calibration_scores = list(self._scores(validation_sequences))
        if not calibration_scores:
            calibration_scores = list(self._scores(normal_sequences))
        self._threshold = _quantile(calibration_scores, self._target_fpr)

    def fit_from_flows(
        self,
        normal_sequences: Sequence[Sequence[Flow]],
        validation_sequences: Sequence[Sequence[Flow]],
        extractor: WebFlowStateExtractor | None = None,
    ) -> None:
        state_extractor = extractor or WebFlowStateExtractor()
        self.fit(
            [_flow_states(seq, state_extractor) for seq in normal_sequences],
            [_flow_states(seq, state_extractor) for seq in validation_sequences],
        )

    def assess_transition(self, previous_state: str, current_state: str) -> MarkovAssessment:
        if self._threshold is None:
            return MarkovAssessment(
                False,
                0.0,
                previous_state,
                current_state,
                "markov model not trained",
                covered=False,
            )

        score = self.score_transition(previous_state, current_state)
        blocked = score < self._threshold
        if blocked:
            reason = (
                f"markov anomaly: transition {previous_state!r} -> {current_state!r} "
                f"score {score:.3f} < threshold {self._threshold:.3f}"
            )
        else:
            reason = (
                f"markov normal: transition {previous_state!r} -> {current_state!r} "
                f"score {score:.3f} >= threshold {self._threshold:.3f}"
            )
        return MarkovAssessment(blocked, score, previous_state, current_state, reason)

    def score_transition(self, previous_state: str, current_state: str) -> float:
        return math.log(self._transition_probability(previous_state, current_state))

    def _transition_probability(self, previous_state: str, current_state: str) -> float:
        vocabulary = max(len(self._states), 1)
        counts = self._transitions.get(previous_state, Counter())
        total = sum(counts.values())
        first_order = (counts[current_state] + self._alpha) / (
            total + self._alpha * vocabulary
        )
        if self._transition_weight >= 1.0:
            return first_order

        target_total = sum(self._target_counts.values())
        target_prior = (self._target_counts[current_state] + self._alpha) / (
            target_total + self._alpha * vocabulary
        )
        probability = (
            self._transition_weight * first_order
            + (1.0 - self._transition_weight) * target_prior
        )
        if counts[current_state] > 0:
            probability = max(probability, 1.0 / len(counts))
        return probability

    def _scores(self, sequences: Iterable[Sequence[str]]) -> Iterable[float]:
        for sequence in sequences:
            previous = START_STATE
            for current in sequence:
                yield self.score_transition(previous, current)
                previous = current


def _flow_states(sequence: Sequence[Flow], extractor: WebFlowStateExtractor) -> list[str]:
    return [extractor.state_from_flow(flow) for flow in sequence]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]
