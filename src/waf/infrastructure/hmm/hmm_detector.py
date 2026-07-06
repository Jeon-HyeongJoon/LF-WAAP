"""HmmDetector — WAF Detector backed by a partitioned, one-class byte-level HMM.

Implements the `Detector` port so it plugs into BlockDecisionService alongside
RuleSetDetector. Internally it adapts each HttpRequest into a Flow and delegates
to a PartitionedHmm. It can also score Flows directly (e.g. raw TCP/TLS traffic).
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.flow import Flow
from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.hmm.features import ByteClassExtractor, FeatureExtractor
from waf.infrastructure.hmm.http_flow import http_request_to_flow
from waf.infrastructure.hmm.partitioned_hmm import Assessment, PartitionedHmm


class HmmDetector:
    name = "hmm"

    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        n_states: int = 8,
        window: int = 0,
        ensemble_size: int = 1,
        max_fit_windows: int | None = None,
        target_fpr: float = 0.01,
        min_train_sequences: int = 10,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        # ByteClassExtractor (8-symbol alphabet) is a robust default for sparse
        # data; swap in RawByteExtractor / WindowTokenExtractor / FlowFeatureExtractor.
        # window > 0 enables HMMPayl sliding-window scoring; ensemble_size > 1 enables
        # the HMMPayl random-restart ensemble.
        self._partitioned = PartitionedHmm(
            extractor=extractor or ByteClassExtractor(),
            n_states=n_states,
            window=window,
            ensemble_size=ensemble_size,
            max_fit_windows=max_fit_windows,
            target_fpr=target_fpr,
            min_train_sequences=min_train_sequences,
            n_iter=n_iter,
            random_state=random_state,
        )

    @property
    def is_trained(self) -> bool:
        return self._partitioned.is_trained

    @property
    def extractor_name(self) -> str:
        return self._partitioned.extractor.name

    # --- model structure (for reporting) --------------------------------

    @property
    def n_states(self) -> int:
        return self._partitioned.n_states

    @property
    def n_symbols(self) -> int:
        return self._partitioned.n_symbols

    @property
    def window(self) -> int:
        return self._partitioned.window

    @property
    def ensemble_size(self) -> int:
        return self._partitioned.ensemble_size

    @property
    def num_partitions(self) -> int:
        return len(self._partitioned.partitions)

    @property
    def train_count(self) -> int | None:
        return self._partitioned.train_count

    @property
    def validation_count(self) -> int | None:
        return self._partitioned.validation_count

    def train(self, normal_flows: Sequence[Flow], validation_flows: Sequence[Flow]) -> None:
        self._partitioned.train(normal_flows, validation_flows)

    def assess_flow(self, flow: Flow) -> Assessment:
        return self._partitioned.assess(flow)

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        if not self.is_trained:
            # Fail open: an untrained model abstains rather than blocking everything.
            return DetectionSignal(self.name, blocked=False, reason="model not trained", score=0.0)

        result = self._partitioned.assess(http_request_to_flow(request))
        # Suspicion margin below threshold (0 when allowed / abstaining).
        suspicion = max(0.0, -result.score) if result.blocked else 0.0
        return DetectionSignal(self.name, blocked=result.blocked, reason=result.reason, score=suspicion)

    # --- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        if not self.is_trained:
            raise RuntimeError("cannot save an untrained model")
        Path(path).write_bytes(pickle.dumps(self._partitioned))

    @classmethod
    def load(cls, path: str | Path) -> "HmmDetector":
        detector = cls.__new__(cls)
        detector._partitioned = pickle.loads(Path(path).read_bytes())
        return detector
