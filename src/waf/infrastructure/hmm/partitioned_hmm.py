"""PartitionedHmm — trains and routes one NormalHMM per partition key."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from waf.domain.model.flow import Flow
from waf.infrastructure.hmm.ensemble_hmm import EnsembleNormalHMM
from waf.infrastructure.hmm.features import FeatureExtractor
from waf.infrastructure.hmm.normal_hmm import NormalHMM, OneClassModel
from waf.infrastructure.hmm.partition import PartitionKey, partition_key


@dataclass(frozen=True, slots=True)
class Assessment:
    blocked: bool
    score: float
    key: PartitionKey
    reason: str
    covered: bool = True  # False when no calibrated model exists for the partition


class PartitionedHmm:
    def __init__(
        self,
        extractor: FeatureExtractor,
        n_states: int = 8,
        window: int = 0,
        ensemble_size: int = 1,
        max_fit_windows: int | None = None,
        target_fpr: float = 0.01,
        min_train_sequences: int = 10,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self._extractor = extractor
        self._n_states = n_states
        self._window = window
        self._ensemble_size = ensemble_size
        self._max_fit_windows = max_fit_windows
        self._target_fpr = target_fpr
        self._min_train = min_train_sequences
        self._n_iter = n_iter
        self._random_state = random_state
        self._models: dict[PartitionKey, OneClassModel] = {}

    def _new_model(self) -> OneClassModel:
        if self._ensemble_size > 1:
            return EnsembleNormalHMM(
                n_symbols=self._extractor.n_symbols,
                n_states=self._n_states,
                n_models=self._ensemble_size,
                window=self._window,
                max_fit_windows=self._max_fit_windows,
                n_iter=self._n_iter,
                random_state=self._random_state,
            )
        return NormalHMM(
            n_symbols=self._extractor.n_symbols,
            n_states=self._n_states,
            window=self._window,
            max_fit_windows=self._max_fit_windows,
            n_iter=self._n_iter,
            random_state=self._random_state,
        )

    @property
    def is_trained(self) -> bool:
        return any(m.is_calibrated for m in self._models.values())

    @property
    def extractor(self) -> FeatureExtractor:
        return self._extractor

    # --- model structure (for reporting) --------------------------------

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_symbols(self) -> int:
        return self._extractor.n_symbols

    @property
    def window(self) -> int:
        return self._window

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    @property
    def train_count(self) -> int | None:
        """Normal flows fitted on, recorded at train time (None for older pickles)."""
        return getattr(self, "_train_count", None)

    @property
    def validation_count(self) -> int | None:
        """Validation flows used to calibrate the threshold (None for older pickles)."""
        return getattr(self, "_val_count", None)

    @property
    def partitions(self) -> tuple[PartitionKey, ...]:
        return tuple(self._models.keys())

    def train(self, normal_flows: Sequence[Flow], validation_flows: Sequence[Flow]) -> None:
        """Fit per partition on normal_flows, calibrate threshold on validation_flows.

        Partitions with fewer than `min_train_sequences` training samples or no
        validation samples are skipped (left to fail open at inspection time).
        """
        self._train_count = len(normal_flows)
        self._val_count = len(validation_flows)
        train_groups: dict[PartitionKey, list[list[int]]] = defaultdict(list)
        for flow in normal_flows:
            train_groups[partition_key(flow)].append(self._extractor.extract(flow))

        val_groups: dict[PartitionKey, list[list[int]]] = defaultdict(list)
        for flow in validation_flows:
            val_groups[partition_key(flow)].append(self._extractor.extract(flow))

        self._models = {}
        for key, sequences in train_groups.items():
            if len(sequences) < self._min_train or not val_groups.get(key):
                continue
            model = self._new_model()
            model.fit(sequences)
            model.calibrate(val_groups[key], target_fpr=self._target_fpr)
            self._models[key] = model

    def assess(self, flow: Flow) -> Assessment:
        key = partition_key(flow)
        model = self._models.get(key)
        if model is None or not model.is_calibrated:
            # Fail open: no calibrated model for this partition -> abstain.
            return Assessment(
                False, 0.0, key, f"no calibrated model for partition {key}", covered=False
            )

        blocked, score = model.is_anomaly(self._extractor.extract(flow))
        thr = model.threshold
        if blocked:
            reason = f"anomaly: score {score:.3f} < threshold {thr:.3f} (partition {key})"
        else:
            reason = f"normal: score {score:.3f} >= threshold {thr:.3f}"
        return Assessment(blocked, score, key, reason)
