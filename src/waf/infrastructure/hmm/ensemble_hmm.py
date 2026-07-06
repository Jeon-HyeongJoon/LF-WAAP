"""EnsembleNormalHMM — HMMPayl-style ensemble of one-class HMMs.

K HMMs are trained on the same data with different random initializations of the
transition/emission matrices, then their per-sample scores are fused. The paper
shows an ensemble both improves accuracy and makes evasion harder than a single HMM.

Combination rules operate in log space (scores are per-symbol log-likelihoods):
  * "mean" — arithmetic mean of member log-scores (= geometric mean of probabilities,
    HMMPayl's geometric-mean rule); the robust default.
  * "min"/"max" — conservative / permissive fusion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from waf.infrastructure.hmm.normal_hmm import NormalHMM, NotCalibratedError


class EnsembleNormalHMM:
    def __init__(
        self,
        n_symbols: int,
        n_states: int = 8,
        n_models: int = 3,
        window: int = 0,
        max_fit_windows: int | None = None,
        combination: str = "mean",
        smoothing: float = 1e-3,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        if n_models < 1:
            raise ValueError("n_models must be >= 1")
        if combination not in ("mean", "min", "max"):
            raise ValueError("combination must be one of: mean, min, max")
        self._combination = combination
        self._members = [
            NormalHMM(
                n_symbols=n_symbols,
                n_states=n_states,
                window=window,
                max_fit_windows=max_fit_windows,
                smoothing=smoothing,
                n_iter=n_iter,
                random_state=random_state + i,  # different random restart per member
            )
            for i in range(n_models)
        ]
        self._threshold: float | None = None

    @property
    def is_fitted(self) -> bool:
        return all(m.is_fitted for m in self._members)

    @property
    def is_calibrated(self) -> bool:
        return self._threshold is not None

    @property
    def threshold(self) -> float | None:
        return self._threshold

    def fit(self, sequences: Sequence[Sequence[int]]) -> "EnsembleNormalHMM":
        for member in self._members:
            member.fit(sequences)
        return self

    def member_scores(self, sequence: Sequence[int]) -> list[float]:
        return [m.score(sequence) for m in self._members]

    def score(self, sequence: Sequence[int]) -> float:
        scores = self.member_scores(sequence)
        if self._combination == "mean":
            return float(np.mean(scores))
        if self._combination == "min":
            return float(np.min(scores))
        return float(np.max(scores))

    def calibrate(
        self,
        validation_sequences: Sequence[Sequence[int]],
        target_fpr: float = 0.01,
    ) -> float:
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        scores = np.array([self.score(s) for s in validation_sequences if len(s) > 0])
        if scores.size == 0:
            raise ValueError("calibration requires non-empty validation sequences")
        self._threshold = float(np.quantile(scores, target_fpr))
        return self._threshold

    def is_anomaly(self, sequence: Sequence[int]) -> tuple[bool, float]:
        if self._threshold is None:
            raise NotCalibratedError("ensemble is not calibrated; call calibrate() first")
        s = self.score(sequence)
        return s < self._threshold, s
