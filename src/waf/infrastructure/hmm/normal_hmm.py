"""NormalHMM — a one-class HMM trained on NORMAL traffic only.

Key modeling decisions (from the spec, grounded in HMMPayl / state-count lit.):

  * One-class / baseline: we fit on benign sequences only and flag low-likelihood
    inputs. We do NOT label hidden states as normal/attack — the hidden states are
    latent *phases* of the payload/flow (e.g. header vs body regions). n_states is
    a capacity hyperparameter, selected empirically (validation metrics), not 2.
  * Score = log P(sequence | normal_HMM) / len(sequence): length-normalized
    log-likelihood so long and short payloads are comparable.
  * Threshold is calibrated on a held-out NORMAL validation set to a target FPR,
    not eyeballed — anomaly iff per-symbol score < threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from hmmlearn.hmm import CategoricalHMM
from numpy.typing import NDArray
from scipy.special import logsumexp


@runtime_checkable
class OneClassModel(Protocol):
    """The interface PartitionedHmm needs — satisfied by NormalHMM and EnsembleNormalHMM."""

    @property
    def is_calibrated(self) -> bool: ...

    @property
    def threshold(self) -> float | None: ...

    def fit(self, sequences: Sequence[Sequence[int]]) -> "OneClassModel": ...

    def calibrate(
        self, validation_sequences: Sequence[Sequence[int]], target_fpr: float = ...
    ) -> float: ...

    def is_anomaly(self, sequence: Sequence[int]) -> tuple[bool, float]: ...


class NotCalibratedError(RuntimeError):
    pass


def sliding_windows(sequence: Sequence[int], window: int) -> list[list[int]]:
    """Split a symbol sequence into length-`window` subsequences, sliding one symbol
    at a time (HMMPayl eq.6: N = L - n + 1). `window <= 0` returns the whole sequence
    as one window; sequences shorter than the window are returned as a single window."""
    seq = list(sequence)
    if window <= 0 or len(seq) <= window:
        return [seq]
    return [seq[i : i + window] for i in range(len(seq) - window + 1)]


class NormalHMM:
    def __init__(
        self,
        n_symbols: int,
        n_states: int = 8,
        window: int = 0,
        max_fit_windows: int | None = None,
        smoothing: float = 1e-3,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self._n_symbols = n_symbols
        self._n_states = n_states
        # window <= 0: score the whole payload as one sequence. window = n:
        # HMMPayl-style — fit on / score length-n sliding subsequences.
        self._window = window
        # cap on training windows (HMMPayl §6.4 sampling) — windowing otherwise
        # produces ~L windows per payload, exploding fit cost.
        self._max_fit_windows = max_fit_windows
        self._smoothing = smoothing
        self._n_iter = n_iter
        self._random_state = random_state
        self._model: CategoricalHMM | None = None
        self._threshold: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def is_calibrated(self) -> bool:
        return self._threshold is not None

    @property
    def threshold(self) -> float | None:
        return self._threshold

    def fit(self, sequences: Sequence[Sequence[int]]) -> "NormalHMM":
        seqs = [s for s in sequences if len(s) > 0]
        if not seqs:
            raise ValueError("fit requires at least one non-empty sequence")

        # With windowing, the HMM is fit on the length-n subsequences, which bounds
        # the effective sequence length (and thus the needed state count).
        windows = [w for s in seqs for w in sliding_windows(s, self._window)]
        if self._max_fit_windows and len(windows) > self._max_fit_windows:
            rng = np.random.default_rng(self._random_state)
            picked = rng.choice(len(windows), self._max_fit_windows, replace=False)
            windows = [windows[i] for i in picked]
        X = np.concatenate([np.asarray(w, dtype=int).reshape(-1, 1) for w in windows])
        lengths = [len(w) for w in windows]

        model = CategoricalHMM(
            n_components=self._n_states,
            n_features=self._n_symbols,
            random_state=self._random_state,
            n_iter=self._n_iter,
        )
        model.fit(X, lengths)

        # Laplace-smooth emissions: symbols unseen in training get a small,
        # non-zero probability instead of -inf (critical with a 256-byte alphabet).
        smoothed = model.emissionprob_ + self._smoothing
        model.emissionprob_ = smoothed / smoothed.sum(axis=1, keepdims=True)
        self._model = model
        return self

    def score(self, sequence: Sequence[int]) -> float:
        """Per-symbol normality score. Higher = more normal.

        window <= 0: log P(seq | normal) / len(seq).
        window = n:  HMMPayl fusion (eq.8) — the arithmetic mean of the N window
                     probabilities, computed stably in log space, per symbol.
        """
        if self._model is None:
            raise NotCalibratedError("model is not fitted")
        seq = np.asarray(sequence, dtype=int)
        if seq.size == 0:
            return 0.0

        if self._window > 0 and seq.size > self._window:
            logliks = self._window_logliks(seq)
            mean_logprob = float(logsumexp(logliks) - np.log(logliks.size))
            return mean_logprob / self._window

        # whole-sequence (window off) or sequence shorter than the window
        loglik = float(self._model.score(seq.reshape(-1, 1)))
        norm = self._window if self._window > 0 else seq.size
        return loglik / max(norm, 1)

    def _window_logliks(self, seq: NDArray[np.int_]) -> NDArray[np.float64]:
        """Forward log-likelihood of every length-`window` subsequence, computed for
        all N windows at once (vectorized) instead of one hmmlearn.score per window."""
        model = self._model
        assert model is not None
        with np.errstate(divide="ignore"):
            log_start = np.log(model.startprob_)  # (S,)
            log_trans = np.log(model.transmat_)  # (S_from, S_to)
            log_emit = np.log(model.emissionprob_)  # (S, n_symbols)

        n = self._window
        emit_pos = log_emit[:, seq].T  # (L, S): per-position emission log-prob
        n_windows = seq.size - n + 1

        # alpha[j, s] = forward log-prob of window j being in state s at the current step.
        alpha = log_start[None, :] + emit_pos[0:n_windows]  # t = 0
        for t in range(1, n):
            trans = alpha[:, :, None] + log_trans[None, :, :]  # (N, S_from, S_to)
            alpha = logsumexp(trans, axis=1) + emit_pos[t : t + n_windows]
        return np.asarray(logsumexp(alpha, axis=1))  # (N,)

    def calibrate(
        self,
        validation_sequences: Sequence[Sequence[int]],
        target_fpr: float = 0.01,
    ) -> float:
        """Set the threshold so ~target_fpr of NORMAL validation samples fall below
        it (i.e. would be false positives). Returns the chosen threshold."""
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        scores = np.array([self.score(s) for s in validation_sequences if len(s) > 0])
        if scores.size == 0:
            raise ValueError("calibration requires non-empty validation sequences")
        # target_fpr fraction of normals sit below this quantile.
        self._threshold = float(np.quantile(scores, target_fpr))
        return self._threshold

    def is_anomaly(self, sequence: Sequence[int]) -> tuple[bool, float]:
        """Return (is_anomalous, per_symbol_score)."""
        if self._threshold is None:
            raise NotCalibratedError("model is not calibrated; call calibrate() first")
        s = self.score(sequence)
        return s < self._threshold, s
