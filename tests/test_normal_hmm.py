import pytest

pytest.importorskip("hmmlearn")

from waf.infrastructure.hmm.evaluation import evaluate  # noqa: E402
from waf.infrastructure.hmm.features import ByteClassExtractor  # noqa: E402
from waf.infrastructure.hmm.normal_hmm import NormalHMM, NotCalibratedError  # noqa: E402

from synthetic import attack_flow, batch, normal_flow  # noqa: E402

EXT = ByteClassExtractor()


def _seqs(flows):
    return [EXT.extract(f) for f in flows]


def test_requires_calibration_before_use() -> None:
    model = NormalHMM(n_symbols=EXT.n_symbols, n_states=4, n_iter=30)
    model.fit(_seqs(batch(normal_flow, 30, seed=1)))
    with pytest.raises(NotCalibratedError):
        model.is_anomaly([1, 2, 3])


def test_calibration_hits_target_fpr_and_separates_attacks() -> None:
    model = NormalHMM(n_symbols=EXT.n_symbols, n_states=4, n_iter=50)
    model.fit(_seqs(batch(normal_flow, 80, seed=1)))
    model.calibrate(_seqs(batch(normal_flow, 60, seed=2)), target_fpr=0.05)

    normal_test = _seqs(batch(normal_flow, 100, seed=3))
    attack_test = _seqs(batch(attack_flow, 100, seed=4))

    fpr = sum(model.is_anomaly(s)[0] for s in normal_test) / len(normal_test)
    tpr = sum(model.is_anomaly(s)[0] for s in attack_test) / len(attack_test)

    assert fpr < 0.20  # near the 5% target, loose for sampling noise
    assert tpr > 0.80  # high-entropy attacks are flagged

    report = evaluate(
        [model.score(s) for s in normal_test],
        [model.score(s) for s in attack_test],
        target_fpr=0.05,
    )
    assert report.roc_auc > 0.90
    assert report.pr_auc > 0.85


def test_window_score_matches_per_window_reference() -> None:
    # Pins the windowed score to the explicit per-window forward-loglik formula, so the
    # vectorized implementation must stay numerically equivalent to scoring each window.
    import numpy as np
    from scipy.special import logsumexp

    from waf.infrastructure.hmm.normal_hmm import sliding_windows

    window = 4
    model = NormalHMM(
        n_symbols=EXT.n_symbols, n_states=3, window=window, max_fit_windows=500, n_iter=30
    )
    model.fit(_seqs(batch(normal_flow, 60, seed=1)))

    seq = _seqs(batch(normal_flow, 1, seed=5))[0]
    windows = sliding_windows(seq, window)
    ref_logliks = [
        float(model._model.score(np.asarray(w, dtype=int).reshape(-1, 1))) for w in windows
    ]
    expected = float(logsumexp(ref_logliks) - np.log(len(windows))) / window

    assert model.score(seq) == pytest.approx(expected, rel=1e-6)


def test_fit_caps_training_windows_yet_still_works() -> None:
    # Windowing explodes the training set (~150 windows/payload); a cap keeps fit
    # bounded (HMMPayl §6.4 sampling) while the model still trains and scores.
    model = NormalHMM(
        n_symbols=EXT.n_symbols, n_states=3, window=4, max_fit_windows=300, n_iter=20
    )
    model.fit(_seqs(batch(normal_flow, 60, seed=1)))  # ~9000 windows uncapped
    model.calibrate(_seqs(batch(normal_flow, 30, seed=2)), target_fpr=0.1)

    blocked, score = model.is_anomaly(_seqs(batch(normal_flow, 1, seed=5))[0])
    assert isinstance(score, float)


def test_sliding_window_scoring_works_with_few_states() -> None:
    # HMMPayl: score length-n windows, average -> the HMM needs far fewer states
    # than the payload length yet still separates normal from anomalous traffic.
    model = NormalHMM(
        n_symbols=EXT.n_symbols, n_states=3, window=4, max_fit_windows=2000, n_iter=50
    )
    model.fit(_seqs(batch(normal_flow, 80, seed=1)))
    model.calibrate(_seqs(batch(normal_flow, 60, seed=2)), target_fpr=0.1)

    normal = [model.score(s) for s in _seqs(batch(normal_flow, 60, seed=3))]
    attack = [model.score(s) for s in _seqs(batch(attack_flow, 60, seed=4))]

    assert sum(normal) / len(normal) > sum(attack) / len(attack)
    assert evaluate(normal, attack, target_fpr=0.1).roc_auc > 0.85
