import pytest

pytest.importorskip("hmmlearn")

from waf.infrastructure.hmm.ensemble_hmm import EnsembleNormalHMM  # noqa: E402
from waf.infrastructure.hmm.evaluation import evaluate  # noqa: E402
from waf.infrastructure.hmm.features import ByteClassExtractor  # noqa: E402

from synthetic import attack_flow, batch, normal_flow  # noqa: E402

EXT = ByteClassExtractor()


def _seqs(flows):
    return [EXT.extract(f) for f in flows]


def test_ensemble_score_is_mean_of_member_scores() -> None:
    ens = EnsembleNormalHMM(n_symbols=EXT.n_symbols, n_states=4, n_models=3, n_iter=30)
    ens.fit(_seqs(batch(normal_flow, 60, seed=1)))

    seq = _seqs(batch(normal_flow, 1, seed=5))[0]
    members = ens.member_scores(seq)

    assert len(members) == 3
    assert ens.score(seq) == pytest.approx(sum(members) / 3)


def test_ensemble_members_differ_by_random_init() -> None:
    ens = EnsembleNormalHMM(n_symbols=EXT.n_symbols, n_states=4, n_models=3, n_iter=30)
    ens.fit(_seqs(batch(normal_flow, 60, seed=1)))
    seq = _seqs(batch(normal_flow, 1, seed=5))[0]
    members = ens.member_scores(seq)
    assert len(set(members)) > 1  # random restarts produce different models


def test_ensemble_separates_after_calibration() -> None:
    ens = EnsembleNormalHMM(n_symbols=EXT.n_symbols, n_states=4, n_models=3, n_iter=40)
    ens.fit(_seqs(batch(normal_flow, 80, seed=1)))
    ens.calibrate(_seqs(batch(normal_flow, 60, seed=2)), target_fpr=0.1)

    normal = [ens.score(s) for s in _seqs(batch(normal_flow, 60, seed=3))]
    attack = [ens.score(s) for s in _seqs(batch(attack_flow, 60, seed=4))]
    assert evaluate(normal, attack, target_fpr=0.1).roc_auc > 0.9
