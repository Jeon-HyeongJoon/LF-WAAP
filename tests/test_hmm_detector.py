import pytest

pytest.importorskip("hmmlearn")

from waf.domain.model.flow import Direction, Flow  # noqa: E402
from waf.domain.model.http_request import HttpRequest  # noqa: E402
from waf.infrastructure.hmm import HmmDetector, PartitionedHmm  # noqa: E402
from waf.infrastructure.hmm.features import ByteClassExtractor  # noqa: E402

from synthetic import attack_flow, batch, normal_flow, normal_request  # noqa: E402


def build_detector() -> HmmDetector:
    det = HmmDetector(n_states=4, target_fpr=0.05, n_iter=50)
    det.train(batch(normal_flow, 80, seed=1), batch(normal_flow, 60, seed=2))
    return det


def test_untrained_detector_fails_open() -> None:
    det = HmmDetector()
    req = HttpRequest("GET", "/", "id=1")
    signal = det.inspect(req)
    assert signal.blocked is False
    assert "not trained" in signal.reason


def test_assess_flow_blocks_attack_allows_normal() -> None:
    det = build_detector()
    import random

    rng = random.Random(99)
    assert det.assess_flow(normal_flow(rng)).blocked is False
    assert det.assess_flow(attack_flow(rng)).blocked is True


def test_inspect_allows_most_normal_http_requests() -> None:
    # At a calibrated ~5% FPR, individual normals can be flagged; assert the
    # aggregate false-positive rate stays low rather than testing one sample.
    det = build_detector()
    import random

    rng = random.Random(7)
    requests = [normal_request(rng) for _ in range(50)]
    blocked = sum(det.inspect(r).blocked for r in requests)
    assert blocked / len(requests) < 0.20


def test_detector_supports_sliding_window_scoring() -> None:
    det = HmmDetector(
        n_states=3, window=4, max_fit_windows=1000,
        target_fpr=0.1, min_train_sequences=5, n_iter=30,
    )
    det.train(batch(normal_flow, 50, seed=1), batch(normal_flow, 30, seed=2))
    import random

    rng = random.Random(99)
    assert det.assess_flow(normal_flow(rng)).blocked is False
    assert det.assess_flow(attack_flow(rng)).blocked is True


def test_detector_supports_ensemble() -> None:
    from waf.infrastructure.hmm import evaluate

    det = HmmDetector(n_states=4, ensemble_size=3, target_fpr=0.1, min_train_sequences=5, n_iter=30)
    det.train(batch(normal_flow, 60, seed=1), batch(normal_flow, 30, seed=2))

    # threshold-independent check: the ensemble produces discriminative scores
    normal = [det.assess_flow(f).score for f in batch(normal_flow, 40, seed=3)]
    attack = [det.assess_flow(f).score for f in batch(attack_flow, 40, seed=4)]
    assert evaluate(normal, attack, target_fpr=0.1).roc_auc > 0.9


def test_partitioned_hmm_fails_open_for_unknown_partition() -> None:
    ph = PartitionedHmm(ByteClassExtractor(), n_states=4, target_fpr=0.05, n_iter=50)
    ph.train(batch(normal_flow, 80, seed=1), batch(normal_flow, 60, seed=2))
    # A flow on a port/length the model never saw -> abstain, not crash.
    unknown = Flow("smtp", 25, Direction.OUTBOUND, b"x" * 9000)
    result = ph.assess(unknown)
    assert result.blocked is False
    assert "no calibrated model" in result.reason


def test_save_and_load_roundtrip(tmp_path) -> None:
    det = build_detector()
    path = tmp_path / "hmm.pkl"
    det.save(path)
    loaded = HmmDetector.load(path)
    assert loaded.is_trained

    import random

    rng = random.Random(123)
    flow = attack_flow(rng)
    assert loaded.assess_flow(flow).blocked == det.assess_flow(flow).blocked
