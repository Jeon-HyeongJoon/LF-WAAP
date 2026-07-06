"""sliding_windows — HMMPayl-style subsequence extraction (eq.6)."""

from waf.infrastructure.hmm.normal_hmm import sliding_windows


def test_slides_byte_by_byte() -> None:
    assert sliding_windows([1, 2, 3, 4, 5], 3) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_count_is_L_minus_n_plus_1() -> None:
    seq = list(range(10))
    assert len(sliding_windows(seq, 5)) == 10 - 5 + 1  # eq.6


def test_window_one_yields_singletons() -> None:
    assert sliding_windows([1, 2, 3], 1) == [[1], [2], [3]]


def test_window_zero_means_whole_sequence() -> None:
    assert sliding_windows([1, 2, 3], 0) == [[1, 2, 3]]


def test_sequence_shorter_than_window_is_single_window() -> None:
    assert sliding_windows([1, 2], 5) == [[1, 2]]
