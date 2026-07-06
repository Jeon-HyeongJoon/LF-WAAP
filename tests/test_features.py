from waf.domain.model.flow import Direction, Flow, PacketMeta
from waf.infrastructure.hmm.features import (
    ByteClassExtractor,
    FlowFeatureExtractor,
    RawByteExtractor,
    WindowTokenExtractor,
    byte_class,
)


def flow(payload: bytes) -> Flow:
    return Flow("tcp", 443, Direction.INBOUND, payload)


def test_byte_class_buckets() -> None:
    assert byte_class(0) == 0
    assert byte_class(ord(" ")) == 1
    assert byte_class(1) == 2
    assert byte_class(ord("5")) == 3
    assert byte_class(ord("A")) == 4
    assert byte_class(ord("z")) == 5
    assert byte_class(ord("!")) == 6
    assert byte_class(200) == 7


def test_raw_byte_extractor() -> None:
    ext = RawByteExtractor()
    assert ext.n_symbols == 256
    assert ext.extract(flow(b"AB")) == [65, 66]
    assert ext.extract(flow(b"")) == [0]  # non-empty guarantee


def test_byte_class_extractor() -> None:
    ext = ByteClassExtractor()
    assert ext.n_symbols == 8
    seq = ext.extract(flow(b"A1 "))
    assert seq == [4, 3, 1]
    assert all(0 <= s < ext.n_symbols for s in seq)


def test_window_token_extractor_cardinality_and_range() -> None:
    ext = WindowTokenExtractor(window=2)
    assert ext.n_symbols == 64
    seq = ext.extract(flow(b"AAA"))
    assert all(0 <= s < 64 for s in seq)
    assert len(seq) == 2  # 3 classes, window 2 -> 2 tokens


def test_flow_feature_extractor() -> None:
    ext = FlowFeatureExtractor()
    f = Flow(
        "tls",
        443,
        Direction.INBOUND,
        packets=(
            PacketMeta(Direction.INBOUND, 100, 5.0, "PA"),
            PacketMeta(Direction.OUTBOUND, 1400, 50.0, "A"),
        ),
    )
    seq = ext.extract(f)
    assert len(seq) == 2
    assert all(0 <= s < ext.n_symbols for s in seq)
