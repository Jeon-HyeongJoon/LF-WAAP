from waf.domain.model.flow import Direction, Flow
from waf.infrastructure.hmm.partition import length_bucket, partition_key


def test_length_buckets() -> None:
    assert length_bucket(10) == "0-64"
    assert length_bucket(64) == "0-64"
    assert length_bucket(65) == "65-256"
    assert length_bucket(2000) == "1025-4096"
    assert length_bucket(99999) == "16385+"


def test_partition_key_separates_by_attributes() -> None:
    small = Flow("http", 80, Direction.INBOUND, b"x" * 50)
    large = Flow("http", 80, Direction.INBOUND, b"x" * 5000)
    other_port = Flow("http", 8080, Direction.INBOUND, b"x" * 50)

    assert partition_key(small) != partition_key(large)
    assert partition_key(small) != partition_key(other_port)
    assert partition_key(small) == partition_key(Flow("http", 80, Direction.INBOUND, b"y" * 60))
