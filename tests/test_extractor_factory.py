"""extractor_by_name — maps a CLI/config string to a FeatureExtractor instance."""

import pytest

from waf.infrastructure.hmm import extractor_by_name
from waf.infrastructure.hmm.features import (
    ByteClassExtractor,
    RawByteExtractor,
    WindowTokenExtractor,
)


def test_maps_names_to_extractor_types() -> None:
    assert isinstance(extractor_by_name("raw_byte"), RawByteExtractor)
    assert isinstance(extractor_by_name("byte_class"), ByteClassExtractor)
    assert isinstance(extractor_by_name("window_token"), WindowTokenExtractor)


def test_is_case_and_whitespace_insensitive() -> None:
    assert isinstance(extractor_by_name("  RAW_BYTE "), RawByteExtractor)


def test_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        extractor_by_name("nope")
