"""Feature extractors — turn a Flow into a sequence of integer observations.

Design rationale (from the spec / literature):

  * Raw byte 0-255 is the most direct observation (PAYL/Anagram/HMMPayl lineage).
  * n-grams are informative but suffer 256^n dimension blow-up, so for HMMs we
    start from raw byte, byte *class* (reduced alphabet), or windowed tokens.
  * For encrypted traffic, payload bytes are opaque — use flow features instead:
    direction + length bin + inter-arrival-time bin + TCP flag combo.

Every extractor reports a fixed `n_symbols` so the HMM's emission matrix has a
stable shape across train / validate / score.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from waf.domain.model.flow import Direction, Flow


def extractor_by_name(name: str) -> "FeatureExtractor":
    """Resolve a feature extractor from a config/CLI string."""
    key = name.strip().lower()
    if key == "raw_byte":
        return RawByteExtractor()
    if key == "byte_class":
        return ByteClassExtractor()
    if key == "window_token":
        return WindowTokenExtractor()
    raise ValueError(
        f"unknown extractor {name!r} (choose: raw_byte, byte_class, window_token)"
    )


@runtime_checkable
class FeatureExtractor(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def n_symbols(self) -> int: ...

    def extract(self, flow: Flow) -> list[int]:
        """Return a non-empty sequence of symbols in [0, n_symbols)."""
        ...


def _nonempty(seq: list[int]) -> list[int]:
    return seq if seq else [0]


class RawByteExtractor:
    """Each payload byte is its own observation. Alphabet = 256."""

    name = "raw_byte"
    n_symbols = 256

    def extract(self, flow: Flow) -> list[int]:
        return _nonempty(list(flow.payload))


# Byte -> reduced class. Keeps the alphabet small (less data-hungry than 256).
_WHITESPACE = {9, 10, 13, 32}


def byte_class(b: int) -> int:
    if b == 0:
        return 0  # NUL
    if b in _WHITESPACE:
        return 1  # whitespace
    if 1 <= b <= 31 or b == 127:
        return 2  # control
    if 48 <= b <= 57:
        return 3  # digit
    if 65 <= b <= 90:
        return 4  # uppercase
    if 97 <= b <= 122:
        return 5  # lowercase
    if b >= 128:
        return 7  # high-bit (non-ASCII)
    return 6  # printable punctuation/symbols


class ByteClassExtractor:
    """Map each byte to one of 8 classes. Robust default for sparse data."""

    name = "byte_class"
    n_symbols = 8

    def extract(self, flow: Flow) -> list[int]:
        return _nonempty([byte_class(b) for b in flow.payload])


class WindowTokenExtractor:
    """Sliding n-gram over byte classes — captures local structure without the
    256^n blow-up of raw-byte n-grams (C^window symbols, C=8)."""

    def __init__(self, window: int = 2) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._classes = 8

    @property
    def name(self) -> str:
        return f"window_token(w={self._window})"

    @property
    def n_symbols(self) -> int:
        return int(self._classes**self._window)

    def extract(self, flow: Flow) -> list[int]:
        classes = [byte_class(b) for b in flow.payload]
        if len(classes) < self._window:
            classes = classes + [0] * (self._window - len(classes))
        tokens: list[int] = []
        for i in range(len(classes) - self._window + 1):
            token = 0
            for c in classes[i : i + self._window]:
                token = token * self._classes + c
            tokens.append(token)
        return _nonempty(tokens)


# --- encrypted / opaque traffic -------------------------------------------

_LENGTH_BINS = (64, 256, 512, 1024, 1460, 4096, 16384)  # -> 8 bins
_IAT_BINS = (1.0, 10.0, 100.0, 1000.0)  # ms -> 5 bins
_FLAG_INDEX = {"S": 0, "SA": 1, "A": 2, "PA": 3, "FA": 4, "F": 5, "R": 6}
_FLAG_OTHER = 7
_N_FLAGS = 8


def _bin(value: float, edges: tuple[float, ...]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


class FlowFeatureExtractor:
    """For encrypted traffic: one symbol per packet from
    (direction, length bin, inter-arrival bin, TCP flag combo)."""

    name = "flow_feature"
    _N_LEN = len(_LENGTH_BINS) + 1  # 8
    _N_IAT = len(_IAT_BINS) + 1  # 5

    @property
    def n_symbols(self) -> int:
        return 2 * self._N_LEN * self._N_IAT * _N_FLAGS

    def extract(self, flow: Flow) -> list[int]:
        symbols: list[int] = []
        for pkt in flow.packets:
            d = 0 if pkt.direction is Direction.INBOUND else 1
            lb = _bin(float(pkt.length), _LENGTH_BINS)
            ib = _bin(pkt.inter_arrival_ms, _IAT_BINS)
            fb = _FLAG_INDEX.get(pkt.tcp_flags.upper(), _FLAG_OTHER)
            symbols.append(((d * self._N_LEN + lb) * self._N_IAT + ib) * _N_FLAGS + fb)
        return _nonempty(symbols)
