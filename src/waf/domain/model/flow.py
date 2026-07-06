"""Flow — the byte-level / flow-level unit the HMM detector observes.

Distinct from HttpRequest (the WAF's L7 inspection unit), Flow carries the raw
bytes and the flow metadata (protocol / port / direction / length, and optional
per-packet records) that the anomaly model needs for:

  * raw-byte observations (PAYL / Anagram / HMMPayl lineage), and
  * model partitioning by (protocol, port, direction, length bucket), and
  * encrypted-traffic features (direction + length bin + inter-arrival + flags)
    when payload bytes are opaque.

An HttpRequest is adapted into a Flow at the detector boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    """Traffic direction relative to the protected service."""

    INBOUND = "INBOUND"  # client -> server (request)
    OUTBOUND = "OUTBOUND"  # server -> client (response)


@dataclass(frozen=True, slots=True)
class PacketMeta:
    """Per-packet metadata for encrypted/opaque flows (no payload inspection)."""

    direction: Direction
    length: int
    inter_arrival_ms: float = 0.0
    tcp_flags: str = ""  # e.g. "S", "SA", "A", "PA", "FA", "R"


@dataclass(frozen=True, slots=True)
class Flow:
    protocol: str  # "http", "tcp", "tls", ...
    port: int
    direction: Direction
    payload: bytes = b""
    packets: tuple[PacketMeta, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol", self.protocol.lower())

    @property
    def length(self) -> int:
        """Bytes used for length bucketing: payload size, or summed packet sizes."""
        if self.payload:
            return len(self.payload)
        return sum(p.length for p in self.packets)
