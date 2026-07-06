"""Model partitioning key.

Per the spec, separate HMMs are trained per (protocol, port, direction, length
bucket). Mixing, say, 40-byte GET query strings with 8 KB POST bodies into one
model blurs the "normal" distribution; partitioning keeps each model's notion of
normal tight, which is what makes a low target FPR achievable.
"""

from __future__ import annotations

from dataclasses import dataclass

from waf.domain.model.flow import Direction, Flow

_LENGTH_EDGES = (64, 256, 1024, 4096, 16384)


def length_bucket(n: int) -> str:
    lo = 0
    for edge in _LENGTH_EDGES:
        if n <= edge:
            return f"{lo}-{edge}"
        lo = edge + 1
    return f"{lo}+"


@dataclass(frozen=True, slots=True)
class PartitionKey:
    protocol: str
    port: int
    direction: Direction
    length_bucket: str

    def __str__(self) -> str:
        return f"{self.protocol}:{self.port}:{self.direction.value}:{self.length_bucket}"


def partition_key(flow: Flow) -> PartitionKey:
    return PartitionKey(
        protocol=flow.protocol,
        port=flow.port,
        direction=flow.direction,
        length_bucket=length_bucket(flow.length),
    )
