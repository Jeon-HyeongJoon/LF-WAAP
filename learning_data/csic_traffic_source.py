"""CsicTrafficSource — adapts the CSIC 2010 CSV into the domain TrafficSource port.

Streams benign packets from the dataset, wraps each as a Flow, and splits them
deterministically into normal (fit) and validation (calibration) subsets. Anomalous
records are excluded — the HMM is trained one-class on normal traffic only.

Lives with the dataset (not in the core package) because it is CSIC-specific; it
depends only inward on waf.domain.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

from csic_to_packets import PacketRecord, iter_packets

from waf.domain.model.flow import Direction, Flow
from waf.domain.model.http_request import HttpRequest

_DEFAULT_HTTP_PORT = 80


def _port_from_host(host: str) -> int:
    if ":" in host:
        try:
            return int(host.rsplit(":", 1)[1])
        except ValueError:
            return _DEFAULT_HTTP_PORT
    return _DEFAULT_HTTP_PORT


def record_to_flow(record: PacketRecord) -> Flow:
    """Map a reconstructed CSIC packet to a domain Flow (shared by source + eval)."""
    return Flow(
        protocol="http",
        port=_port_from_host(record.host),
        direction=Direction.INBOUND,
        payload=record.raw,
    )


def record_to_request(record: PacketRecord) -> HttpRequest:
    """Map a reconstructed CSIC packet to the WAF's L7 request value object."""
    head, _, body = record.raw.decode("latin-1").partition("\r\n\r\n")
    lines = head.split("\r\n")
    method, target, *_ = lines[0].split(" ")
    split = urlsplit(target)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers[name] = value.strip()
    return HttpRequest(
        method=method,
        path=split.path or "/",
        query=split.query,
        headers=headers,
        body=body,
    )


class CsicTrafficSource:
    def __init__(
        self,
        csv_path: str | Path,
        validation_ratio: float = 0.2,
        limit: int | None = None,
    ) -> None:
        if not 0.0 <= validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in [0, 1)")
        self._csv_path = csv_path
        self._limit = limit
        # Deterministic stride: every `stride`-th benign record is held out for
        # validation. ratio 0 -> no held-out records (all go to normal/fit).
        self._stride = round(1 / validation_ratio) if validation_ratio > 0 else 0

    def _is_validation(self, benign_index: int) -> bool:
        if self._stride == 0:
            return False
        return benign_index % self._stride == self._stride - 1

    def _benign_flows(self) -> Iterator[tuple[int, Flow]]:
        benign_index = 0
        for record in iter_packets(self._csv_path, limit=self._limit):
            if record.is_normal:
                yield benign_index, record_to_flow(record)
                benign_index += 1

    def normal(self) -> Iterator[Flow]:
        for i, flow in self._benign_flows():
            if not self._is_validation(i):
                yield flow

    def validation(self) -> Iterator[Flow]:
        for i, flow in self._benign_flows():
            if self._is_validation(i):
                yield flow
