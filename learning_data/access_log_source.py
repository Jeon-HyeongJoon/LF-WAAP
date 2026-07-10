"""Access log training source for workflow behavior learning.

This module intentionally accepts a file path and parses records at runtime.
Callers can train from production access logs without exposing raw log lines in
tests, summaries, or agent-visible output.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from waf.domain.model.http_request import HttpRequest

_ACCESS_LOG_PATTERN = re.compile(
    r"^(?P<remote>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] "
    r'"(?P<request>[^"]*)" (?P<status>\d{3}|-) (?P<size>\S+)'
    r'(?: "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)")?.*$'
)


@dataclass(frozen=True, slots=True)
class AccessLogRecord:
    request: HttpRequest
    remote_addr: str
    timestamp: str
    status: int | None
    session_key: str


@dataclass(frozen=True, slots=True)
class AccessLogDatasetSummary:
    path_fingerprint: str
    parsed_records: int
    skipped_records: int
    training_records: int
    session_count: int
    route_count: int
    status_counts: dict[str, int]
    min_session_length: int
    max_session_length: int
    avg_session_length: float

    def to_dict(self) -> dict[str, object]:
        return {
            "path_fingerprint": self.path_fingerprint,
            "parsed_records": self.parsed_records,
            "skipped_records": self.skipped_records,
            "training_records": self.training_records,
            "session_count": self.session_count,
            "route_count": self.route_count,
            "status_counts": self.status_counts,
            "min_session_length": self.min_session_length,
            "max_session_length": self.max_session_length,
            "avg_session_length": self.avg_session_length,
        }


@dataclass(frozen=True, slots=True)
class AccessLogTrainingDataset:
    sequences: tuple[tuple[HttpRequest, ...], ...]
    summary: AccessLogDatasetSummary

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        min_session_length: int = 2,
        max_records: int | None = None,
    ) -> "AccessLogTrainingDataset":
        source = Path(path)
        records = list(iter_access_log_records(source, max_records=max_records))
        status_counts = _status_counts(records)
        training_records = [
            record for record in records if record.status is None or record.status < 400
        ]
        grouped = _group_by_session(training_records)
        sequences = tuple(
            tuple(requests)
            for requests in grouped.values()
            if len(requests) >= min_session_length
        )
        lengths = [len(sequence) for sequence in sequences]
        summary = AccessLogDatasetSummary(
            path_fingerprint=_path_fingerprint(source),
            parsed_records=len(records),
            skipped_records=_count_lines(source, max_records=max_records) - len(records),
            training_records=len(training_records),
            session_count=len(sequences),
            route_count=len({request.path for sequence in sequences for request in sequence}),
            status_counts=status_counts,
            min_session_length=min(lengths) if lengths else 0,
            max_session_length=max(lengths) if lengths else 0,
            avg_session_length=round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
        )
        return cls(sequences=sequences, summary=summary)


def iter_access_log_records(
    path: str | Path, *, max_records: int | None = None
) -> Iterable[AccessLogRecord]:
    source = Path(path)
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        yielded = 0
        for line in handle:
            record = parse_access_log_line(line)
            if record is None:
                continue
            yield record
            yielded += 1
            if max_records is not None and yielded >= max_records:
                break


def parse_access_log_line(line: str) -> AccessLogRecord | None:
    match = _ACCESS_LOG_PATTERN.match(line.strip())
    if match is None:
        return None
    request_line = match.group("request")
    parts = request_line.split()
    if len(parts) < 2:
        return None
    method = parts[0]
    target = parts[1]
    parsed_target = urlsplit(target)
    path = parsed_target.path or "/"
    query = parsed_target.query
    remote_addr = match.group("remote")
    user_agent = match.group("user_agent") or ""
    headers = {"User-Agent": user_agent} if user_agent else {}
    if parsed_target.netloc:
        headers["Host"] = parsed_target.netloc
    request = HttpRequest(
        method=method,
        path=path,
        query=query,
        headers=headers,
        client_ip=remote_addr,
    )
    return AccessLogRecord(
        request=request,
        remote_addr=remote_addr,
        timestamp=match.group("timestamp"),
        status=_parse_status(match.group("status")),
        session_key=_session_key(remote_addr, user_agent),
    )


def _group_by_session(records: Iterable[AccessLogRecord]) -> dict[str, list[HttpRequest]]:
    grouped: dict[str, list[HttpRequest]] = {}
    for record in records:
        grouped.setdefault(record.session_key, []).append(record.request)
    return grouped


def _status_counts(records: Iterable[AccessLogRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        family = "unknown" if record.status is None else f"{record.status // 100}xx"
        counts[family] += 1
    return dict(sorted(counts.items()))


def _session_key(remote_addr: str, user_agent: str) -> str:
    identity = f"{remote_addr}\n{user_agent}".encode("utf-8", errors="replace")
    return hashlib.sha256(identity).hexdigest()[:16]


def _path_fingerprint(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8", errors="replace")).hexdigest()[:16]


def _parse_status(raw: str) -> int | None:
    if raw == "-":
        return None
    return int(raw)


def _count_lines(path: Path, *, max_records: int | None) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for count, _ in enumerate(handle, start=1):
            if max_records is not None and count >= max_records:
                return count
    return count
