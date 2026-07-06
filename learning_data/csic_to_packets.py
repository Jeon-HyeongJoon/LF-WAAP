"""CSIC 2010 HTTP dataset -> reconstructed network packets (byte sequences).

Each CSV row is one HTTP request flattened into columns. This module streams the
file row-by-row (never loading it whole), reconstructs the original raw HTTP
request bytes, and yields them as a byte sequence suitable for byte-level models.

Design notes:
  * The header row (line 1) is read at runtime to map columns by NAME, so the
    code does not depend on a fixed column order. Unknown columns are ignored.
  * The file is read with latin-1 (bytes map 1:1, never raises) and re-encoded
    with latin-1, so the reconstructed packet is byte-faithful.
  * Streaming generator: rows are processed one at a time until EOF.

CLI:
    python learning_data/csic_to_packets.py [csv_path] [--limit N] [--show K]
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

# normalized-header -> semantic field. Normalization strips case and separators,
# so "Accept-encoding", "accept_encoding", "ACCEPTENCODING" all collapse together.
_COLUMN_ALIASES: dict[str, str] = {
    "method": "method",
    "useragent": "user_agent",
    "pragma": "pragma",
    "cachecontrol": "cache_control",
    "accept": "accept",
    "acceptencoding": "accept_encoding",
    "acceptcharset": "accept_charset",
    "language": "accept_language",
    "acceptlanguage": "accept_language",
    "host": "host",
    "cookie": "cookie",
    "contenttype": "content_type",
    "connection": "connection",
    "lenght": "content_length",  # dataset's well-known typo
    "length": "content_length",
    "contentlength": "content_length",
    "content": "body",
    "body": "body",
    "payload": "body",
    "url": "url",
    "classification": "label",
    "class": "label",
    "label": "label",
}

# Emitted in this order; only non-empty headers are included.
_HEADER_ORDER: list[tuple[str, str]] = [
    ("host", "Host"),
    ("user_agent", "User-Agent"),
    ("pragma", "Pragma"),
    ("cache_control", "Cache-Control"),
    ("accept", "Accept"),
    ("accept_encoding", "Accept-Encoding"),
    ("accept_charset", "Accept-Charset"),
    ("accept_language", "Accept-Language"),
    ("cookie", "Cookie"),
    ("content_type", "Content-Type"),
    ("connection", "Connection"),
    ("content_length", "Content-Length"),
]


_NORMAL_LABELS = {"normal", "0"}
_ANOMALOUS_LABELS = {"anomalous", "anomaly", "1"}


@dataclass(frozen=True, slots=True)
class PacketRecord:
    label: str  # raw label cell: "Normal"/"Anomalous" or "0"/"1" depending on CSV variant
    method: str
    host: str
    raw: bytes  # the reconstructed HTTP request packet

    @property
    def byte_sequence(self) -> list[int]:
        """The packet as a sequence of byte values 0-255."""
        return list(self.raw)

    @property
    def is_normal(self) -> bool:
        """True for benign traffic. Handles both string and numeric label variants."""
        return self.label.strip().lower() in _NORMAL_LABELS

    @property
    def is_anomalous(self) -> bool:
        return self.label.strip().lower() in _ANOMALOUS_LABELS


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _strip_known_prefix(value: str, header_name: str) -> str:
    """Some cells embed the header name, e.g. 'Connection: close'. Strip only the
    matching prefix (never a generic 'word:' — that would corrupt 'localhost:8080')."""
    m = re.match(rf"^\s*{re.escape(header_name)}\s*:\s*", value, re.IGNORECASE)
    return value[m.end() :] if m else value


def _split_url(url_field: str) -> tuple[str, str]:
    """'http://host:8080/a/b.jsp?x=1 HTTP/1.1' -> ('/a/b.jsp?x=1', 'HTTP/1.1')."""
    url_field = url_field.strip()
    version = "HTTP/1.1"
    head, _, tail = url_field.rpartition(" ")
    if head and tail.upper().startswith("HTTP/"):
        url_field, version = head, tail
    parts = urlsplit(url_field)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return path, version


def _build_field_map(header: list[str]) -> dict[str, int]:
    """Map semantic field name -> column index, using the header row."""
    field_to_index: dict[str, int] = {}
    for i, col in enumerate(header):
        semantic = _COLUMN_ALIASES.get(_normalize(col))
        if semantic is not None and semantic not in field_to_index:
            field_to_index[semantic] = i
    return field_to_index


def _reconstruct(fields: dict[str, str]) -> bytes:
    method = (fields.get("method") or "GET").strip().upper()
    path, version = _split_url(fields.get("url", ""))

    body = fields.get("body", "")
    content_length = _strip_known_prefix(fields.get("content_length", ""), "Content-Length").strip()
    if not content_length and body:
        content_length = str(len(body.encode("latin-1")))

    lines = [f"{method} {path} {version}"]
    for key, header_name in _HEADER_ORDER:
        if key == "content_length":
            value = content_length
        else:
            value = _strip_known_prefix(fields.get(key, ""), header_name).strip()
        if value:
            lines.append(f"{header_name}: {value}")

    raw = "\r\n".join(lines) + "\r\n\r\n" + body
    return raw.encode("latin-1", errors="backslashreplace")


def iter_packets(csv_path: str | Path, limit: int | None = None) -> Iterator[PacketRecord]:
    """Stream the CSV from the second line to EOF, yielding one packet per row."""
    with open(csv_path, encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)  # line 1: column names
        except StopIteration:
            return
        fmap = _build_field_map(header)

        for count, row in enumerate(reader):  # from line 2 onward
            if limit is not None and count >= limit:
                break
            fields = {
                field: (row[idx] if idx < len(row) else "") for field, idx in fmap.items()
            }
            yield PacketRecord(
                label=fields.get("label", "").strip(),
                method=(fields.get("method") or "").strip().upper(),
                host=(fields.get("host") or "").strip(),
                raw=_reconstruct(fields),
            )


def iter_balanced_packets(
    csv_path: str | Path, limit: int | None = None
) -> Iterator[PacketRecord]:
    """평가용으로 두 클래스를 균형 있게 흘려보낸다.

    CSIC 혼합 CSV는 정상이 앞에, 공격이 뒤에 몰려 있다. 단순히 앞에서 limit개만 자르면
    공격이 0개가 되어 지표를 못 낸다(함정). limit이 주어지면 정상·공격을 각각 약 limit/2개씩
    모아 두 클래스가 항상 포함되게 하고, limit=None이면 전체를 그대로 흘려보낸다.
    """
    if limit is None:
        yield from iter_packets(csv_path)
        return
    per_class = max(1, limit // 2)
    n_normal = n_attack = 0
    for record in iter_packets(csv_path):
        if record.is_normal and n_normal < per_class:
            n_normal += 1
            yield record
        elif record.is_anomalous and n_attack < per_class:
            n_attack += 1
            yield record
        if n_normal >= per_class and n_attack >= per_class:
            break  # 두 클래스 할당량을 모두 채웠으면 조기 종료


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct HTTP packets from CSIC 2010 CSV")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(Path(__file__).with_name("csic_database.csv")),
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N rows")
    parser.add_argument("--show", type=int, default=1, help="print the first K packets")
    args = parser.parse_args()

    total = 0
    bytes_total = 0
    for record in iter_packets(args.csv_path, limit=args.limit):
        if total < args.show:
            print(f"--- record {total} [{record.label or '?'}] {len(record.raw)} bytes ---")
            print(record.raw.decode("latin-1"))
            print()
        total += 1
        bytes_total += len(record.raw)

    print(f"reconstructed {total} packets, {bytes_total} bytes total")


if __name__ == "__main__":
    main()
