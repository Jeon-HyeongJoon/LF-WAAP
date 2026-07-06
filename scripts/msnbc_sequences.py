"""Utilities for bounded, streaming access to the MSNBC sequence dataset."""

from __future__ import annotations

import gzip
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "learning_data"
    / "msnbc_com_anonymous_web_data"
    / "msnbc990928.seq"
)

CATEGORIES: tuple[str, ...] = (
    "frontpage",
    "news",
    "tech",
    "local",
    "opinion",
    "on-air",
    "misc",
    "weather",
    "msn-news",
    "health",
    "living",
    "business",
    "msn-sports",
    "sports",
    "summary",
    "bbs",
    "travel",
)


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    user_index: int
    states: tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.states)


def resolve_sequence_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        direct = candidate / candidate.name
        if direct.is_file():
            return direct
        seq_files = sorted(candidate.glob("*.seq"))
        if seq_files:
            return seq_files[0]
        gz_files = sorted(candidate.glob("*.seq.gz"))
        if gz_files:
            return gz_files[0]
    gz_candidate = candidate.with_suffix(candidate.suffix + ".gz")
    if gz_candidate.is_file():
        return gz_candidate
    raise FileNotFoundError(f"cannot find MSNBC sequence file at {candidate}")


def iter_sequences(
    path: str | Path = DEFAULT_DATASET,
    *,
    max_users: int | None = 50_000,
) -> Iterator[SequenceRecord]:
    """Yield parsed user sequences without reading the whole dataset into memory."""

    sequence_path = resolve_sequence_path(path)
    opener = gzip.open if sequence_path.suffix == ".gz" else open
    yielded = 0
    with opener(sequence_path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if max_users is not None and yielded >= max_users:
                break
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            tokens = line.split()
            if not all(token.isdigit() for token in tokens):
                continue
            states = tuple(_state_from_token(token) for token in tokens)
            if not states:
                continue
            yielded += 1
            yield SequenceRecord(yielded, states)


def reservoir_sample(
    records: Iterable[SequenceRecord],
    *,
    sample_size: int,
    seed: int,
) -> list[SequenceRecord]:
    rng = random.Random(seed)
    sample: list[SequenceRecord] = []
    for seen, record in enumerate(records, start=1):
        if len(sample) < sample_size:
            sample.append(record)
            continue
        index = rng.randint(1, seen)
        if index <= sample_size:
            sample[index - 1] = record
    return sample


def split_records(
    records: Sequence[SequenceRecord],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[list[SequenceRecord], list[SequenceRecord], list[SequenceRecord]]:
    if train_ratio <= 0.0 or validation_ratio < 0.0 or train_ratio + validation_ratio >= 1.0:
        raise ValueError("ratios must satisfy train > 0, validation >= 0, train + validation < 1")
    train_end = int(len(records) * train_ratio)
    validation_end = train_end + int(len(records) * validation_ratio)
    return (
        list(records[:train_end]),
        list(records[train_end:validation_end]),
        list(records[validation_end:]),
    )


def _state_from_token(token: str) -> str:
    try:
        index = int(token)
    except ValueError:
        return f"unknown:{token}"
    if 1 <= index <= len(CATEGORIES):
        return CATEGORIES[index - 1]
    return f"unknown:{index}"
