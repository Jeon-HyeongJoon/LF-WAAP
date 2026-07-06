"""Create LLM-friendly profiles and deterministic samples for MSNBC sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from msnbc_sequences import DEFAULT_DATASET, SequenceRecord, iter_sequences, reservoir_sample


@dataclass(frozen=True, slots=True)
class LengthSummary:
    minimum: int
    p50: int
    p90: int
    p95: int
    p99: int
    maximum: int
    average: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--max-users", type=int, default=50_000)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--edge-sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", default="reports/data_cards")
    parser.add_argument("--sample-dir", default="reports/samples")
    args = parser.parse_args()

    records = list(iter_sequences(args.dataset, max_users=args.max_users))
    if not records:
        raise SystemExit("no MSNBC records parsed")

    state_counts: Counter[str] = Counter()
    transition_counts: Counter[tuple[str, str]] = Counter()
    lengths: list[int] = []
    for record in records:
        lengths.append(record.length)
        state_counts.update(record.states)
        transition_counts.update(zip(record.states, record.states[1:]))

    sampled = reservoir_sample(
        iter_sequences(args.dataset, max_users=args.max_users),
        sample_size=min(args.sample_size, len(records)),
        seed=args.seed,
    )
    edge_samples = _edge_samples(records, limit=args.edge_sample_size)

    report_dir = Path(args.report_dir)
    sample_dir = Path(args.sample_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "dataset": str(args.dataset),
        "read_policy": {
            "max_users": args.max_users,
            "note": "bounded streaming read; the full dataset is not loaded unless max_users is raised",
        },
        "row_count_read": len(records),
        "event_count": sum(lengths),
        "length_summary": asdict(_length_summary(lengths)),
        "state_top_k": state_counts.most_common(20),
        "transition_top_k": [
            {"from": src, "to": dst, "count": count}
            for (src, dst), count in transition_counts.most_common(30)
        ],
    }
    (report_dir / "msnbc_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "msnbc.md").write_text(_markdown(profile), encoding="utf-8")
    _write_jsonl(sample_dir / "msnbc_sample.jsonl", sampled)
    _write_jsonl(sample_dir / "msnbc_edge_cases.jsonl", edge_samples)
    print(f"wrote {report_dir / 'msnbc.md'}")
    print(f"wrote {report_dir / 'msnbc_profile.json'}")
    print(f"wrote {sample_dir / 'msnbc_sample.jsonl'}")
    print(f"wrote {sample_dir / 'msnbc_edge_cases.jsonl'}")


def _length_summary(lengths: list[int]) -> LengthSummary:
    ordered = sorted(lengths)
    return LengthSummary(
        minimum=ordered[0],
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
        average=sum(ordered) / len(ordered),
    )


def _percentile(ordered: list[int], q: float) -> int:
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _edge_samples(records: list[SequenceRecord], *, limit: int) -> list[SequenceRecord]:
    return sorted(records, key=lambda record: record.length, reverse=True)[:limit]


def _write_jsonl(path: Path, records: list[SequenceRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# MSNBC Sequence Data Card",
        "",
        f"- dataset: `{profile['dataset']}`",
        f"- row_count_read: `{profile['row_count_read']}`",
        f"- event_count: `{profile['event_count']}`",
        "- read_policy: bounded streaming read; increase `--max-users` only when needed.",
        "",
        "## Session Length",
        "",
        "```json",
        json.dumps(profile["length_summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## State Top-K",
        "",
        "| state | count |",
        "| --- | ---: |",
    ]
    for state, count in profile["state_top_k"]:
        lines.append(f"| {state} | {count} |")
    lines.extend(["", "## Transition Top-K", "", "| from | to | count |", "| --- | --- | ---: |"])
    for row in profile["transition_top_k"]:
        lines.append(f"| {row['from']} | {row['to']} | {row['count']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
