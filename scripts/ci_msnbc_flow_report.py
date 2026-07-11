"""Train a Markov flow model on MSNBC sequences and emit CI flow metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from msnbc_sequences import SequenceRecord, iter_sequences, split_records  # noqa: E402
from waf.infrastructure.behavior.markov_flow import MarkovFlowModel  # noqa: E402

DEFAULT_MSNBC_DATASET = (
    ROOT
    / "learning_data"
    / "msnbc_com_anonymous_web_data"
    / "msnbc990928.seq"
    / "msnbc990928.seq"
)
TARGET_FPR = 0.01
ALPHA = 0.1
RARE_FLOW_PROBABILITY_THRESHOLD = 0.001
FLOW_MAP_LIMIT = 100
DEFAULT_MAX_USERS = 10_000


def run(
    dataset_path: Path,
    output_dir: Path,
    *,
    max_users: int | None,
    flow_map_limit: int = FLOW_MAP_LIMIT,
) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_sequences(dataset_path, max_users=max_users))
    if len(records) < 10:
        metrics = {
            "dataset_present": dataset_path.is_file(),
            "dataset_path": str(dataset_path),
            "trained": False,
            "reason": "need at least 10 MSNBC sequences to train/evaluate",
        }
        _write_outputs(output_dir, metrics)
        raise RuntimeError(metrics["reason"])

    train, validation, test = split_records(records)
    model = MarkovFlowModel(target_fpr=TARGET_FPR, alpha=ALPHA)
    model.fit([record.states for record in train], [record.states for record in validation])
    flow_distribution = _build_flow_distribution(
        records,
        rare_probability_threshold=RARE_FLOW_PROBABILITY_THRESHOLD,
        flow_map_limit=flow_map_limit,
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "dataset_present": True,
        "trained": model.is_trained,
        "elapsed_seconds": round(elapsed, 3),
        "read_policy": {
            "max_users": max_users,
            "note": "bounded streaming read; raw sequences are not printed",
        },
        "split": {
            "total_sequences": len(records),
            "train_sequences": len(train),
            "validation_sequences": len(validation),
            "test_sequences": len(test),
        },
        "model": {
            "target_fpr": TARGET_FPR,
            "alpha": ALPHA,
            "threshold": model.threshold,
            "state_count": len(model.states),
        },
        "flow_distribution": flow_distribution,
    }
    if not model.is_trained or model.threshold is None:
        _write_outputs(output_dir, metrics)
        raise RuntimeError("MSNBC Markov flow training did not produce a threshold")
    _write_outputs(output_dir, metrics)
    return metrics


def _build_flow_distribution(
    records: list[SequenceRecord],
    *,
    rare_probability_threshold: float,
    flow_map_limit: int,
) -> dict[str, object]:
    counts: Counter[tuple[str, ...]] = Counter(record.states for record in records)
    total = sum(counts.values())
    all_rows: list[dict[str, object]] = []
    for index, (states, count) in enumerate(
        sorted(counts.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        probability = count / total if total else 0.0
        all_rows.append(
            {
                "flow_type_id": f"msnbc_flow_{index:06d}",
                "flow_hash": _flow_hash(states),
                "count": count,
                "probability": round(probability, 6),
                "is_rare": probability <= rare_probability_threshold,
                "step_count": len(states),
                "states": list(states),
            }
        )
    rare_rows = [row for row in all_rows if bool(row["is_rare"])]
    return {
        "total_flows": total,
        "unique_flow_types": len(all_rows),
        "rare_flow_probability_threshold": rare_probability_threshold,
        "rare_flow_types": len(rare_rows),
        "rare_flow_occurrences": sum(int(row["count"]) for row in rare_rows),
        "rare_probability_mass": round(
            sum(float(row["probability"]) for row in rare_rows),
            6,
        ),
        "flow_type_map_limit": flow_map_limit,
        "flow_type_map": all_rows[:flow_map_limit],
    }


def _flow_hash(states: tuple[str, ...]) -> str:
    encoded = "\n".join(states).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_outputs(output_dir: Path, metrics: dict[str, object]) -> None:
    (output_dir / "msnbc_flow_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "msnbc_flow_summary.md").write_text(
        render_summary(metrics),
        encoding="utf-8",
    )


def render_summary(metrics: dict[str, object]) -> str:
    if not metrics.get("dataset_present"):
        return "\n".join(
            [
                "# MSNBC Flow Training",
                "",
                "- dataset_present: false",
                f"- reason: {metrics['reason']}",
                "",
            ]
        )
    read_policy = metrics.get("read_policy", {})
    split = metrics.get("split", {})
    model = metrics.get("model", {})
    flow_distribution = metrics.get("flow_distribution", {})
    assert isinstance(read_policy, dict)
    assert isinstance(split, dict)
    assert isinstance(model, dict)
    assert isinstance(flow_distribution, dict)
    flow_type_map = flow_distribution.get("flow_type_map", [])
    assert isinstance(flow_type_map, list)
    return "\n".join(
        [
            "# MSNBC Flow Training",
            "",
            f"- dataset_present: {metrics['dataset_present']}",
            f"- trained: {metrics.get('trained', False)}",
            f"- elapsed_seconds: {metrics.get('elapsed_seconds', 0.0)}",
            f"- max_users: {read_policy.get('max_users', '')}",
            f"- total_sequences: {split.get('total_sequences', 0)}",
            f"- train_sequences: {split.get('train_sequences', 0)}",
            f"- validation_sequences: {split.get('validation_sequences', 0)}",
            f"- test_sequences: {split.get('test_sequences', 0)}",
            f"- target_fpr: {model.get('target_fpr', 0.0)}",
            f"- alpha: {model.get('alpha', 0.0)}",
            f"- threshold: {model.get('threshold', '')}",
            f"- state_count: {model.get('state_count', 0)}",
            f"- flow_total: {flow_distribution.get('total_flows', 0)}",
            f"- flow_type_count: {flow_distribution.get('unique_flow_types', 0)}",
            f"- rare_flow_probability_threshold: "
            f"{flow_distribution.get('rare_flow_probability_threshold', 0.0)}",
            f"- rare_flow_types: {flow_distribution.get('rare_flow_types', 0)}",
            f"- rare_flow_occurrences: "
            f"{flow_distribution.get('rare_flow_occurrences', 0)}",
            f"- rare_flow_probability_mass: "
            f"{flow_distribution.get('rare_probability_mass', 0.0)}",
            f"- flow_type_map_limit: {flow_distribution.get('flow_type_map_limit', 0)}",
            *_render_flow_type_map(flow_type_map),
            "",
        ]
    )


def _render_flow_type_map(flow_type_map: list[object]) -> list[str]:
    lines = ["- flow_type_map:"]
    for row in flow_type_map:
        assert isinstance(row, dict)
        lines.append(
            "  - "
            f"{row.get('flow_type_id', '')}: "
            f"count={row.get('count', 0)}, "
            f"probability={row.get('probability', 0.0)}, "
            f"is_rare={row.get('is_rare', False)}, "
            f"step_count={row.get('step_count', 0)}, "
            f"flow_hash={row.get('flow_hash', '')}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_MSNBC_DATASET))
    parser.add_argument("--output-dir", default="reports/ci/msnbc-flow")
    parser.add_argument("--max-users", type=int, default=DEFAULT_MAX_USERS)
    parser.add_argument("--flow-map-limit", type=int, default=FLOW_MAP_LIMIT)
    args = parser.parse_args()
    metrics = run(
        Path(args.dataset),
        Path(args.output_dir),
        max_users=args.max_users,
        flow_map_limit=args.flow_map_limit,
    )
    print(render_summary(metrics))


if __name__ == "__main__":
    main()
