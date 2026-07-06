"""Train and evaluate the Markov flow model on bounded MSNBC sequence samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from msnbc_sequences import DEFAULT_DATASET, SequenceRecord, iter_sequences, split_records
from waf.infrastructure.behavior.markov_flow import START_STATE, MarkovFlowModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--max-users", type=int, default=50_000)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output-dir", default="reports/msnbc")
    args = parser.parse_args()

    records = list(iter_sequences(args.dataset, max_users=args.max_users))
    if len(records) < 10:
        raise SystemExit("need at least 10 sequences to train/evaluate")
    train, validation, test = split_records(records)

    model = MarkovFlowModel(target_fpr=args.target_fpr, alpha=args.alpha)
    model.fit([record.states for record in train], [record.states for record in validation])
    metrics, fp_samples = evaluate_model(model, test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "read_policy": {
            "max_users": args.max_users,
            "note": "bounded streaming read; the full dataset is not consumed by default",
        },
        "split": {
            "train_sequences": len(train),
            "validation_sequences": len(validation),
            "test_sequences": len(test),
        },
        "model": {
            "target_fpr": args.target_fpr,
            "alpha": args.alpha,
            "threshold": model.threshold,
            "states": model.states,
        },
        "metrics": metrics,
    }
    (output_dir / "markov_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_samples(output_dir / "fp_samples.jsonl", fp_samples)
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {output_dir / 'markov_metrics.json'}")
    print(f"wrote {output_dir / 'fp_samples.jsonl'}")


def evaluate_model(
    model: MarkovFlowModel,
    test: list[SequenceRecord],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    transitions = 0
    blocked = 0
    blocked_by_transition: Counter[tuple[str, str]] = Counter()
    fp_samples: list[dict[str, object]] = []
    for record in test:
        previous = START_STATE
        for position, current in enumerate(record.states):
            result = model.assess_transition(previous, current)
            transitions += 1
            if result.blocked:
                blocked += 1
                blocked_by_transition[(previous, current)] += 1
                if len(fp_samples) < 100:
                    fp_samples.append(
                        {
                            "user_index": record.user_index,
                            "position": position,
                            "previous": previous,
                            "current": current,
                            "score": result.score,
                            "threshold": model.threshold,
                            "sequence": record.states,
                        }
                    )
            previous = current
    fp_rate = blocked / transitions if transitions else 0.0
    return (
        {
            "test_transitions": transitions,
            "blocked_transitions": blocked,
            "estimated_fp_rate_on_heldout_normal": fp_rate,
            "top_blocked_transitions": [
                {"from": src, "to": dst, "count": count}
                for (src, dst), count in blocked_by_transition.most_common(20)
            ],
        },
        fp_samples,
    )


def _write_samples(path: Path, samples: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
