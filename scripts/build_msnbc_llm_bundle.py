"""Build the bounded MSNBC artifacts that should be handed to an LLM."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from msnbc_sequences import DEFAULT_DATASET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--max-users", type=int, default=50_000)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--edge-sample-size", type=int, default=100)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", default="reports/data_cards")
    parser.add_argument("--sample-dir", default="reports/samples")
    parser.add_argument("--model-report-dir", default="reports/msnbc")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    commands = [
        [
            sys.executable,
            str(root / "scripts" / "profile_msnbc_data.py"),
            "--dataset",
            args.dataset,
            "--max-users",
            str(args.max_users),
            "--sample-size",
            str(args.sample_size),
            "--edge-sample-size",
            str(args.edge_sample_size),
            "--seed",
            str(args.seed),
            "--report-dir",
            args.report_dir,
            "--sample-dir",
            args.sample_dir,
        ],
        [
            sys.executable,
            str(root / "scripts" / "train_msnbc_markov.py"),
            "--dataset",
            args.dataset,
            "--max-users",
            str(args.max_users),
            "--target-fpr",
            str(args.target_fpr),
            "--output-dir",
            args.model_report_dir,
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)

    print("LLM bundle ready:")
    print(f"- {Path(args.report_dir) / 'msnbc.md'}")
    print(f"- {Path(args.report_dir) / 'msnbc_profile.json'}")
    print(f"- {Path(args.sample_dir) / 'msnbc_sample.jsonl'}")
    print(f"- {Path(args.sample_dir) / 'msnbc_edge_cases.jsonl'}")
    print(f"- {Path(args.model_report_dir) / 'markov_metrics.json'}")
    print(f"- {Path(args.model_report_dir) / 'fp_samples.jsonl'}")


if __name__ == "__main__":
    main()
