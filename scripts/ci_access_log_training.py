"""Train workflow behavior from learning_data/access.log and emit CI metrics.

The script deliberately reports aggregate counts and model metadata only. It
does not print raw access log lines or raw URL samples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learning_data"))

from access_log_source import AccessLogTrainingDataset  # noqa: E402
from waf.domain.model.http_request import HttpRequest  # noqa: E402
from waf.domain.model.detection import SignalAction  # noqa: E402
from waf.infrastructure.behavior import CanonicalEventMapper  # noqa: E402
from waf.infrastructure.behavior.workflow_detector import (  # noqa: E402
    EnforcementMode,
    WorkflowBehaviorDetector,
)


def split_sequences(
    sequences: tuple[tuple[HttpRequest, ...], ...],
) -> tuple[tuple[tuple[HttpRequest, ...], ...], tuple[tuple[HttpRequest, ...], ...]]:
    if len(sequences) < 2:
        return sequences, sequences
    validation_count = max(1, len(sequences) // 5)
    split_at = len(sequences) - validation_count
    return sequences[:split_at], sequences[split_at:]


def run(access_log: Path, output_dir: Path, *, max_records: int | None) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not access_log.is_file():
        metrics = {
            "dataset_present": False,
            "access_log_path": str(access_log),
            "reason": "learning_data/access.log is required for CI access-log training",
        }
        _write_outputs(output_dir, metrics)
        raise FileNotFoundError(metrics["reason"])

    dataset = AccessLogTrainingDataset.from_path(access_log, max_records=max_records)
    if not dataset.sequences:
        metrics = {
            "dataset_present": True,
            "trained": False,
            "dataset": dataset.summary.to_dict(),
            "reason": "access log did not contain enough successful multi-request sessions",
        }
        _write_outputs(output_dir, metrics)
        raise RuntimeError(metrics["reason"])

    train_sequences, validation_sequences = split_sequences(dataset.sequences)
    mapper = CanonicalEventMapper(tenant_id="ci", service_id="access-log")
    detector = WorkflowBehaviorDetector(mapper=mapper, mode=EnforcementMode.ALERT)

    before_ready = detector.is_trained
    detector.train_request_sequences(train_sequences, validation_sequences)
    after_ready = detector.is_trained
    snapshot = detector.export_model_snapshot()

    validation_assessments = _assess_validation(detector, validation_sequences)
    elapsed = time.perf_counter() - started
    metrics = {
        "dataset_present": True,
        "trained": after_ready,
        "elapsed_seconds": round(elapsed, 3),
        "dataset": dataset.summary.to_dict(),
        "training": {
            "train_sessions": len(train_sequences),
            "validation_sessions": len(validation_sequences),
            "train_requests": sum(len(sequence) for sequence in train_sequences),
            "validation_requests": sum(len(sequence) for sequence in validation_sequences),
            "baseline_ready": before_ready,
            "trained_ready": after_ready,
        },
        "model": {
            "action_count": detector.action_count,
            "transition_count": detector.transition_count,
            "threshold": detector.threshold,
            "fingerprint": snapshot.fingerprint,
        },
        "validation": validation_assessments,
    }
    if not after_ready or detector.action_count <= 0 or detector.transition_count <= 0:
        _write_outputs(output_dir, metrics)
        raise RuntimeError("access log workflow training did not produce a ready model")
    _write_outputs(output_dir, metrics)
    return metrics


def _assess_validation(
    detector: WorkflowBehaviorDetector,
    validation_sequences: tuple[tuple[HttpRequest, ...], ...],
) -> dict[str, object]:
    inspected = 0
    alerts = 0
    for index, sequence in enumerate(validation_sequences):
        detector.reset_session(f"ci-validation-{index}")
        for request in sequence:
            result = detector.inspect(request)
            inspected += 1
            alerts += int(result.action is not SignalAction.ALLOW)
    return {
        "inspected_requests": inspected,
        "alerts": alerts,
        "alert_rate": round(alerts / inspected, 6) if inspected else 0.0,
    }


def _write_outputs(output_dir: Path, metrics: dict[str, object]) -> None:
    (output_dir / "access_log_training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "access_log_training_summary.md").write_text(
        render_summary(metrics),
        encoding="utf-8",
    )


def render_summary(metrics: dict[str, object]) -> str:
    if not metrics.get("dataset_present"):
        return "\n".join(
            [
                "# Access Log Workflow Training",
                "",
                "- dataset_present: false",
                f"- reason: {metrics['reason']}",
                "",
            ]
        )
    dataset = metrics.get("dataset", {})
    training = metrics.get("training", {})
    model = metrics.get("model", {})
    validation = metrics.get("validation", {})
    assert isinstance(dataset, dict)
    assert isinstance(training, dict)
    assert isinstance(model, dict)
    assert isinstance(validation, dict)
    return "\n".join(
        [
            "# Access Log Workflow Training",
            "",
            f"- dataset_present: {metrics['dataset_present']}",
            f"- trained: {metrics.get('trained', False)}",
            f"- elapsed_seconds: {metrics.get('elapsed_seconds', 0.0)}",
            f"- parsed_records: {dataset.get('parsed_records', 0)}",
            f"- skipped_records: {dataset.get('skipped_records', 0)}",
            f"- training_records: {dataset.get('training_records', 0)}",
            f"- session_count: {dataset.get('session_count', 0)}",
            f"- route_count: {dataset.get('route_count', 0)}",
            f"- avg_session_length: {dataset.get('avg_session_length', 0.0)}",
            f"- train_sessions: {training.get('train_sessions', 0)}",
            f"- validation_sessions: {training.get('validation_sessions', 0)}",
            f"- baseline_ready: {training.get('baseline_ready', False)}",
            f"- trained_ready: {training.get('trained_ready', False)}",
            f"- action_count: {model.get('action_count', 0)}",
            f"- transition_count: {model.get('transition_count', 0)}",
            f"- model_fingerprint: {model.get('fingerprint', '')}",
            f"- validation_inspected_requests: {validation.get('inspected_requests', 0)}",
            f"- validation_alert_rate: {validation.get('alert_rate', 0.0)}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-log", default=str(ROOT / "learning_data" / "access.log"))
    parser.add_argument("--output-dir", default="reports/ci/access-log")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    metrics = run(Path(args.access_log), Path(args.output_dir), max_records=args.max_records)
    print(render_summary(metrics))


if __name__ == "__main__":
    main()
