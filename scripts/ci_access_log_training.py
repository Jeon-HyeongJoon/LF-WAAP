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
from waf.infrastructure.behavior import CanonicalEventMapper  # noqa: E402
from waf.infrastructure.behavior.session_identity import SessionIdentityResolver  # noqa: E402
from waf.infrastructure.behavior.workflow_detector import (  # noqa: E402
    EnforcementMode,
    WorkflowBehaviorDetector,
    WorkflowTransitionModel,
)
from waf.infrastructure.runtime import OperationalWaf, WafRuntimeConfig  # noqa: E402

TARGET_FPR_CANDIDATES = (0.01, 0.02, 0.05)
MAX_SELECTED_NORMAL_ALERT_RATE = 0.02


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
    profiles: list[dict[str, object]] = []
    selected_detector: WorkflowBehaviorDetector | None = None
    selected_before_ready = False
    selected_profile: dict[str, object] | None = None
    for target_fpr in TARGET_FPR_CANDIDATES:
        detector = _new_detector(target_fpr=target_fpr)
        before_ready = detector.is_trained
        detector.train_request_sequences(train_sequences, validation_sequences)
        profile = _profile_detector(detector, validation_sequences, target_fpr)
        profiles.append(profile)
        if selected_profile is None or _is_better_profile(profile, selected_profile):
            selected_detector = detector
            selected_before_ready = before_ready
            selected_profile = profile

    if selected_detector is None or selected_profile is None:
        raise RuntimeError("access log workflow training did not produce a candidate model")

    shadow_profile = _best_shadow_profile(profiles)
    detector = selected_detector
    before_ready = selected_before_ready
    after_ready = detector.is_trained
    snapshot = detector.export_model_snapshot()
    snapshot_path = output_dir / "access_log_workflow_model.json"
    snapshot.save(snapshot_path)

    validation_assessments = selected_profile["validation"]
    runtime_boot = _verify_runtime_boot(snapshot_path, snapshot.fingerprint)
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
            "target_fpr": selected_profile["target_fpr"],
            "fingerprint": snapshot.fingerprint,
            "artifact_path": str(snapshot_path),
        },
        "validation": validation_assessments,
        "calibration": {
            "candidate_target_fprs": list(TARGET_FPR_CANDIDATES),
            "max_selected_normal_alert_rate": MAX_SELECTED_NORMAL_ALERT_RATE,
            "selected_target_fpr": selected_profile["target_fpr"],
            "shadow_target_fpr": shadow_profile["target_fpr"],
            "profiles": profiles,
        },
        "operational_assessment": _operational_assessment(
            selected_profile=selected_profile,
            shadow_profile=shadow_profile,
        ),
        "runtime_boot": runtime_boot,
    }
    if not after_ready or detector.action_count <= 0 or detector.transition_count <= 0:
        _write_outputs(output_dir, metrics)
        raise RuntimeError("access log workflow training did not produce a ready model")
    _write_outputs(output_dir, metrics)
    return metrics


def _new_detector(*, target_fpr: float) -> WorkflowBehaviorDetector:
    mapper = CanonicalEventMapper(
        tenant_id="ci",
        service_id="access-log",
        session_resolver=SessionIdentityResolver(include_user_agent_in_ip_fallback=True),
    )
    return WorkflowBehaviorDetector(
        mapper=mapper,
        mode=EnforcementMode.ALERT,
        model=WorkflowTransitionModel(target_fpr=target_fpr),
    )


def _profile_detector(
    detector: WorkflowBehaviorDetector,
    validation_sequences: tuple[tuple[HttpRequest, ...], ...],
    target_fpr: float,
) -> dict[str, object]:
    return {
        "target_fpr": target_fpr,
        "threshold": detector.threshold,
        "action_count": detector.action_count,
        "transition_count": detector.transition_count,
        "validation": _assess_validation(detector, validation_sequences),
    }


def _is_better_profile(
    candidate: dict[str, object],
    incumbent: dict[str, object],
) -> bool:
    candidate_key = _profile_selection_key(candidate)
    incumbent_key = _profile_selection_key(incumbent)
    return candidate_key > incumbent_key


def _profile_selection_key(profile: dict[str, object]) -> tuple[int, float, float, float]:
    validation = profile["validation"]
    assert isinstance(validation, dict)
    normal_alert_rate = float(validation["alert_rate"])
    within_budget = int(normal_alert_rate <= MAX_SELECTED_NORMAL_ALERT_RATE)
    return (
        within_budget,
        float(validation["alert_rate_lift"]),
        float(validation["score_separation"]),
        -normal_alert_rate,
    )


def _best_shadow_profile(profiles: list[dict[str, object]]) -> dict[str, object]:
    return max(
        profiles,
        key=lambda profile: (
            _profile_validation_float(profile, "alert_rate_lift"),
            _profile_validation_float(profile, "score_separation"),
        ),
    )


def _operational_assessment(
    *,
    selected_profile: dict[str, object],
    shadow_profile: dict[str, object],
) -> dict[str, object]:
    selected_validation = selected_profile["validation"]
    shadow_validation = shadow_profile["validation"]
    assert isinstance(selected_validation, dict)
    assert isinstance(shadow_validation, dict)
    selected_lift = float(selected_validation["alert_rate_lift"])
    selected_separation = float(selected_validation["score_separation"])
    shadow_lift = float(shadow_validation["alert_rate_lift"])
    return {
        "serving_target_fpr": selected_profile["target_fpr"],
        "serving_normal_alert_rate": selected_validation["alert_rate"],
        "serving_counterfactual_alert_rate": _counterfactual_alert_rate(
            selected_validation
        ),
        "serving_alert_rate_lift": selected_lift,
        "serving_score_separation": selected_separation,
        "shadow_target_fpr": shadow_profile["target_fpr"],
        "shadow_normal_alert_rate": shadow_validation["alert_rate"],
        "shadow_counterfactual_alert_rate": _counterfactual_alert_rate(
            shadow_validation
        ),
        "shadow_alert_rate_lift": shadow_lift,
        "needs_labeled_attack_dataset": selected_lift <= 0.0,
        "recommendation": _recommendation(selected_lift, shadow_lift),
    }


def _counterfactual_alert_rate(validation: dict[str, object]) -> float:
    counterfactual = validation["counterfactual"]
    assert isinstance(counterfactual, dict)
    return float(counterfactual["alert_rate"])


def _profile_validation_float(profile: dict[str, object], name: str) -> float:
    validation = profile["validation"]
    assert isinstance(validation, dict)
    return float(validation[name])


def _recommendation(selected_lift: float, shadow_lift: float) -> str:
    if selected_lift > 0.0:
        return "serving_profile_has_counterfactual_sensitivity"
    if shadow_lift > 0.0:
        return "keep_serving_profile_conservative_and_validate_shadow_profile_with_labeled_attacks"
    return "collect_labeled_attack_or_incident_dataset_before_tightening_enforcement"


def _verify_runtime_boot(snapshot_path: Path, expected_fingerprint: str) -> dict[str, object]:
    waf = OperationalWaf.from_config(
        WafRuntimeConfig(
            tenant_id="ci",
            service_id="access-log",
            runtime_version="ci-access-log",
            workflow_mode=EnforcementMode.ALERT,
            workflow_model_path=snapshot_path,
            enable_ruleset=False,
        )
    )
    status = waf.status().to_dict()
    detectors = status["detectors"]
    assert isinstance(detectors, dict)
    workflow = detectors["workflow"]
    assert isinstance(workflow, dict)
    metadata = workflow["metadata"]
    assert isinstance(metadata, dict)
    model_fingerprint = metadata["model_fingerprint"]
    return {
        "ready": status["ready"],
        "serving": status["serving"],
        "serving_detectors": status["serving_detectors"],
        "degraded_detectors": status["degraded_detectors"],
        "runtime_version": status["runtime_version"],
        "model_fingerprint": model_fingerprint,
        "model_fingerprint_matched": model_fingerprint == expected_fingerprint,
    }


def _assess_validation(
    detector: WorkflowBehaviorDetector,
    validation_sequences: tuple[tuple[HttpRequest, ...], ...],
) -> dict[str, object]:
    normal = _assess_request_sequences(detector, validation_sequences)
    disrupted_sequences = tuple(
        _disrupt_sequence(sequence)
        for sequence in validation_sequences
        if len(sequence) >= 2
    )
    disrupted = _assess_request_sequences(detector, disrupted_sequences)
    normal_avg = float(normal["avg_transition_score"])
    disrupted_avg = float(disrupted["avg_transition_score"])
    normal_alert_rate = float(normal["alert_rate"])
    disrupted_alert_rate = float(disrupted["alert_rate"])
    return {
        **normal,
        "counterfactual": disrupted,
        "score_separation": round(normal_avg - disrupted_avg, 6),
        "alert_rate_lift": round(disrupted_alert_rate - normal_alert_rate, 6),
    }


def _assess_request_sequences(
    detector: WorkflowBehaviorDetector,
    sequences: tuple[tuple[HttpRequest, ...], ...],
) -> dict[str, object]:
    inspected = 0
    alerts = 0
    covered = 0
    scores: list[float] = []
    for sequence in sequences:
        events = [detector.map_request(request) for request in sequence]
        for event in events:
            detector.reset_session(event.session_id)
        for event in events:
            result = detector.assess_event(event)
            inspected += 1
            alerts += int(result.blocked)
            covered += int(result.covered)
            scores.append(result.score)
    return {
        "inspected_requests": inspected,
        "covered_requests": covered,
        "coverage_rate": round(covered / inspected, 6) if inspected else 0.0,
        "alerts": alerts,
        "alert_rate": round(alerts / inspected, 6) if inspected else 0.0,
        "avg_transition_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "min_transition_score": round(min(scores), 6) if scores else 0.0,
    }


def _disrupt_sequence(sequence: tuple[HttpRequest, ...]) -> tuple[HttpRequest, ...]:
    return tuple(reversed(sequence))


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
    calibration = metrics.get("calibration", {})
    operational = metrics.get("operational_assessment", {})
    runtime_boot = metrics.get("runtime_boot", {})
    assert isinstance(dataset, dict)
    assert isinstance(training, dict)
    assert isinstance(model, dict)
    assert isinstance(validation, dict)
    assert isinstance(calibration, dict)
    assert isinstance(operational, dict)
    assert isinstance(runtime_boot, dict)
    counterfactual = validation.get("counterfactual", {})
    assert isinstance(counterfactual, dict)
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
            f"- selected_target_fpr: {calibration.get('selected_target_fpr', 0.0)}",
            f"- shadow_target_fpr: {calibration.get('shadow_target_fpr', 0.0)}",
            f"- max_selected_normal_alert_rate: "
            f"{calibration.get('max_selected_normal_alert_rate', 0.0)}",
            f"- action_count: {model.get('action_count', 0)}",
            f"- transition_count: {model.get('transition_count', 0)}",
            f"- model_target_fpr: {model.get('target_fpr', 0.0)}",
            f"- model_fingerprint: {model.get('fingerprint', '')}",
            f"- model_artifact_path: {model.get('artifact_path', '')}",
            f"- validation_inspected_requests: {validation.get('inspected_requests', 0)}",
            f"- validation_coverage_rate: {validation.get('coverage_rate', 0.0)}",
            f"- validation_alert_rate: {validation.get('alert_rate', 0.0)}",
            f"- validation_avg_transition_score: {validation.get('avg_transition_score', 0.0)}",
            f"- validation_min_transition_score: {validation.get('min_transition_score', 0.0)}",
            f"- validation_score_separation: {validation.get('score_separation', 0.0)}",
            f"- validation_alert_rate_lift: {validation.get('alert_rate_lift', 0.0)}",
            f"- counterfactual_inspected_requests: "
            f"{counterfactual.get('inspected_requests', 0)}",
            f"- counterfactual_alert_rate: {counterfactual.get('alert_rate', 0.0)}",
            f"- counterfactual_avg_transition_score: "
            f"{counterfactual.get('avg_transition_score', 0.0)}",
            f"- shadow_alert_rate_lift: {operational.get('shadow_alert_rate_lift', 0.0)}",
            f"- shadow_counterfactual_alert_rate: "
            f"{operational.get('shadow_counterfactual_alert_rate', 0.0)}",
            f"- needs_labeled_attack_dataset: "
            f"{operational.get('needs_labeled_attack_dataset', False)}",
            f"- recommendation: {operational.get('recommendation', '')}",
            f"- runtime_boot_ready: {runtime_boot.get('ready', False)}",
            f"- runtime_boot_serving: {runtime_boot.get('serving', False)}",
            f"- runtime_boot_fingerprint_matched: "
            f"{runtime_boot.get('model_fingerprint_matched', False)}",
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
