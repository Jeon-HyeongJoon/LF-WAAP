"""Fast deterministic HMM / Markov+HMM benchmark for GitHub Actions.

The goal is not to replace real dataset evaluation. It gives every branch a
small, reproducible learning/evaluation workload so Actions can expose whether a
change improved or regressed the detector under the same traffic mix.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from waf.domain.model.flow import Direction, Flow
from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior import HybridFlowDetector
from waf.infrastructure.behavior.markov_flow import MarkovFlowModel
from waf.infrastructure.hmm import HmmDetector
from waf.infrastructure.hmm.benchmark import ModelBenchmark, VariantResult, render_comparison
from waf.infrastructure.hmm.evaluation import ClassificationMetrics, EvaluationReport
from waf.infrastructure.hmm.http_flow import http_request_to_flow
from waf.infrastructure.hmm.model_variant import ModelVariant

_PATHS = ("/", "/login", "/products", "/search", "/cart", "/checkout", "/account")
_AGENTS = ("Mozilla/5.0", "curl/8.4.0", "PostmanRuntime/7.36")


@dataclass(frozen=True, slots=True)
class HybridMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    markov_blocks: int
    hmm_blocks: int


@dataclass(frozen=True, slots=True)
class HybridSweepResult:
    name: str
    markov_target_fpr: float
    hmm_target_fpr: float
    hmm_config: str
    n_states: int
    window: int
    ensemble_size: int
    max_fit_windows: int | None
    metrics: HybridMetrics


_HMM_TARGET_FPRS = (0.02, 0.05, 0.1, 0.2)
_MARKOV_TARGET_FPRS = (0.02, 0.05, 0.1, 0.2)

_HMM_CONFIGS = (
    ModelVariant("ci_baseline", n_states=4, n_iter=15, min_train_sequences=5),
    ModelVariant(
        "ci_window",
        n_states=3,
        window=4,
        max_fit_windows=800,
        n_iter=15,
        min_train_sequences=5,
    ),
    ModelVariant(
        "ci_hybrid_hmm",
        n_states=3,
        window=4,
        ensemble_size=2,
        max_fit_windows=800,
        n_iter=15,
        min_train_sequences=5,
    ),
)

# Keep the hybrid grid small enough for CI while still exposing both decision levers:
# Markov calibration gets a full target_fpr sweep; HMM coverage includes the default
# operating point plus representative config changes.
_HYBRID_HMM_OPERATING_POINTS = (
    ModelVariant("ci_baseline_fpr_0_05", n_states=4, n_iter=15, target_fpr=0.05, min_train_sequences=5),
    ModelVariant("ci_baseline_fpr_0_10", n_states=4, n_iter=15, target_fpr=0.1, min_train_sequences=5),
    ModelVariant(
        "ci_window_fpr_0_10",
        n_states=3,
        window=4,
        max_fit_windows=800,
        n_iter=15,
        target_fpr=0.1,
        min_train_sequences=5,
    ),
    ModelVariant(
        "ci_hybrid_hmm_fpr_0_10",
        n_states=3,
        window=4,
        ensemble_size=2,
        max_fit_windows=800,
        n_iter=15,
        target_fpr=0.1,
        min_train_sequences=5,
    ),
)


def normal_request(rng: random.Random, user_id: int) -> HttpRequest:
    step = rng.choice(_PATHS)
    if step == "/products":
        path = f"/products/{rng.randint(1, 80)}"
        query = f"ref=home&page={rng.randint(1, 5)}"
    elif step == "/search":
        path = step
        query = f"q=item{rng.randint(1, 20)}&page={rng.randint(1, 4)}"
    else:
        path = step
        query = ""
    body = ""
    if step in {"/cart", "/checkout"}:
        body = f"user={user_id}&quantity={rng.randint(1, 3)}&coupon=SAVE{rng.randint(10, 99)}"
    return HttpRequest(
        method="POST" if step in {"/login", "/cart", "/checkout"} else "GET",
        path=path,
        query=query,
        headers={
            "Host": "shop.example.com",
            "User-Agent": rng.choice(_AGENTS),
            "Accept": "text/html,application/json",
        },
        body=body,
        client_ip=f"10.0.0.{user_id % 250 + 1}",
    )


def normal_session(user_id: int, seed: int) -> list[Flow]:
    rng = random.Random(seed + user_id)
    requests = [
        HttpRequest("GET", "/", headers={"Host": "shop.example.com"}, client_ip=f"10.0.0.{user_id}"),
        HttpRequest(
            "GET", "/login", headers={"Host": "shop.example.com"}, client_ip=f"10.0.0.{user_id}"
        ),
        HttpRequest(
            "POST",
            "/login",
            headers={"Host": "shop.example.com"},
            body=f"user={user_id}&password=ok",
            client_ip=f"10.0.0.{user_id}",
        ),
        normal_request(rng, user_id),
        HttpRequest(
            "POST",
            "/cart",
            headers={"Host": "shop.example.com"},
            body=f"user={user_id}&sku={rng.randint(1, 100)}&quantity=1",
            client_ip=f"10.0.0.{user_id}",
        ),
        HttpRequest(
            "POST",
            "/checkout",
            headers={"Host": "shop.example.com"},
            body=f"user={user_id}&payment=card",
            client_ip=f"10.0.0.{user_id}",
        ),
    ]
    return [http_request_to_flow(request) for request in requests]


def attack_session(user_id: int, seed: int) -> list[Flow]:
    rng = random.Random(seed + user_id)
    normal_prefix = normal_session(user_id, seed)[: rng.randint(1, 3)]
    attack = rng.choice(
        [
            HttpRequest(
                "POST",
                "/admin/delete",
                headers={"Host": "shop.example.com"},
                body="id=1 OR 1=1",
                client_ip=f"10.0.1.{user_id}",
            ),
            HttpRequest(
                "GET",
                "/download",
                query="file=../../../../etc/passwd",
                headers={"Host": "shop.example.com"},
                client_ip=f"10.0.1.{user_id}",
            ),
            HttpRequest(
                "GET",
                "/search",
                query="q=<script>alert(1)</script>",
                headers={"Host": "shop.example.com"},
                client_ip=f"10.0.1.{user_id}",
            ),
        ]
    )
    suffix: list[Flow] = []
    if rng.random() < 0.35:
        suffix.append(
            Flow(
                "http",
                80,
                Direction.INBOUND,
                bytes(rng.randint(0, 255) for _ in range(rng.randint(80, 160))),
            )
        )
    return normal_prefix + [http_request_to_flow(attack), *suffix]


def flatten(sequences: list[list[Flow]]) -> list[Flow]:
    return [flow for sequence in sequences for flow in sequence]


def evaluate_hybrid(
    detector: HybridFlowDetector,
    normal_sessions: list[list[Flow]],
    attack_sessions: list[list[Flow]],
) -> HybridMetrics:
    tp = fp = fn = tn = markov_blocks = hmm_blocks = 0
    for idx, sequence in enumerate(normal_sessions):
        for flow in sequence:
            result = detector.assess_flow(flow, session_id=f"normal-{idx}")
            fp += int(result.blocked)
            tn += int(not result.blocked)
            markov_blocks += int(result.markov.blocked)
            hmm_blocks += int(result.hmm.blocked)
    for idx, sequence in enumerate(attack_sessions):
        session_blocked = False
        for flow in sequence:
            result = detector.assess_flow(flow, session_id=f"attack-{idx}")
            session_blocked = session_blocked or result.blocked
            markov_blocks += int(result.markov.blocked)
            hmm_blocks += int(result.hmm.blocked)
        tp += int(session_blocked)
        fn += int(not session_blocked)

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return HybridMetrics(
        accuracy=(tp + tn) / total if total else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        markov_blocks=markov_blocks,
        hmm_blocks=hmm_blocks,
    )


def hmm_sweep_variants() -> list[ModelVariant]:
    variants: list[ModelVariant] = []
    for config in _HMM_CONFIGS:
        for target_fpr in _HMM_TARGET_FPRS:
            suffix = str(target_fpr).replace(".", "_")
            variants.append(
                ModelVariant(
                    f"{config.name}_fpr_{suffix}",
                    extractor=config.extractor,
                    n_states=config.n_states,
                    window=config.window,
                    ensemble_size=config.ensemble_size,
                    target_fpr=target_fpr,
                    n_iter=config.n_iter,
                    max_fit_windows=config.max_fit_windows,
                    min_train_sequences=config.min_train_sequences,
                )
            )
    return variants


def run_hybrid_sweep(
    train: list[list[Flow]],
    validation: list[list[Flow]],
    eval_normal: list[list[Flow]],
    eval_attack: list[list[Flow]],
) -> list[HybridSweepResult]:
    results: list[HybridSweepResult] = []
    for markov_target_fpr in _MARKOV_TARGET_FPRS:
        for hmm_config in _HYBRID_HMM_OPERATING_POINTS:
            detector = HybridFlowDetector(
                markov_model=MarkovFlowModel(target_fpr=markov_target_fpr),
                hmm_detector=HmmDetector(
                    n_states=hmm_config.n_states,
                    window=hmm_config.window,
                    ensemble_size=hmm_config.ensemble_size,
                    max_fit_windows=hmm_config.max_fit_windows,
                    n_iter=hmm_config.n_iter,
                    min_train_sequences=hmm_config.min_train_sequences,
                    target_fpr=hmm_config.target_fpr,
                ),
            )
            detector.train_sequences(train, validation)
            suffix = str(markov_target_fpr).replace(".", "_")
            results.append(
                HybridSweepResult(
                    name=f"markov_fpr_{suffix}__{hmm_config.name}",
                    markov_target_fpr=markov_target_fpr,
                    hmm_target_fpr=hmm_config.target_fpr,
                    hmm_config=hmm_config.name,
                    n_states=hmm_config.n_states,
                    window=hmm_config.window,
                    ensemble_size=hmm_config.ensemble_size,
                    max_fit_windows=hmm_config.max_fit_windows,
                    metrics=evaluate_hybrid(detector, eval_normal, eval_attack),
                )
            )
    return results


def best_hmm_by_accuracy(results: list[VariantResult]) -> VariantResult:
    return max(
        (r for r in results if r.classification is not None),
        key=lambda r: (
            r.classification.accuracy if r.classification is not None else -1.0,
            r.classification.f1 if r.classification is not None else -1.0,
            r.classification.recall if r.classification is not None else -1.0,
            -r.classification.fp if r.classification is not None else 0,
            r.variant.name,
        ),
    )


def best_hmm_by_f1(results: list[VariantResult]) -> VariantResult:
    return max(
        (r for r in results if r.classification is not None),
        key=lambda r: (
            r.classification.f1 if r.classification is not None else -1.0,
            r.classification.accuracy if r.classification is not None else -1.0,
            r.classification.recall if r.classification is not None else -1.0,
            -r.classification.fp if r.classification is not None else 0,
            r.variant.name,
        ),
    )


def best_hybrid_by_accuracy(results: list[HybridSweepResult]) -> HybridSweepResult:
    return max(
        results,
        key=lambda r: (
            r.metrics.accuracy,
            r.metrics.f1,
            r.metrics.recall,
            -r.metrics.fp,
            r.name,
        ),
    )


def best_hybrid_by_f1(results: list[HybridSweepResult]) -> HybridSweepResult:
    return max(
        results,
        key=lambda r: (
            r.metrics.f1,
            r.metrics.accuracy,
            r.metrics.recall,
            -r.metrics.fp,
            r.name,
        ),
    )


def result_to_dict(result: VariantResult) -> dict[str, object]:
    report = _report_dict(result.report)
    classification = _classification_dict(result.classification)
    return {
        "variant": result.variant.name,
        "extractor": result.variant.extractor,
        "n_states": result.variant.n_states,
        "window": result.variant.window,
        "ensemble_size": result.variant.ensemble_size,
        "target_fpr": result.variant.target_fpr,
        "n_iter": result.variant.n_iter,
        "max_fit_windows": result.variant.max_fit_windows,
        "min_train_sequences": result.variant.min_train_sequences,
        "train_count": result.train_count,
        "validation_count": result.validation_count,
        "covered": result.covered,
        "uncovered": result.uncovered,
        "eval_normal": result.eval_normal,
        "eval_attack": result.eval_attack,
        "report": report,
        "classification": classification,
    }


def hybrid_sweep_to_dict(result: HybridSweepResult) -> dict[str, object]:
    return {
        "name": result.name,
        "markov_target_fpr": result.markov_target_fpr,
        "hmm_target_fpr": result.hmm_target_fpr,
        "hmm_config": result.hmm_config,
        "n_states": result.n_states,
        "window": result.window,
        "ensemble_size": result.ensemble_size,
        "max_fit_windows": result.max_fit_windows,
        "metrics": asdict(result.metrics),
    }


def _report_dict(report: EvaluationReport | None) -> dict[str, float | int] | None:
    return asdict(report) if report is not None else None


def _classification_dict(
    classification: ClassificationMetrics | None,
) -> dict[str, float | int] | None:
    return asdict(classification) if classification is not None else None


def render_markdown(
    results: list[VariantResult],
    hybrid: HybridMetrics,
    hybrid_sweep: list[HybridSweepResult],
    elapsed: float,
) -> str:
    best_aucp = max(
        (r for r in results if r.report is not None),
        key=lambda r: r.report.partial_auc if r.report is not None else -1.0,
    )
    hmm_accuracy = best_hmm_by_accuracy(results)
    hmm_f1 = best_hmm_by_f1(results)
    hybrid_accuracy = best_hybrid_by_accuracy(hybrid_sweep)
    hybrid_f1 = best_hybrid_by_f1(hybrid_sweep)
    lines = [
        "# HMM / Hybrid Flow Benchmark",
        "",
        f"- elapsed_seconds: {elapsed:.2f}",
        f"- best_hmm_partial_auc_variant: {best_aucp.variant.name}",
        f"- best_hmm_partial_auc: {best_aucp.report.partial_auc:.3f}" if best_aucp.report else "",
        f"- best_hmm_accuracy_variant: {hmm_accuracy.variant.name}",
        (
            "- best_hmm_accuracy: "
            f"{hmm_accuracy.classification.accuracy:.3f}"
            if hmm_accuracy.classification
            else ""
        ),
        f"- best_hmm_f1_variant: {hmm_f1.variant.name}",
        f"- best_hmm_f1: {hmm_f1.classification.f1:.3f}" if hmm_f1.classification else "",
        f"- best_hybrid_accuracy_config: {hybrid_accuracy.name}",
        f"- best_hybrid_accuracy: {hybrid_accuracy.metrics.accuracy:.3f}",
        f"- best_hybrid_f1_config: {hybrid_f1.name}",
        f"- best_hybrid_f1: {hybrid_f1.metrics.f1:.3f}",
        "",
        "## HMM Target FPR / Config Sweep",
        "",
        "```text",
        render_comparison(results),
        "```",
        "",
        "## Default Hybrid Markov + HMM",
        "",
        "| accuracy | precision | recall | f1 | tp | fp | fn | tn | markov_blocks | hmm_blocks |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {hybrid.accuracy:.3f} | {hybrid.precision:.3f} | {hybrid.recall:.3f} | "
            f"{hybrid.f1:.3f} | {hybrid.tp} | {hybrid.fp} | {hybrid.fn} | {hybrid.tn} | "
            f"{hybrid.markov_blocks} | {hybrid.hmm_blocks} |"
        ),
        "",
        "## Hybrid Markov Target FPR / HMM Config Sweep",
        "",
        (
            "| config | markov_fpr | hmm_fpr | accuracy | precision | recall | f1 | "
            "tp | fp | fn | tn | markov_blocks | hmm_blocks |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            (
                f"| {result.name} | {result.markov_target_fpr:.2f} | "
                f"{result.hmm_target_fpr:.2f} | {result.metrics.accuracy:.3f} | "
                f"{result.metrics.precision:.3f} | {result.metrics.recall:.3f} | "
                f"{result.metrics.f1:.3f} | {result.metrics.tp} | {result.metrics.fp} | "
                f"{result.metrics.fn} | {result.metrics.tn} | {result.metrics.markov_blocks} | "
                f"{result.metrics.hmm_blocks} |"
            )
            for result in sorted(
                hybrid_sweep,
                key=lambda r: (r.metrics.accuracy, r.metrics.f1, -r.metrics.fp, r.name),
                reverse=True,
            )
        ],
        "",
    ]
    return "\n".join(line for line in lines if line != "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/ci")
    parser.add_argument("--train-sessions", type=int, default=18)
    parser.add_argument("--validation-sessions", type=int, default=6)
    parser.add_argument("--eval-sessions", type=int, default=12)
    args = parser.parse_args()

    started = time.perf_counter()
    train = [normal_session(i, 1000) for i in range(args.train_sessions)]
    validation = [
        normal_session(i + 100, 2000) for i in range(args.validation_sessions)
    ]
    eval_normal = [normal_session(i + 200, 3000) for i in range(args.eval_sessions)]
    eval_attack = [attack_session(i + 300, 4000) for i in range(args.eval_sessions)]

    variants = hmm_sweep_variants()
    benchmark = ModelBenchmark(
        flatten(train),
        flatten(validation),
        [(flow, False) for flow in flatten(eval_normal)]
        + [(flow, True) for flow in flatten(eval_attack)],
    )
    hmm_results = benchmark.run(variants)

    hybrid_detector = HybridFlowDetector(
        markov_model=MarkovFlowModel(target_fpr=0.1),
        hmm_detector=HmmDetector(n_states=3, n_iter=15, min_train_sequences=5, target_fpr=0.1),
    )
    hybrid_detector.train_sequences(train, validation)
    hybrid_metrics = evaluate_hybrid(hybrid_detector, eval_normal, eval_attack)
    hybrid_sweep = run_hybrid_sweep(train, validation, eval_normal, eval_attack)
    elapsed = time.perf_counter() - started

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hmm_accuracy = best_hmm_by_accuracy(hmm_results)
    hmm_f1 = best_hmm_by_f1(hmm_results)
    hybrid_accuracy = best_hybrid_by_accuracy(hybrid_sweep)
    hybrid_f1 = best_hybrid_by_f1(hybrid_sweep)
    metrics = {
        "elapsed_seconds": round(elapsed, 3),
        "dataset": {
            "train_sessions": args.train_sessions,
            "validation_sessions": args.validation_sessions,
            "eval_normal_sessions": args.eval_sessions,
            "eval_attack_sessions": args.eval_sessions,
            "train_flows": len(flatten(train)),
            "validation_flows": len(flatten(validation)),
            "eval_normal_flows": len(flatten(eval_normal)),
            "eval_attack_flows": len(flatten(eval_attack)),
        },
        "hmm": [result_to_dict(result) for result in hmm_results],
        "hmm_best": {
            "accuracy": result_to_dict(hmm_accuracy),
            "f1": result_to_dict(hmm_f1),
        },
        "hybrid": asdict(hybrid_metrics),
        "hybrid_sweep": [hybrid_sweep_to_dict(result) for result in hybrid_sweep],
        "hybrid_best": {
            "accuracy": hybrid_sweep_to_dict(hybrid_accuracy),
            "f1": hybrid_sweep_to_dict(hybrid_f1),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = render_markdown(hmm_results, hybrid_metrics, hybrid_sweep, elapsed)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
