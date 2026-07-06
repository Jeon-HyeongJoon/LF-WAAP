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


def result_to_dict(result: VariantResult) -> dict[str, object]:
    report = _report_dict(result.report)
    classification = _classification_dict(result.classification)
    return {
        "variant": result.variant.name,
        "extractor": result.variant.extractor,
        "n_states": result.variant.n_states,
        "window": result.variant.window,
        "ensemble_size": result.variant.ensemble_size,
        "train_count": result.train_count,
        "validation_count": result.validation_count,
        "covered": result.covered,
        "uncovered": result.uncovered,
        "eval_normal": result.eval_normal,
        "eval_attack": result.eval_attack,
        "report": report,
        "classification": classification,
    }


def _report_dict(report: EvaluationReport | None) -> dict[str, float | int] | None:
    return asdict(report) if report is not None else None


def _classification_dict(
    classification: ClassificationMetrics | None,
) -> dict[str, float | int] | None:
    return asdict(classification) if classification is not None else None


def render_markdown(results: list[VariantResult], hybrid: HybridMetrics, elapsed: float) -> str:
    best = max(
        (r for r in results if r.report is not None),
        key=lambda r: r.report.partial_auc if r.report is not None else -1.0,
    )
    lines = [
        "# HMM / Hybrid Flow Benchmark",
        "",
        f"- elapsed_seconds: {elapsed:.2f}",
        f"- best_hmm_variant: {best.variant.name}",
        f"- best_hmm_partial_auc: {best.report.partial_auc:.3f}" if best.report else "",
        "",
        "## HMM Variants",
        "",
        "```text",
        render_comparison(results),
        "```",
        "",
        "## Hybrid Markov + HMM",
        "",
        "| accuracy | precision | recall | f1 | tp | fp | fn | tn | markov_blocks | hmm_blocks |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {hybrid.accuracy:.3f} | {hybrid.precision:.3f} | {hybrid.recall:.3f} | "
            f"{hybrid.f1:.3f} | {hybrid.tp} | {hybrid.fp} | {hybrid.fn} | {hybrid.tn} | "
            f"{hybrid.markov_blocks} | {hybrid.hmm_blocks} |"
        ),
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

    variants = [
        ModelVariant("ci_baseline", n_states=4, n_iter=15, target_fpr=0.1, min_train_sequences=5),
        ModelVariant(
            "ci_window",
            n_states=3,
            window=4,
            max_fit_windows=800,
            n_iter=15,
            target_fpr=0.1,
            min_train_sequences=5,
        ),
        ModelVariant(
            "ci_hybrid_hmm",
            n_states=3,
            window=4,
            ensemble_size=2,
            max_fit_windows=800,
            n_iter=15,
            target_fpr=0.1,
            min_train_sequences=5,
        ),
    ]
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
    elapsed = time.perf_counter() - started

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        "hybrid": asdict(hybrid_metrics),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = render_markdown(hmm_results, hybrid_metrics, elapsed)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
