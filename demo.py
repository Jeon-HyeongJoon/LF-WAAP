"""End-to-end POC demo.

Part A — WAF: rule-set + HMM detectors combined via BlockDecisionService.
Part B — HMM as an IDS: train one-class byte-level HMM on normal traffic,
         calibrate threshold to a target FPR, report ROC-AUC / PR-AUC / TPR@FPR.

Run:  python demo.py   (after `pip install -e .`)
"""

from __future__ import annotations

import random

from waf.application import InspectRequest
from waf.domain.model.flow import Direction, Flow
from waf.domain.model.http_request import HttpRequest
from waf.domain.service import BlockDecisionService, BlockPolicy
from waf.infrastructure.hmm import HmmDetector, evaluate
from waf.infrastructure.hmm.features import ByteClassExtractor
from waf.infrastructure.hmm.http_flow import http_request_to_flow
from waf.infrastructure.ruleset import RuleSetDetector

PATHS = ["/index.html", "/products", "/api/items", "/search", "/account"]


def normal_request(rng: random.Random) -> HttpRequest:
    return HttpRequest(
        "GET",
        rng.choice(PATHS),
        f"id={rng.randint(1, 9999)}&sort=price&page={rng.randint(1, 20)}",
        headers={"Host": "shop.example.com", "User-Agent": "Mozilla/5.0", "Accept": "text/html"},
    )


def normal_flow(rng: random.Random) -> Flow:
    return http_request_to_flow(normal_request(rng))


def attack_flow(rng: random.Random) -> Flow:
    n = rng.randint(90, 200)
    return Flow("http", 80, Direction.INBOUND, bytes(rng.randint(0, 255) for _ in range(n)))


def build_waf() -> tuple[InspectRequest, HmmDetector]:
    rng = random.Random(1)
    hmm = HmmDetector(extractor=ByteClassExtractor(), n_states=6, target_fpr=0.01, n_iter=80)
    hmm.train(
        normal_flows=[normal_flow(rng) for _ in range(150)],
        validation_flows=[normal_flow(rng) for _ in range(80)],
    )
    service = BlockDecisionService([RuleSetDetector.default(), hmm], policy=BlockPolicy.ANY)
    return InspectRequest(service), hmm


def main() -> None:
    inspect, hmm = build_waf()

    print("== Part A: WAF (ruleset + HMM) ==")
    samples = [
        HttpRequest("GET", "/api/products", "id=42&sort=price"),
        HttpRequest("GET", "/api/products", "id=1 UNION SELECT password FROM users"),
        HttpRequest("GET", "/search", "q=<script>alert(1)</script>"),
    ]
    for req in samples:
        verdict = inspect(req)
        mark = "BLOCK" if verdict.is_blocked else "ALLOW"
        print(f"[{mark}] {req.path}?{req.query}")
        if verdict.is_blocked:
            print(f"        {verdict.reason}")

    print("\n== Part B: HMM IDS metrics (held-out normal vs binary attacks) ==")
    rng = random.Random(99)
    normal_scores = [hmm.assess_flow(normal_flow(rng)).score for _ in range(200)]
    attack_scores = [hmm.assess_flow(attack_flow(rng)).score for _ in range(200)]
    report = evaluate(normal_scores, attack_scores, target_fpr=0.01)
    print(f"   {report}")


if __name__ == "__main__":
    main()
