"""Deterministic synthetic traffic generators for HMM tests.

Normal = ASCII HTTP text; attacks = high-entropy/binary payloads. The byte-class
distributions differ enough that a one-class HMM trained on normal separates them.
"""

from __future__ import annotations

import random

from waf.domain.model.flow import Direction, Flow
from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.flow.http_flow import http_request_to_flow

_PATHS = ["/index.html", "/products", "/api/items", "/search", "/account"]
_AGENTS = ["Mozilla/5.0", "curl/8.4.0", "PostmanRuntime/7.36"]


def normal_request(rng: random.Random) -> HttpRequest:
    path = rng.choice(_PATHS)
    return HttpRequest(
        method="GET",
        path=path,
        query=f"id={rng.randint(1, 9999)}&sort=price&page={rng.randint(1, 20)}",
        headers={
            "Host": "shop.example.com",
            "User-Agent": rng.choice(_AGENTS),
            "Accept": "text/html,application/json",
        },
    )


def normal_flow(rng: random.Random) -> Flow:
    # Go through the same adapter inspect() uses, so train/score share one
    # byte representation (no train/serve skew).
    return http_request_to_flow(normal_request(rng))


def attack_flow(rng: random.Random) -> Flow:
    """High-entropy binary payload (shellcode / encrypted-blob-like)."""
    n = rng.randint(90, 200)
    payload = bytes(rng.randint(0, 255) for _ in range(n))
    return Flow("http", 80, Direction.INBOUND, payload)


def batch(fn, n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [fn(rng) for _ in range(n)]
