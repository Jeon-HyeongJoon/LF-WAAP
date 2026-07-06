"""RuleSetDetector — signature/regex based detection (OWASP-style)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from waf.domain.model.detection import DetectionSignal
from waf.domain.model.http_request import HttpRequest


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    description: str
    pattern: re.Pattern[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Rule":
        return cls(
            id=str(raw["id"]),
            description=str(raw["description"]),
            pattern=re.compile(str(raw["pattern"]), re.IGNORECASE | re.MULTILINE),
        )


class RuleSetDetector:
    name = "ruleset"

    def __init__(self, rules: list[Rule]) -> None:
        self._rules = rules

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuleSetDetector":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls([Rule.from_dict(r) for r in data["rules"]])

    @classmethod
    def default(cls) -> "RuleSetDetector":
        return cls.from_yaml(Path(__file__).with_name("rules.yaml"))

    def inspect(self, request: HttpRequest) -> DetectionSignal:
        payload = request.inspectable_payload
        for rule in self._rules:
            if rule.pattern.search(payload):
                return DetectionSignal(
                    detector=self.name,
                    blocked=True,
                    reason=f"{rule.id}: {rule.description}",
                    score=1.0,
                )
        return DetectionSignal(self.name, blocked=False, reason="no rule matched", score=0.0)
