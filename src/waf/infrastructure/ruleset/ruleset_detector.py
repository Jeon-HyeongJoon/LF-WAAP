"""RuleSetDetector — signature/regex based detection (OWASP-style)."""

from __future__ import annotations

import hashlib
import json
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
    pattern_text: str
    pattern: re.Pattern[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Rule":
        for field in ("id", "description", "pattern"):
            if field not in raw:
                raise ValueError(f"ruleset rule is missing required field: {field}")
        rule_id = _required_rule_text(raw, "id")
        description = _required_rule_text(raw, "description")
        pattern = _required_rule_text(raw, "pattern")
        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            raise ValueError(f"ruleset rule {rule_id} has invalid regex: {exc}") from exc
        return cls(
            id=rule_id,
            description=description,
            pattern_text=pattern,
            pattern=compiled,
        )

    def to_fingerprint_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "description": self.description,
            "pattern": self.pattern_text,
        }


class RuleSetDetector:
    name = "ruleset"

    def __init__(self, rules: list[Rule]) -> None:
        self._rules = rules
        self._model_fingerprint = _rules_fingerprint(rules)

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def model_fingerprint(self) -> str:
        return self._model_fingerprint

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuleSetDetector":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        rules = _rules_from_document(data)
        parsed_rules = [Rule.from_dict(rule) for rule in rules]
        _validate_unique_rule_ids(parsed_rules)
        detector = cls(parsed_rules)
        _validate_declared_fingerprint(data, detector.model_fingerprint)
        return detector

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


def _rules_fingerprint(rules: list[Rule]) -> str:
    document = [rule.to_fingerprint_dict() for rule in rules]
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _required_rule_text(raw: dict[str, Any], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str):
        raise ValueError(f"ruleset rule field must be a string: {field}")
    if not value.strip():
        raise ValueError(f"ruleset rule has blank required field: {field}")
    return value


def _validate_unique_rule_ids(rules: list[Rule]) -> None:
    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            raise ValueError(f"ruleset artifact contains duplicate rule id: {rule.id}")
        seen.add(rule.id)


def _rules_from_document(data: object) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("ruleset artifact must be a YAML object")
    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("ruleset artifact must contain a non-empty rules list")
    parsed: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"ruleset rule at index {index} must be an object")
        parsed.append(rule)
    return parsed


def _validate_declared_fingerprint(data: object, actual_fingerprint: str) -> None:
    if not isinstance(data, dict) or "fingerprint" not in data:
        return
    declared = str(data["fingerprint"])
    if declared != actual_fingerprint:
        raise ValueError("ruleset artifact fingerprint mismatch")
