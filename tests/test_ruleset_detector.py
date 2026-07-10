from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.ruleset import RuleSetDetector
import pytest


def make(path: str = "/", query: str = "", body: str = "") -> HttpRequest:
    return HttpRequest(method="GET", path=path, query=query, body=body)


def test_allows_benign_request() -> None:
    detector = RuleSetDetector.default()
    signal = detector.inspect(make(path="/products", query="id=42"))
    assert signal.blocked is False


def test_blocks_sql_injection_union_select() -> None:
    detector = RuleSetDetector.default()
    signal = detector.inspect(make(path="/items", query="id=1 UNION SELECT password FROM users"))
    assert signal.blocked is True
    assert "SQLI" in signal.reason


def test_blocks_xss_script_tag() -> None:
    detector = RuleSetDetector.default()
    signal = detector.inspect(make(path="/search", query="q=<script>alert(1)</script>"))
    assert signal.blocked is True
    assert "XSS" in signal.reason


def test_blocks_path_traversal() -> None:
    detector = RuleSetDetector.default()
    signal = detector.inspect(make(path="/files", query="name=../../etc/passwd"))
    assert signal.blocked is True
    assert "LFI" in signal.reason


def test_ruleset_detector_exposes_stable_model_fingerprint() -> None:
    first = RuleSetDetector.default()
    second = RuleSetDetector.default()

    assert first.model_fingerprint
    assert first.model_fingerprint == second.model_fingerprint
    assert first.rule_count >= 1


def test_ruleset_loader_rejects_missing_rules_list(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("not_rules: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rules"):
        RuleSetDetector.from_yaml(path)


def test_ruleset_loader_rejects_invalid_regex_with_rule_id(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
rules:
  - id: BAD-REGEX
    description: invalid regex
    pattern: "["
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="BAD-REGEX"):
        RuleSetDetector.from_yaml(path)


def test_ruleset_loader_rejects_blank_pattern(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
rules:
  - id: EMPTY-PATTERN
    description: empty pattern blocks every request
    pattern: ""
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pattern"):
        RuleSetDetector.from_yaml(path)


def test_ruleset_loader_rejects_null_pattern(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
rules:
  - id: NULL-PATTERN
    description: null pattern must not become a literal string
    pattern:
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pattern"):
        RuleSetDetector.from_yaml(path)


def test_ruleset_loader_rejects_duplicate_rule_ids(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
rules:
  - id: DUPLICATE
    description: first rule
    pattern: "/one"
  - id: DUPLICATE
    description: second rule
    pattern: "/two"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="DUPLICATE"):
        RuleSetDetector.from_yaml(path)


def test_ruleset_loader_rejects_rule_id_with_surrounding_whitespace(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
rules:
  - id: " CUSTOM-ADMIN "
    description: custom admin path block
    pattern: "/internal-admin"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="id"):
        RuleSetDetector.from_yaml(path)


def test_ruleset_loader_rejects_declared_fingerprint_mismatch(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
fingerprint: deadbeef
rules:
  - id: CUSTOM-ADMIN
    description: custom admin path block
    pattern: "/internal-admin"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        RuleSetDetector.from_yaml(path)


def test_ruleset_loader_accepts_declared_matching_fingerprint(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    rules_yaml = """
rules:
  - id: CUSTOM-ADMIN
    description: custom admin path block
    pattern: "/internal-admin"
""".lstrip()
    path.write_text(rules_yaml, encoding="utf-8")
    expected_fingerprint = RuleSetDetector.from_yaml(path).model_fingerprint

    path.write_text(
        f"fingerprint: {expected_fingerprint}\n{rules_yaml}",
        encoding="utf-8",
    )

    detector = RuleSetDetector.from_yaml(path)

    assert detector.model_fingerprint == expected_fingerprint
