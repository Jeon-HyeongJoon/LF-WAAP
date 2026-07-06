from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.ruleset import RuleSetDetector


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
