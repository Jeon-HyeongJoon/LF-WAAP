from scripts.ci_access_log_training import run


def _line(ip: str, user_agent: str, second: int, target: str) -> str:
    return (
        f'{ip} - - [10/Jul/2026:10:00:{second:02d} +0900] '
        f'"GET {target} HTTP/1.1" 200 10 "-" "{user_agent}"'
    )


def test_ci_access_log_training_reports_holdout_transition_quality(tmp_path) -> None:
    access_log = tmp_path / "access.log"
    rows: list[str] = []
    for session in range(6):
        ip = f"203.0.113.{session}"
        agent = f"agent-{session}"
        rows.extend(
            [
                _line(ip, agent, 0, "/"),
                _line(ip, agent, 1, "/products/42"),
                _line(ip, agent, 2, "/checkout"),
            ]
        )
    access_log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    metrics = run(access_log, tmp_path / "reports", max_records=None)
    validation = metrics["validation"]
    runtime_boot = metrics["runtime_boot"]
    artifact_path = tmp_path / "reports" / "access_log_workflow_model.json"

    assert isinstance(validation, dict)
    assert isinstance(runtime_boot, dict)
    assert artifact_path.is_file()
    assert validation["inspected_requests"] > 0
    assert validation["covered_requests"] == validation["inspected_requests"]
    assert validation["coverage_rate"] == 1.0
    assert validation["avg_transition_score"] < 0.0
    assert runtime_boot["ready"] is True
    assert runtime_boot["serving_detectors"] == ["workflow"]
    assert runtime_boot["model_fingerprint_matched"] is True
