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
    calibration = metrics["calibration"]
    operational = metrics["operational_assessment"]
    counterfactual = validation["counterfactual"]
    artifact_path = tmp_path / "reports" / "access_log_workflow_model.json"

    assert isinstance(validation, dict)
    assert isinstance(runtime_boot, dict)
    assert isinstance(calibration, dict)
    assert isinstance(operational, dict)
    assert isinstance(counterfactual, dict)
    assert artifact_path.is_file()
    assert validation["inspected_requests"] > 0
    assert validation["covered_requests"] == validation["inspected_requests"]
    assert validation["coverage_rate"] == 1.0
    assert validation["avg_transition_score"] < 0.0
    assert validation["score_separation"] > 0.0
    assert validation["alert_rate_lift"] > 0.0
    assert counterfactual["inspected_requests"] == validation["inspected_requests"]
    assert counterfactual["alert_rate"] > validation["alert_rate"]
    assert counterfactual["avg_transition_score"] < validation["avg_transition_score"]
    assert calibration["selected_target_fpr"] == metrics["model"]["target_fpr"]
    assert calibration["shadow_target_fpr"] in calibration["candidate_target_fprs"]
    assert len(calibration["profiles"]) == 3
    assert operational["serving_target_fpr"] == calibration["selected_target_fpr"]
    assert operational["shadow_target_fpr"] == calibration["shadow_target_fpr"]
    assert operational["serving_score_separation"] > 0.0
    assert operational["needs_labeled_attack_dataset"] is False
    assert runtime_boot["ready"] is True
    assert runtime_boot["serving_detectors"] == ["workflow"]
    assert runtime_boot["model_fingerprint_matched"] is True


def test_ci_access_log_training_reports_generated_flow_map_and_rare_flows(
    tmp_path,
) -> None:
    access_log = tmp_path / "access.log"
    rows: list[str] = []
    for session in range(1000):
        ip = f"203.0.{session // 250}.{session % 250}"
        agent = f"common-agent-{session}"
        rows.extend(
            [
                _line(ip, agent, 0, "/"),
                _line(ip, agent, 1, "/products/42"),
                _line(ip, agent, 2, "/checkout"),
            ]
        )
    rows.extend(
        [
            _line("198.51.100.10", "rare-agent", 0, "/"),
            _line("198.51.100.10", "rare-agent", 1, "/admin/export"),
            _line("198.51.100.10", "rare-agent", 2, "/checkout"),
        ]
    )
    access_log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    metrics = run(access_log, tmp_path / "reports", max_records=None)
    flow_distribution = metrics["flow_distribution"]
    assert isinstance(flow_distribution, dict)
    flow_map = flow_distribution["flow_type_map"]
    assert isinstance(flow_map, list)

    rare_flows: list[dict[str, object]] = []
    for flow in flow_map:
        assert isinstance(flow, dict)
        if flow["is_rare"]:
            rare_flows.append(flow)
    assert flow_distribution["total_flows"] == 1001
    assert flow_distribution["unique_flow_types"] == 2
    assert flow_distribution["rare_probability_threshold"] == 0.001
    assert flow_distribution["rare_flow_types"] == 1
    assert flow_distribution["rare_flow_occurrences"] == 1
    assert rare_flows[0]["count"] == 1
    assert rare_flows[0]["probability"] == 0.000999
