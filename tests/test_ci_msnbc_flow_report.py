from scripts.ci_msnbc_flow_report import run


def test_ci_msnbc_flow_report_emits_threshold_and_flow_distribution(tmp_path) -> None:
    dataset = tmp_path / "msnbc.seq"
    rows = ["% Sequences:"]
    rows.extend(["1 2 3"] * 1000)
    rows.append("1 4 3")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")

    metrics = run(dataset, tmp_path / "reports", max_users=None, flow_map_limit=10)
    model = metrics["model"]
    flow_distribution = metrics["flow_distribution"]
    assert isinstance(model, dict)
    assert isinstance(flow_distribution, dict)
    flow_map = flow_distribution["flow_type_map"]
    assert isinstance(flow_map, list)

    rare_flows: list[dict[str, object]] = []
    for flow in flow_map:
        assert isinstance(flow, dict)
        if flow["is_rare"]:
            rare_flows.append(flow)

    assert model["threshold"] is not None
    assert flow_distribution["total_flows"] == 1001
    assert flow_distribution["unique_flow_types"] == 2
    assert flow_distribution["rare_flow_probability_threshold"] == 0.001
    assert flow_distribution["rare_flow_types"] == 1
    assert rare_flows[0]["count"] == 1
    assert rare_flows[0]["probability"] == 0.000999
