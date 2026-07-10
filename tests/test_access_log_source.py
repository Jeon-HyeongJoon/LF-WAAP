from access_log_source import AccessLogTrainingDataset, parse_access_log_line


def test_parse_combined_access_log_line_to_http_request() -> None:
    line = (
        '203.0.113.10 - - [10/Jul/2026:10:00:00 +0900] '
        '"GET /products/42?ref=home HTTP/1.1" 200 123 "-" "Mozilla/5.0"'
    )

    record = parse_access_log_line(line)

    assert record is not None
    assert record.request.method == "GET"
    assert record.request.path == "/products/42"
    assert record.request.query == "ref=home"
    assert record.request.client_ip == "203.0.113.10"
    assert record.request.headers["user-agent"] == "Mozilla/5.0"
    assert record.status == 200
    assert record.session_key


def test_access_log_training_dataset_groups_successful_sessions(tmp_path) -> None:
    access_log = tmp_path / "access.log"
    access_log.write_text(
        "\n".join(
            [
                '203.0.113.10 - - [10/Jul/2026:10:00:00 +0900] '
                '"GET / HTTP/1.1" 200 10 "-" "Mozilla/5.0"',
                '203.0.113.10 - - [10/Jul/2026:10:00:01 +0900] '
                '"GET /products/42 HTTP/1.1" 200 10 "-" "Mozilla/5.0"',
                '203.0.113.10 - - [10/Jul/2026:10:00:02 +0900] '
                '"GET /admin HTTP/1.1" 404 10 "-" "Mozilla/5.0"',
                '198.51.100.7 - - [10/Jul/2026:10:01:00 +0900] '
                '"GET / HTTP/1.1" 200 10 "-" "curl/8.0"',
                '198.51.100.7 - - [10/Jul/2026:10:01:01 +0900] '
                '"GET /search?q=item HTTP/1.1" 302 10 "-" "curl/8.0"',
                "not an access log line",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = AccessLogTrainingDataset.from_path(access_log)

    assert len(dataset.sequences) == 2
    assert [len(sequence) for sequence in dataset.sequences] == [2, 2]
    assert dataset.summary.parsed_records == 5
    assert dataset.summary.skipped_records == 1
    assert dataset.summary.training_records == 4
    assert dataset.summary.session_count == 2
    assert dataset.summary.route_count == 3
    assert dataset.summary.status_counts == {"2xx": 3, "3xx": 1, "4xx": 1}
