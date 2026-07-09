"""CsicTrafficSource — CSIC CSV -> Flow adapter implementing the TrafficSource port.

All tests use small in-memory CSVs written to tmp_path; the large real dataset is
never read.
"""

import pytest

import csic_traffic_source as cts

from waf.domain.model.flow import Direction, Flow
from waf.domain.port.traffic_source import TrafficSource

HEADER = "classification,Method,host,content,URL"
NORMAL = "0,GET,localhost:8080,,http://localhost:8080/a HTTP/1.1"
ANOMALOUS = "1,GET,localhost:8080,,http://localhost:8080/evil HTTP/1.1"


def _csv(tmp_path, rows: list[str]):
    path = tmp_path / "d.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="latin-1")
    return path


def _normal(i: int) -> str:
    return f"0,GET,localhost:8080,,http://localhost:8080/p{i} HTTP/1.1"


def test_normal_yields_flows_from_benign_records(tmp_path) -> None:
    src = cts.CsicTrafficSource(_csv(tmp_path, [NORMAL] * 5), validation_ratio=0.0)
    flows = list(src.normal())

    assert len(flows) == 5
    f = flows[0]
    assert isinstance(f, Flow)
    assert f.protocol == "http"
    assert f.port == 8080
    assert f.direction is Direction.INBOUND
    assert b"GET /a HTTP/1.1" in f.payload


def test_record_to_request_parses_reconstructed_http_request(tmp_path) -> None:
    row = (
        "0,POST,localhost:8080,"
        "id=3&nombre=Vino&cantidad=1,"
        "http://localhost:8080/tienda1/publico/anadir.jsp?ref=home HTTP/1.1"
    )
    record = next(cts.iter_packets(_csv(tmp_path, [row])))

    request = cts.record_to_request(record)

    assert request.method == "POST"
    assert request.path == "/tienda1/publico/anadir.jsp"
    assert request.query == "ref=home"
    assert request.headers["host"] == "localhost:8080"
    assert request.body == "id=3&nombre=Vino&cantidad=1"


def test_split_is_deterministic_disjoint_and_complete(tmp_path) -> None:
    path = _csv(tmp_path, [_normal(i) for i in range(10)])
    src = cts.CsicTrafficSource(path, validation_ratio=0.2)

    normal = {f.payload for f in src.normal()}
    validation = {f.payload for f in src.validation()}

    assert len(validation) == 2  # 20% of 10
    assert len(normal) == 8
    assert normal.isdisjoint(validation)  # no leakage between fit and calibration
    assert len(normal | validation) == 10  # every benign record covered

    # deterministic across instances/iterations
    again = {f.payload for f in cts.CsicTrafficSource(path, validation_ratio=0.2).validation()}
    assert again == validation


def test_anomalous_records_excluded_from_both_streams(tmp_path) -> None:
    rows = [_normal(0), ANOMALOUS, _normal(1), ANOMALOUS, _normal(2)]
    src = cts.CsicTrafficSource(_csv(tmp_path, rows), validation_ratio=0.34)

    all_flows = list(src.normal()) + list(src.validation())
    assert len(all_flows) == 3  # only the 3 benign records
    assert all(b"/evil" not in f.payload for f in all_flows)


def test_satisfies_traffic_source_port(tmp_path) -> None:
    src = cts.CsicTrafficSource(_csv(tmp_path, [NORMAL]), validation_ratio=0.0)
    assert isinstance(src, TrafficSource)


def test_end_to_end_calibration_with_real_hmm(tmp_path) -> None:
    pytest.importorskip("hmmlearn")
    import random

    from waf.application.calibrate_model import CalibrateModel
    from waf.infrastructure.hmm import HmmDetector

    path = _csv(tmp_path, [_normal(i) for i in range(50)])
    src = cts.CsicTrafficSource(path, validation_ratio=0.2)

    class MemRepo:
        def __init__(self) -> None:
            self.saved = None

        def save(self, detector) -> None:
            self.saved = detector

        def load(self):
            return self.saved

    repo = MemRepo()
    detector = HmmDetector(n_states=4, target_fpr=0.1, min_train_sequences=5, n_iter=40)

    report = CalibrateModel(src, repo)(detector)

    assert report.normal_count == 40
    assert report.validation_count == 10
    assert detector.is_trained
    assert repo.saved is detector

    # the calibrated detector flags a clearly anomalous (binary) flow in the
    # same partition as the trained benign GET traffic
    rng = random.Random(0)
    attack = Flow("http", 8080, Direction.INBOUND, bytes(rng.randint(0, 255) for _ in range(50)))
    assert detector.assess_flow(attack).blocked
