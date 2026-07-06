"""iter_balanced_packets — 두 클래스를 균형 있게 뽑는 평가 샘플러.

CSIC 혼합 CSV는 '정상 먼저, 공격 나중' 순서라 앞에서 limit개만 자르면 공격이 0개가 되는
함정이 있다. 이 샘플러는 정상·공격을 각각 약 limit/2개씩 모아 두 클래스가 항상 포함되게 한다.
"""

from csic_to_packets import iter_balanced_packets

HEADER = "classification,Method,host,content,URL"


def _write(path, rows: list[str]) -> None:
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="latin-1")


def _normal(i: int) -> str:
    return f"0,GET,localhost:8080,,http://localhost:8080/p{i} HTTP/1.1"


def _anomalous(i: int) -> str:
    return f"1,GET,localhost:8080,,http://localhost:8080/q{i} HTTP/1.1"


def test_limit_pulls_both_classes_despite_normals_first(tmp_path) -> None:
    path = tmp_path / "eval.csv"
    _write(path, [_normal(i) for i in range(30)] + [_anomalous(i) for i in range(30)])

    records = list(iter_balanced_packets(path, limit=20))

    n_normal = sum(r.is_normal for r in records)
    n_attack = sum(r.is_anomalous for r in records)
    assert n_normal == 10
    assert n_attack == 10


def test_no_limit_yields_everything(tmp_path) -> None:
    path = tmp_path / "eval.csv"
    _write(path, [_normal(i) for i in range(5)] + [_anomalous(i) for i in range(7)])

    records = list(iter_balanced_packets(path, limit=None))

    assert len(records) == 12
