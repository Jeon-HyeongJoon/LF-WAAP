"""Kafka 학습/평가/추론 파이프라인 CLI.

멘탈 모델: **데이터를 파이프(토픽)에 넣으면 → 학습/평가/추론이 돈다.**

서브커맨드
  produce-train   CSIC 정상 CSV  → waf.train.normal / waf.train.validation 토픽
  produce-eval    CSIC 혼합 CSV  → waf.eval.labeled 토픽(정답 라벨 포함)
  train           학습 토픽 소비 → 모델 학습·보정 → .pkl 저장
  evaluate        평가 토픽 소비 → 저장 모델 채점 → 지표 출력
  demo            브로커 없이(InMemoryChannel) 전체 루프를 한 프로세스에서 시연

실제 브로커가 필요한 명령(produce-*/train/evaluate)은 --bootstrap 으로 주소를 받는다.
먼저  `docker compose up -d`  로 Kafka를 띄운다. 설치:  pip install -e ".[kafka]".

이 파일은 '배선(wiring)'만 담당한다 — 학습/평가/추론 로직은 모두 src/waf 안에서 이미
테스트된 컴포넌트이고, 여기서는 CSIC 어댑터와 파이프 어댑터를 연결할 뿐이다.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import cast

from csic_to_packets import iter_balanced_packets
from csic_traffic_source import CsicTrafficSource, record_to_flow

from waf.application.calibrate_model import CalibrateModel
from waf.domain.model.flow import Direction, Flow
from waf.domain.model.http_request import HttpRequest
from waf.domain.service.block_decision_service import BlockDecisionService
from waf.infrastructure.adapter.persistence import PickleModelRepository
from waf.infrastructure.hmm import HmmDetector, ModelVariant
from waf.infrastructure.hmm.benchmark import ScoredMetrics
from waf.infrastructure.ruleset.ruleset_detector import RuleSetDetector
from waf.infrastructure.streaming.channel import MessageChannel
from waf.infrastructure.streaming.codecs import encode_flow, encode_labeled_flow, encode_request
from waf.infrastructure.streaming.evaluation import StreamingLabeledSource, evaluate_stream
from waf.infrastructure.streaming.inference import StreamingInspector
from waf.infrastructure.streaming.memory import InMemoryChannel
from waf.infrastructure.streaming.traffic_source import (
    DEFAULT_NORMAL_TOPIC,
    DEFAULT_VALIDATION_TOPIC,
    StreamingTrafficSource,
)

_DATA_DIR = Path(__file__).parent
_DEFAULT_NORMAL = _DATA_DIR / "csic_database_normal.csv"
_DEFAULT_EVAL = _DATA_DIR / "csic_database.csv"
_DEFAULT_MODEL = _DATA_DIR / "hmm_model.pkl"
_EVAL_TOPIC = "waf.eval.labeled"


# --- 채널 선택 ----------------------------------------------------------------


def make_channel(bootstrap: str) -> MessageChannel:
    """실제 Kafka 채널을 만든다(kafka-python 필요)."""
    from waf.infrastructure.streaming.kafka_channel import KafkaChannel

    return KafkaChannel(bootstrap_servers=bootstrap)


# --- producer: 데이터를 파이프에 넣기 ----------------------------------------


def produce_train(
    channel: MessageChannel,
    normal_csv: str | Path,
    *,
    validation_ratio: float = 0.2,
    limit: int | None = None,
) -> tuple[int, int]:
    """CSIC 정상 트래픽을 학습/검증 토픽으로 발행. (정상수, 검증수) 반환."""
    source = CsicTrafficSource(normal_csv, validation_ratio=validation_ratio, limit=limit)
    n_normal = 0
    for flow in source.normal():
        channel.publish(DEFAULT_NORMAL_TOPIC, encode_flow(flow))
        n_normal += 1
    n_val = 0
    for flow in source.validation():
        channel.publish(DEFAULT_VALIDATION_TOPIC, encode_flow(flow))
        n_val += 1
    return n_normal, n_val


def produce_eval(
    channel: MessageChannel,
    eval_csv: str | Path,
    *,
    limit: int | None = None,
) -> int:
    """CSIC 혼합 트래픽을 라벨과 함께 평가 토픽으로 발행. 발행 건수 반환."""
    count = 0
    for record in iter_balanced_packets(eval_csv, limit):
        if not (record.is_normal or record.is_anomalous):
            continue
        message = encode_labeled_flow(record_to_flow(record), is_attack=record.is_anomalous)
        channel.publish(_EVAL_TOPIC, message)
        count += 1
    return count


# --- 파이프에서 학습 / 평가 --------------------------------------------------


def train_from_pipe(
    channel: MessageChannel,
    model_path: str | Path,
    variant: ModelVariant,
) -> tuple[int, int]:
    """학습 토픽을 소비해 모델을 학습·보정하고 저장한다. (정상수, 검증수) 반환.

    학습 코드는 기존 CalibrateModel 그대로 — TrafficSource를 StreamingTrafficSource로
    갈아끼웠을 뿐이다(포트/어댑터의 핵심).
    """
    source = StreamingTrafficSource(channel)
    repository = PickleModelRepository(model_path)
    report = CalibrateModel(source, repository)(variant.build_detector())
    return report.normal_count, report.validation_count


def evaluate_from_pipe(
    channel: MessageChannel,
    model_path: str | Path,
    *,
    target_fpr: float = 0.05,
) -> ScoredMetrics:
    """평가 토픽을 소비해 저장된 모델을 채점한다."""
    detector = cast(HmmDetector, PickleModelRepository(model_path).load())
    return evaluate_stream(detector, StreamingLabeledSource(channel), target_fpr=target_fpr)


# --- 출력 --------------------------------------------------------------------


def _print_scored(metrics: ScoredMetrics, target_fpr: float) -> None:
    print(f"[evaluate] 커버 {metrics.covered} / 미커버 {metrics.uncovered}, "
          f"정상 {metrics.eval_normal} / 공격 {metrics.eval_attack}")
    if metrics.report is None or metrics.classification is None:
        print("[evaluate] 두 클래스 표본이 부족해 지표 계산 불가")
        return
    r, c = metrics.report, metrics.classification
    print(f"[evaluate] ROC-AUC={r.roc_auc:.3f}  AUCp@{target_fpr:.0%}={r.partial_auc:.3f}  "
          f"TPR@{target_fpr:.0%}FPR={r.tpr_at_fpr:.3f}")
    print(f"[evaluate] Accuracy={c.accuracy:.3f}  Precision={c.precision:.3f}  "
          f"Recall={c.recall:.3f}  F1={c.f1:.3f}")


# --- demo: 브로커 없이 전체 루프 시연 ---------------------------------------


def _demo_normal_flow(rng: random.Random) -> Flow:
    path = rng.choice(["/index.html", "/products", "/api/items", "/search"])
    text = (
        f"GET {path}?id={rng.randint(1, 9999)}&sort=price HTTP/1.1\r\n"
        f"Host: shop.example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
    )
    return Flow("http", 80, Direction.INBOUND, text.encode())


def _demo_attack_flow(rng: random.Random) -> Flow:
    payload = bytes(rng.randint(0, 255) for _ in range(rng.randint(90, 200)))
    return Flow("http", 80, Direction.INBOUND, payload)


def run_demo() -> None:
    """InMemoryChannel로 produce→train→evaluate→inspect 전 과정을 한 프로세스에서 보여준다.

    Kafka/도커 없이 '파이프에 넣으면 학습·평가·추론이 돈다'는 흐름을 그대로 따라갈 수 있다.
    """
    channel = InMemoryChannel()
    rng = random.Random(7)

    # ① 학습/검증 데이터를 파이프에 넣는다.
    for _ in range(80):
        channel.publish(DEFAULT_NORMAL_TOPIC, encode_flow(_demo_normal_flow(rng)))
    for _ in range(40):
        channel.publish(DEFAULT_VALIDATION_TOPIC, encode_flow(_demo_normal_flow(rng)))
    print("[demo] ① 학습/검증 Flow를 파이프에 발행")

    # ② 파이프에서 학습.
    variant = ModelVariant("demo", n_states=4, target_fpr=0.1, min_train_sequences=5, n_iter=30)
    model_path = _DATA_DIR / "_demo_model.pkl"
    n_normal, n_val = train_from_pipe(channel, model_path, variant)
    print(f"[demo] ② 파이프 학습 완료: 정상 {n_normal} / 검증 {n_val}")

    # ③ 평가용 라벨 데이터를 파이프에 넣고 채점.
    for _ in range(30):
        channel.publish(_EVAL_TOPIC, encode_labeled_flow(_demo_normal_flow(rng), is_attack=False))
    for _ in range(30):
        channel.publish(_EVAL_TOPIC, encode_labeled_flow(_demo_attack_flow(rng), is_attack=True))
    metrics = evaluate_from_pipe(channel, model_path, target_fpr=0.1)
    print("[demo] ③ 파이프 평가:")
    _print_scored(metrics, target_fpr=0.1)

    # ④ 실시간 추론: 요청을 파이프에 넣고 판정을 받는다(룰셋 + 학습된 HMM 조합).
    detector = cast(HmmDetector, PickleModelRepository(model_path).load())
    service = BlockDecisionService([RuleSetDetector.default(), detector])
    channel.publish("waf.inspect.requests", encode_request(HttpRequest("GET", "/products", "id=42")))
    channel.publish(
        "waf.inspect.requests",
        encode_request(HttpRequest("GET", "/items", "id=1 OR 1=1")),
    )
    processed = StreamingInspector(channel, service).run()
    print(f"[demo] ④ 실시간 추론: 요청 {processed}건 판정 → waf.inspect.verdicts")
    from waf.infrastructure.streaming.codecs import decode_verdict

    for verdict in (decode_verdict(m) for m in channel.poll("waf.inspect.verdicts")):
        mark = "BLOCK" if verdict.blocked else "ALLOW"
        print(f"          [{mark}] {verdict.reason}")

    model_path.unlink(missing_ok=True)


# --- CLI ---------------------------------------------------------------------


def _add_variant_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--extractor", default="byte_class",
                   choices=["byte_class", "raw_byte", "window_token"])
    p.add_argument("--n-states", type=int, default=8)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--ensemble", type=int, default=1)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--n-iter", type=int, default=100)


def _variant_from_args(args: argparse.Namespace) -> ModelVariant:
    return ModelVariant(
        name="cli",
        extractor=args.extractor,
        n_states=args.n_states,
        window=args.window,
        ensemble_size=args.ensemble,
        target_fpr=args.target_fpr,
        n_iter=args.n_iter,
    )


def main() -> None:
    # 한글/박스 글자 출력이 cp949 콘솔에서 깨지지 않도록 UTF-8 강제.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="LF-WAAP Kafka 학습/평가/추론 파이프라인")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka 부트스트랩 주소")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pt = sub.add_parser("produce-train", help="CSIC 정상 → 학습 토픽")
    p_pt.add_argument("--normal-csv", default=str(_DEFAULT_NORMAL))
    p_pt.add_argument("--validation-ratio", type=float, default=0.2)
    p_pt.add_argument("--limit", type=int, default=None)

    p_pe = sub.add_parser("produce-eval", help="CSIC 혼합 → 평가 토픽")
    p_pe.add_argument("--eval-csv", default=str(_DEFAULT_EVAL))
    p_pe.add_argument("--limit", type=int, default=None)

    p_tr = sub.add_parser("train", help="학습 토픽 소비 → 모델 저장")
    p_tr.add_argument("--model", default=str(_DEFAULT_MODEL))
    _add_variant_args(p_tr)

    p_ev = sub.add_parser("evaluate", help="평가 토픽 소비 → 모델 채점")
    p_ev.add_argument("--model", default=str(_DEFAULT_MODEL))
    p_ev.add_argument("--target-fpr", type=float, default=0.05)

    sub.add_parser("demo", help="브로커 없이 전체 루프 시연(InMemoryChannel)")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
        return

    if args.command == "produce-train":
        channel = make_channel(args.bootstrap)
        n_normal, n_val = produce_train(
            channel, args.normal_csv, validation_ratio=args.validation_ratio, limit=args.limit
        )
        print(f"[produce-train] 정상 {n_normal} → {DEFAULT_NORMAL_TOPIC}, "
              f"검증 {n_val} → {DEFAULT_VALIDATION_TOPIC}")
    elif args.command == "produce-eval":
        channel = make_channel(args.bootstrap)
        count = produce_eval(channel, args.eval_csv, limit=args.limit)
        print(f"[produce-eval] 라벨 {count}건 → {_EVAL_TOPIC}")
    elif args.command == "train":
        channel = make_channel(args.bootstrap)
        n_normal, n_val = train_from_pipe(channel, args.model, _variant_from_args(args))
        print(f"[train] 정상 {n_normal} / 검증 {n_val}로 학습·보정 후 저장 → {args.model}")
    elif args.command == "evaluate":
        channel = make_channel(args.bootstrap)
        _print_scored(
            evaluate_from_pipe(channel, args.model, target_fpr=args.target_fpr),
            target_fpr=args.target_fpr,
        )


if __name__ == "__main__":
    main()
